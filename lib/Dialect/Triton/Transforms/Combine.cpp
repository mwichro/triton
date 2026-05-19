#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/DiscardableAttributes.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONCOMBINEOPS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool isZero(Value val) {
  return (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()));
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto loadOp = trueValue.getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    auto splatOp = mask.getDefiningOp<SplatOp>();
    if (!splatOp)
      return failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return failure();

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
private:
  static bool isAddFloat(const Operation *op) {
    return isa<arith::AddFOp>(op);
  }

public:
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    // only support reduce with simple addition of floats
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddFloat(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = reduceOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = mulOp.getOperand(0).getDefiningOp<BroadcastOp>();
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = mulOp.getOperand(1).getDefiningOp<BroadcastOp>();
    if (!broadcastRhsOp)
      return failure();
    // broadcast operand is expand dims
    auto expandLhsOp = broadcastLhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandLhsOp)
      return failure();
    auto expandRhsOp = broadcastRhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandRhsOp)
      return failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastRhsOp.getType()).getShape();
    // For FP64 MMA (m8n8k4 on A100) allow sizes down to 8; otherwise use 16.
    auto elemType =
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType();
    int64_t minSize = elemType.isF64() ? 8 : 16;
    if (broadcastLhsShape[2] < minSize || broadcastRhsShape[0] < minSize)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]}, elemType);
    rewriter.setInsertionPoint(op);
    auto zeroAttr = FloatAttr::get(elemType, 0.0);
    auto newAcc =
        SplatOp::create(rewriter, op->getLoc(), newAccType,
                        arith::ConstantOp::create(rewriter, op->getLoc(),
                                                  zeroAttr));
    rewriter.replaceOpWithNewOp<DotOp>(op, expandLhsOp.getSrc(),
                                       expandRhsOp.getSrc(), newAcc,
                                       InputPrecision::IEEE, 0);
    return success();
  }
};

// sum(u[:, :, :, None, :] * W[None, None, None, :, :], last_axis)
// -> reshape(dot(reshape(u, [B, K]), trans(W, [1, 0])), [batch..., N])
//
// This handles the batched contraction pattern where:
//   u  has shape (batch..., K),   expanded at axis R-2 to (batch..., 1, K)
//   W  has shape (1,...,1, N, K), batch dims all 1
//   reduce is over the last axis (K)
// Result: (batch..., N)
//
// This avoids materialising the large intermediate product tensor
// (batch..., N, K) in registers.  Instead the contraction is emitted as a
// single tt.dot after appropriate reshapes.
class CombineBroadcastMulReduceToMatmulPattern : public RewritePattern {
private:
  static bool isAddFloat(const Operation *op) { return isa<arith::AddFOp>(op); }

  struct MatchResult {
    Value lhsSrc;      // shape: (batch..., 1, K)
    Value rhsSrc;      // shape: (1,...,1, N, K)
    int64_t N, K;
    int batchRank;
    int64_t batchTotal;
  };

  // Try to match with a specific assignment of lhs/rhs broadcast operands.
  // lhsBroadcast must be the "u" side (batch dims present, size-1 at R-2).
  // rhsBroadcast must be the "W" side (all batch dims == 1).
  static std::optional<MatchResult>
  tryMatchOrdered(BroadcastOp lhsBroadcast, BroadcastOp rhsBroadcast, int R,
                  ArrayRef<int64_t> broadcastShape) {
    int batchRank = R - 2;
    int64_t N = broadcastShape[R - 2];
    int64_t K = broadcastShape[R - 1];

    // LHS src: (batch..., 1, K) -- size 1 at axis R-2, batch dims match
    auto lhsSrcShape =
        cast<RankedTensorType>(lhsBroadcast.getSrc().getType()).getShape();
    if (static_cast<int>(lhsSrcShape.size()) != R)
      return std::nullopt;
    for (int i = 0; i < batchRank; i++)
      if (lhsSrcShape[i] != broadcastShape[i])
        return std::nullopt;
    if (lhsSrcShape[R - 2] != 1 || lhsSrcShape[R - 1] != K)
      return std::nullopt;

    // RHS src: (1,...,1, N, K) -- all batch dims must be 1
    auto rhsSrcShape =
        cast<RankedTensorType>(rhsBroadcast.getSrc().getType()).getShape();
    if (static_cast<int>(rhsSrcShape.size()) != R)
      return std::nullopt;
    for (int i = 0; i < batchRank; i++)
      if (rhsSrcShape[i] != 1)
        return std::nullopt;
    if (rhsSrcShape[R - 2] != N || rhsSrcShape[R - 1] != K)
      return std::nullopt;

    int64_t batchTotal = 1;
    for (int i = 0; i < batchRank; i++)
      batchTotal *= broadcastShape[i];

    return MatchResult{lhsBroadcast.getSrc(), rhsBroadcast.getSrc(), N, K,
                       batchRank, batchTotal};
  }

public:
  CombineBroadcastMulReduceToMatmulPattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto reduceOp = dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();

    // Only support reduce with simple float addition.
    Region &combineRegion = reduceOp.getCombineOp();
    if (!combineRegion.hasOneBlock() ||
        combineRegion.front().getOperations().size() != 2 ||
        !isAddFloat(&*combineRegion.front().getOperations().begin()))
      return failure();

    if (reduceOp.getNumOperands() != 1)
      return failure();

    auto inputType =
        dyn_cast<RankedTensorType>(reduceOp.getOperand(0).getType());
    if (!inputType)
      return failure();

    int R = inputType.getRank();
    if (R < 2)
      return failure();

    // Only handle reduce over the last axis.
    if (reduceOp.getAxis() != R - 1)
      return failure();

    // Require floating-point element type.
    auto elemType = inputType.getElementType();
    if (!isa<FloatType>(elemType))
      return failure();

    // The operand of reduce must be a mul.
    auto mulOp = reduceOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();

    // Both mul operands must come from BroadcastOp.
    auto op0 = mulOp.getOperand(0).getDefiningOp<BroadcastOp>();
    auto op1 = mulOp.getOperand(1).getDefiningOp<BroadcastOp>();
    if (!op0 || !op1)
      return failure();

    auto broadcastShape = inputType.getShape();

    // Try both orderings: (u=op0, W=op1) and (u=op1, W=op0).
    std::optional<MatchResult> match = tryMatchOrdered(op0, op1, R, broadcastShape);
    if (!match)
      match = tryMatchOrdered(op1, op0, R, broadcastShape);
    if (!match)
      return failure();

    auto [lhsSrc, rhsSrc, N, K, batchRank, batchTotal] = *match;

    // Minimum size check for dot to be beneficial.
    // FP64 MMA on SM80/SM90 uses m8n8k4, so require N>=8 and K>=4.
    // For other types use conservative thresholds matching tl.dot expectations.
    int64_t minN = elemType.isF64() ? 8 : 16;
    int64_t minK = elemType.isF64() ? 4 : 16;
    if (N < minN || K < minK || batchTotal < 1)
      return failure();

    auto loc = op->getLoc();

    // Flatten lhsSrc from (batch..., 1, K) to (batchTotal, K).
    // This squeezes the size-1 dim at R-2 and flattens any batch dims.
    int64_t effectiveBatch = (batchRank == 0) ? 1 : batchTotal;
    auto lhs2d = ReshapeOp::create(rewriter, loc,
                                   SmallVector<int64_t>{effectiveBatch, K},
                                   lhsSrc);

    // Reshape rhsSrc from (1,...,1, N, K) to (N, K), stripping leading 1s.
    auto rhsFlat =
        ReshapeOp::create(rewriter, loc, SmallVector<int64_t>{N, K}, rhsSrc);

    // Transpose rhsFlat from (N, K) to (K, N).
    auto rhsT = TransOp::create(rewriter, loc, rhsFlat,
                                SmallVector<int32_t>{1, 0});

    // Accumulator: zeros of shape (effectiveBatch, N).
    auto accType = RankedTensorType::get({effectiveBatch, N}, elemType);
    auto zeroAttr = FloatAttr::get(elemType, 0.0);
    auto zeroVal = arith::ConstantOp::create(rewriter, loc, zeroAttr);
    auto acc = SplatOp::create(rewriter, loc, accType, zeroVal);

    // Emit the dot: (effectiveBatch, K) * (K, N) -> (effectiveBatch, N).
    auto result2d =
        DotOp::create(rewriter, loc, lhs2d, rhsT, acc, InputPrecision::IEEE, 0);

    // Reshape result from (effectiveBatch, N) back to (batch..., N).
    SmallVector<int64_t> resultShape(broadcastShape.begin(),
                                     broadcastShape.begin() + batchRank);
    resultShape.push_back(N);
    rewriter.replaceOpWithNewOp<ReshapeOp>(op, resultShape, result2d);
    return success();
  }
};

// When reducing a 1D tensor the order of elements of the tensor doesn't matter.
// Therefore we can relax the reshape to allow it to re-order elements.
class CombineReshapeReducePatterns : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (reshapeOp.getAllowReorder())
      return failure();
    if (reshapeOp.getType().getRank() != 1)
      return failure();
    for (Operation *user : reshapeOp->getUsers()) {
      if (!isa<triton::ReduceOp, triton::HistogramOp>(user))
        return failure();
    }
    rewriter.modifyOpInPlace(reshapeOp,
                             [&]() { reshapeOp.setAllowReorder(true); });
    return success();
  }
};

class RankedReduceDescriptorLoads : public mlir::OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp reshapeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto loadDef = reshapeOp.getSrc().getDefiningOp<triton::DescriptorLoadOp>();
    if (!loadDef || !loadDef->hasOneUse())
      return failure();
    int loadRank = loadDef.getType().getRank();
    int reshapeRank = reshapeOp.getType().getRank();
    if (!(reshapeRank < loadRank))
      return failure();
    ArrayRef<int64_t> loadShape = loadDef.getType().getShape();
    ArrayRef<int64_t> reshapeShape = reshapeOp.getType().getShape();
    for (int i = 0; i < loadRank - reshapeRank; ++i) {
      // Only rank reduce unit dims.
      if (loadShape[i] != 1)
        return failure();
    }
    if (loadShape.take_back(reshapeRank) != reshapeShape)
      return failure();
    rewriter.modifyOpInPlace(
        loadDef, [&]() { loadDef.getResult().setType(reshapeOp.getType()); });
    rewriter.replaceOp(reshapeOp, loadDef.getResult());
    return success();
  }
};

template <typename DotOpType, typename AddOpType>
class CombineDotAddPattern : public mlir::OpRewritePattern<AddOpType> {
public:
  using OpRewritePattern<AddOpType>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(AddOpType addOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = addOp.getRhs().template getDefiningOp<DotOpType>();
    bool isDotLHS = false;
    if (!dotOp) {
      dotOp = addOp.getLhs().template getDefiningOp<DotOpType>();
      if (!dotOp) {
        return failure();
      }
      isDotLHS = true;
    }
    if (!dotOp->hasOneUse()) {
      return failure();
    }
    if (!isZero(dotOp.getC()))
      return failure();
    if constexpr (std::is_same_v<DotOpType, DotOp> &&
                  std::is_same_v<AddOpType, arith::AddFOp>) {
      if (dotOp.getMaxNumImpreciseAcc() != 0) {
        return failure();
      }
    }
    rewriter.modifyOpInPlace(dotOp, [&] {
      dotOp.getCMutable().assign(isDotLHS ? addOp.getRhs() : addOp.getLhs());
      dotOp->moveBefore(addOp);
    });
    rewriter.replaceAllUsesWith(addOp, dotOp.getResult());
    return success();
  }
};

// AddIOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddFOp(DotOp(a, b, c), d) and c==0 => DotOp(a, b, d)
// AddIOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
// AddFOp(d, DotOp(a, b, c)) and c==0 => DotOp(a, b, d)
using CombineDotAddIPattern = CombineDotAddPattern<DotOp, arith::AddIOp>;
using CombineDotAddFPattern = CombineDotAddPattern<DotOp, arith::AddFOp>;
using CombineDotScaledAddFPattern =
    CombineDotAddPattern<DotScaledOp, arith::AddFOp>;

} // anonymous namespace

class CombineOpsPass : public impl::TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineDotScaledAddFPattern>(context);
    patterns.add<CombineSelectMaskedLoadPattern>(context);
    patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);
    patterns.add<CombineBroadcastMulReduceToMatmulPattern>(context);
    patterns.add<CombineReshapeReducePatterns>(context);
    patterns.add<RankedReduceDescriptorLoads>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir::triton
