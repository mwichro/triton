// Eliminate the post-trans `ttg.convert_layout` that appears when a
// `tt.trans` op is followed (possibly through layout-preserving ops) by a
// layout conversion to a store-friendly layout. Such a chain shows up after
// CoalescePass + RemoveLayoutConversions whenever the coalesced load layout
// L_load is not compatible with the coalesced store layout S_store under the
// permutation `perm` (i.e., inferTrans(L_load, perm) != S_store). The leftover
// convert is the offending cross-warp/cross-block conversion that lowers to
// shared memory.
//
// The fix is to propagate `S_store` backward through the trans op: pick
// L_new = inferTrans(S_store, inv_perm) and retype the upstream slice
// (load, pointer arithmetic, elementwise ops between load and trans) to use
// L_new. After retyping, the trans naturally produces `S_store`, the
// downstream convert becomes a no-op, and the shared-memory round-trip is
// gone.
//
// RemoveLayoutConversions cannot do this because it refuses to retype
// "expensive" loads. This pass is targeted: it only kicks in when the slice
// crosses exactly one `tt.trans` op, and only when the entire upstream slice
// is composed of retypeable ops (loads, splats, make_range, broadcasts,
// elementwise ops, etc.).

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-optimize-trans-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZETRANSLAYOUT
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;

// Whether `op`'s result encoding can be freely changed in place. Unlike
// RemoveLayoutConversions::canBeRemat, we allow expensive loads — we are
// retyping the existing op rather than cloning it, so no extra memory traffic
// is generated. Note this is intentionally a narrow whitelist; ops we don't
// recognise cause the rewrite to bail out.
bool isInPlaceRetypeable(Operation *op) {
  // Layout-changing ops (their result encoding is derived from operand
  // encoding via inferSrcEncoding/inferDstEncoding).
  if (isa<tt::ExpandDimsOp, tt::BroadcastOp, tt::ReshapeOp, tt::JoinOp,
          tt::SplitOp, tt::TransOp>(op))
    return true;
  // Roots that can take any layout.
  if (isa<tt::LoadOp, tt::SplatOp, tt::MakeRangeOp, arith::ConstantOp>(op))
    return true;
  // Pointer arithmetic and elementwise ops preserve layout.
  if (isa<tt::AddPtrOp>(op))
    return true;
  if (op->hasTrait<OpTrait::Elementwise>() ||
      op->hasTrait<OpTrait::SameOperandsAndResultEncoding>())
    return true;
  return false;
}

// Walk backward from `root` (the value that should now have encoding
// `targetEnc`) and collect a slice of values whose encoding must be updated.
// `layoutMap` is populated with each value's new encoding.
// Returns failure if the slice cannot be fully retyped in place.
LogicalResult collectRetypeSlice(Value root, Attribute targetEnc,
                                 llvm::SetVector<Value> &slice,
                                 llvm::DenseMap<Value, Attribute> &layoutMap) {
  SmallVector<std::pair<Value, Attribute>> worklist;
  worklist.push_back({root, targetEnc});

  while (!worklist.empty()) {
    auto [v, enc] = worklist.pop_back_val();
    auto tensorTy = dyn_cast<RankedTensorType>(v.getType());
    if (!tensorTy)
      continue;

    // If we already visited this value with a different encoding, the
    // retyping is inconsistent and we must bail.
    auto it = layoutMap.find(v);
    if (it != layoutMap.end()) {
      if (it->second != enc) {
        LDBG("conflicting encodings for value " << v);
        return failure();
      }
      continue;
    }

    // If the value already has the target encoding, nothing to do; do not
    // recurse further along this path.
    if (tensorTy.getEncoding() == enc) {
      layoutMap[v] = enc;
      continue;
    }

    layoutMap[v] = enc;
    slice.insert(v);

    Operation *def = v.getDefiningOp();
    if (!def) {
      LDBG("hit block argument; bailing");
      return failure();
    }
    if (!isInPlaceRetypeable(def)) {
      LDBG("non-retypeable op in slice: " << *def);
      return failure();
    }

    // Walk through each tensor operand. For trans/expand_dims/etc. the
    // operand encoding differs from the result encoding; inferSrcEncoding
    // handles all supported cases.
    for (OpOperand &operandUse : def->getOpOperands()) {
      Value operand = operandUse.get();
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      Attribute srcEnc = inferSrcEncoding(def, enc);
      if (!srcEnc) {
        LDBG("inferSrcEncoding failed for " << *def);
        return failure();
      }
      worklist.push_back({operand, srcEnc});
    }
  }
  return success();
}

// Verify that retyping the slice won't break any uses outside the slice.
// A use outside the slice would need a new ConvertLayoutOp to convert back
// to the original encoding, which defeats the optimization. For now we
// require that every value in the slice is only consumed by other values in
// the slice plus (optionally) the `terminalConvert` we are removing.
bool isSliceClosed(const llvm::SetVector<Value> &slice,
                   ttg::ConvertLayoutOp terminalConvert) {
  for (Value v : slice) {
    for (Operation *user : v.getUsers()) {
      if (user == terminalConvert)
        continue;
      // The user must produce a value that's also in the slice.
      bool userInSlice = false;
      for (Value r : user->getResults()) {
        if (slice.count(r)) {
          userInSlice = true;
          break;
        }
      }
      if (!userInSlice) {
        LDBG("value " << v << " has out-of-slice user " << *user);
        return false;
      }
    }
  }
  return true;
}

// Retype `arith.constant` in place, including its `value` attribute.
// Only supports DenseElementsAttr (the only kind of tensor constant Triton
// emits). For splats the new attribute is reconstructed directly; for non-
// splat data we rebuild via the raw byte buffer (encoding doesn't affect the
// canonical row-major byte layout used by DenseElementsAttr).
LogicalResult retypeConstant(arith::ConstantOp constOp, Attribute newEnc) {
  auto rtt = dyn_cast<RankedTensorType>(constOp.getType());
  if (!rtt)
    return failure();
  auto newType = rtt.cloneWithEncoding(newEnc);
  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr)
    return failure();
  DenseElementsAttr newAttr;
  if (denseAttr.isSplat()) {
    newAttr = DenseElementsAttr::get(cast<ShapedType>(newType),
                                     denseAttr.getSplatValue<Attribute>());
  } else {
    // Non-splat: reuse raw data (canonical layout is encoding-independent).
    newAttr = denseAttr.reshape(cast<ShapedType>(newType));
  }
  constOp.setValueAttr(newAttr);
  constOp.getResult().setType(newType);
  return success();
}

// Apply the retyping in topological order. Returns failure on the first op
// that cannot be retyped (e.g. an unsupported constant attribute).
LogicalResult
applyRetyping(const llvm::SetVector<Value> &slice,
              const llvm::DenseMap<Value, Attribute> &layoutMap) {
  llvm::SetVector<Operation *> opsToRetype;
  for (Value v : slice) {
    if (Operation *def = v.getDefiningOp())
      opsToRetype.insert(def);
  }
  auto sorted = mlir::topologicalSort(opsToRetype);
  for (Operation *op : sorted) {
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      auto it = layoutMap.find(constOp.getResult());
      if (it == layoutMap.end())
        continue;
      if (failed(retypeConstant(constOp, it->second)))
        return failure();
      continue;
    }
    for (Value result : op->getResults()) {
      auto it = layoutMap.find(result);
      if (it == layoutMap.end())
        continue;
      auto rtt = dyn_cast<RankedTensorType>(result.getType());
      if (!rtt)
        continue;
      result.setType(rtt.cloneWithEncoding(it->second));
    }
  }
  return success();
}

// Try to eliminate `convertOp` by propagating its target encoding backward
// through an intervening `tt.trans` (and any layout-preserving ops) and
// retyping the upstream slice. Returns true on success.
bool tryOptimizeConvert(ttg::ConvertLayoutOp convertOp) {
  // For the slice walker to flip the encoding via the trans, the trans must
  // appear inside the slice. We don't need to find it explicitly here; we
  // simply start the slice walk from the convert's source with the convert's
  // target encoding, and let `inferSrcEncoding` apply the correct
  // permutation when it hits the trans. We just require *at least one* trans
  // in the slice; otherwise this pass has nothing extra to offer over the
  // existing RemoveLayoutConversions logic.

  Value srcVal = convertOp.getSrc();
  Attribute targetEnc = convertOp.getType().getEncoding();

  llvm::SetVector<Value> slice;
  llvm::DenseMap<Value, Attribute> layoutMap;
  if (failed(collectRetypeSlice(srcVal, targetEnc, slice, layoutMap)))
    return false;
  if (slice.empty())
    return false;

  // Require at least one TransOp in the slice; otherwise existing passes
  // already handle this case (or refuse to for legitimate reasons).
  bool hasTrans = false;
  for (Value v : slice) {
    if (auto *def = v.getDefiningOp())
      if (isa<tt::TransOp>(def)) {
        hasTrans = true;
        break;
      }
  }
  if (!hasTrans)
    return false;

  // Require closed slice: no external users that would need a fix-up convert.
  if (!isSliceClosed(slice, convertOp))
    return false;

  LDBG("rewriting convert " << convertOp << " with slice of size "
                            << slice.size());

  if (failed(applyRetyping(slice, layoutMap)))
    return false;

  // After retyping, convertOp's source has the target encoding. Replace
  // its uses with the source and erase the convert.
  convertOp.getResult().replaceAllUsesWith(convertOp.getSrc());
  convertOp.erase();
  return true;
}

struct OptimizeTransLayoutPass
    : public impl::TritonGPUOptimizeTransLayoutBase<OptimizeTransLayoutPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<ttg::ConvertLayoutOp> candidates;
      mod.walk([&](ttg::ConvertLayoutOp cvt) { candidates.push_back(cvt); });
      for (ttg::ConvertLayoutOp cvt : candidates) {
        // The convert may have been erased by a previous iteration's slice.
        if (!cvt || !cvt->getParentOp())
          continue;
        if (tryOptimizeConvert(cvt))
          changed = true;
      }
    }
  }
};

} // namespace

} // namespace mlir::triton::gpu
