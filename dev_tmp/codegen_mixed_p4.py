"""Codegen for the p=4 (N=5) mixed MMA + CUDA-core Laplacian kernel.

Generates a Triton kernel source file that implements the volumetric Laplacian
  L*u = (K⊗M⊗M + M⊗K⊗M + M⊗M⊗K) * u
using the split-tensor design: every (5,5,5) cube is represented as 8 tensors
indexed by a per-axis tag (c = core 0..3, t = tail = 4).

Per phase, for each of the 4 (k-tag, j-tag)-or-equivalent slabs:
  * MMA(N_MMA=4): (BE * outer * outer, 4) × (4, 8) → (M, 8). Top 4 cols = M-out,
    bot 4 cols = D-out for the output index ∈ {0..3}.
  * K-tail: outer-product add for input index = 4.
  * Output-row tail (i'=4 / j'=4 / k'=4): scalar dot (sum over the full 5-wide
    input row) using a tiny (5,)-wide operator row.

Generated artifact: a single @triton.jit function plus a wrapper. No tl.cat,
no width-8 padded intermediates — each "tail" lives in its own Triton tensor.
"""
from __future__ import annotations

import textwrap
from typing import List, Tuple

# Tag conventions per axis:
#   c = "core" indices 0..3 (width N_MMA = 4)
#   t = "tail" index 4
# A representation of a (5,5,5) cube along (axA, axB, axC) is 8 tensors,
# keyed by triples in {c, t}^3.

TAGS = ["c", "t"]
TRIPLES: List[Tuple[str, str, str]] = [
    (a, b, c) for a in TAGS for b in TAGS for c in TAGS
]


def axis_size_expr(tag: str) -> str:
    """The size of an axis given its tag, as a constexpr-friendly string."""
    return "N_MMA" if tag == "c" else "1"


def tensor_shape(name: str, axes: Tuple[str, str, str]) -> str:
    """Comment describing the logical shape of a split tensor (axes given as tags)."""
    parts = ["BE"]
    for ax_tag, ax_name in zip(axes, ["axA", "axB", "axC"]):
        parts.append(axis_size_expr(ax_tag))
    return f"({', '.join(parts)})  [{''.join(axes)}]"


# ---------------------------------------------------------------------------
# Source-code building blocks
# ---------------------------------------------------------------------------

_HEADER = '''"""Auto-generated Triton kernel: mixed MMA + CUDA-core Laplacian for p=4 (N=5).

DO NOT EDIT BY HAND — regenerate via codegen_mixed_p4.py.
"""
import triton
import triton.language as tl


@triton.jit
def _laplace_kernel_mixed_p4_codegen(
    u_ptr,
    O_low4_ptr,    # (8, 4)  stacked [M[0:4,0:4]; D[0:4,0:4]]
    O_col4_ptr,    # (8,)    stacked [M[0:4,4]; D[0:4,4]]
    M_row4_ptr,    # (5,) — M[4, :]  (5-wide, contiguous)
    D_row4_ptr,    # (5,) — D[4, :]
    out_ptr,
    num_elements,
    N: tl.constexpr,
    N_MMA: tl.constexpr,
    BLOCK_SIZE_DATA: tl.constexpr,
    BLOCK_SIZE_EXEC: tl.constexpr,
):
    pid = tl.program_id(0)
    BE: tl.constexpr = BLOCK_SIZE_EXEC
    e_off = tl.arange(0, BE)
    e_global = pid * BE + e_off
    mask_e = e_global < num_elements

    block_b = e_global // BLOCK_SIZE_DATA
    e_inblock = e_global % BLOCK_SIZE_DATA
    block_base = block_b * N * N * N * BLOCK_SIZE_DATA   # (BE,)

    # ---- Load operator tiles ----
    rO = tl.arange(0, 2 * N_MMA)
    cO = tl.arange(0, N_MMA)
    O_low4 = tl.load(O_low4_ptr + rO[:, None] * N_MMA + cO[None, :])   # (8, 4)
    O_low4_T = tl.trans(O_low4)                                          # (4, 8)
    O_col4 = tl.load(O_col4_ptr + rO)                                    # (8,)
    # M_row4, D_row4: full 5 values, loaded into (4,)+scalar pieces
    mrow_c = tl.load(M_row4_ptr + tl.arange(0, N_MMA))     # M[4, 0..3]   (4,)
    mrow_t = tl.load(M_row4_ptr + 4)                       # M[4, 4]      scalar
    drow_c = tl.load(D_row4_ptr + tl.arange(0, N_MMA))     # (4,)
    drow_t = tl.load(D_row4_ptr + 4)                       # scalar
'''


def _gen_load_block() -> str:
    """Generate the load section.

    NOTE: an alternative one-big-load + register-split version was attempted
    but ran ~35% slower (1.48 vs 2.23 TFLOPS). The split/permute Triton ops
    hit shared memory and cost more than the load-fragmentation savings. The
    8-load version below is the current best — each load goes straight into
    registers in the right layout.
    """
    if True:
        # Restored per-tag 8-load implementation.
        lines = []
        lines.append("")
        lines.append("    # =================================================================")
        lines.append("    # 8 per-tag tl.load calls, each immediately early-permuted to BE-first.")
        lines.append("    # =================================================================")
        lines.append("    k_c = tl.arange(0, N_MMA)")
        lines.append("    j_c = tl.arange(0, N_MMA)")
        lines.append("    i_c = tl.arange(0, N_MMA)")
        lines.append("")
        # Emit per-tag load with BE in the last slot of offsets, then early permute to BE-first.
        for k_t, j_t, i_t in TRIPLES:
            spatial_dims = []
            if k_t == "c":
                spatial_dims.append("k")
            if j_t == "c":
                spatial_dims.append("j")
            if i_t == "c":
                spatial_dims.append("i")
            n_alive = len(spatial_dims)
            rank = n_alive + 1
            e_pos = n_alive

            def slot_for_axis(pos: int, _r=rank) -> str:
                slots = ["None"] * _r
                slots[pos] = ":"
                return "[" + ", ".join(slots) + "]"

            axis_position = {}
            cursor = 0
            for ax in ["k", "j", "i"]:
                tag = {"k": k_t, "j": j_t, "i": i_t}[ax]
                if tag == "c":
                    axis_position[ax] = cursor
                    cursor += 1
            parts = []
            parts.append(f"block_base{slot_for_axis(e_pos)}")
            if k_t == "c":
                parts.append(f"k_c{slot_for_axis(axis_position['k'])} * N * N * BLOCK_SIZE_DATA")
            else:
                parts.append("4 * N * N * BLOCK_SIZE_DATA")
            if j_t == "c":
                parts.append(f"j_c{slot_for_axis(axis_position['j'])} * N * BLOCK_SIZE_DATA")
            else:
                parts.append("4 * N * BLOCK_SIZE_DATA")
            if i_t == "c":
                parts.append(f"i_c{slot_for_axis(axis_position['i'])} * BLOCK_SIZE_DATA")
            else:
                parts.append("4 * BLOCK_SIZE_DATA")
            parts.append(f"e_inblock{slot_for_axis(e_pos)}")
            s = f"{k_t}{j_t}{i_t}"
            lines.append(f"    offs_{s} = (" + " + ".join(parts) + ")")
            lines.append(f"    mask_{s} = mask_e{slot_for_axis(e_pos)}")
            if n_alive == 0:
                lines.append(f"    u_{s} = tl.load(u_ptr + offs_{s}, mask=mask_{s}, other=0.0)")
            else:
                lines.append(f"    _raw = tl.load(u_ptr + offs_{s}, mask=mask_{s}, other=0.0)")
                perm_to_be_first = tuple([n_alive] + list(range(n_alive)))
                lines.append(f"    u_{s} = tl.permute(_raw, {perm_to_be_first})  # BE → slot 0")
        return "\n".join(lines)
    # (unreachable — kept for reference of the big-load attempt)
    lines = []
    lines.append("")
    lines.append("    # =================================================================")
    lines.append("    # One big tl.load + register-side carve into 8 split tensors")
    lines.append("    # =================================================================")
    lines.append("    N_PAD: tl.constexpr = 8")
    lines.append("    p_idx = tl.arange(0, N_PAD)         # (8,)")
    lines.append("    mask_p = p_idx < N")
    lines.append("    safe_p = tl.where(mask_p, p_idx, 0)")
    lines.append("")
    lines.append("    offs_full = (block_base[None, None, None, :] +")
    lines.append("                 safe_p[:, None, None, None] * N * N * BLOCK_SIZE_DATA +")
    lines.append("                 safe_p[None, :, None, None] * N * BLOCK_SIZE_DATA +")
    lines.append("                 safe_p[None, None, :, None] * BLOCK_SIZE_DATA +")
    lines.append("                 e_inblock[None, None, None, :])")
    lines.append("    spatial_mask = (mask_p[:, None, None, None] &")
    lines.append("                    mask_p[None, :, None, None] &")
    lines.append("                    mask_p[None, None, :, None])")
    lines.append("    load_mask = spatial_mask & mask_e[None, None, None, :]")
    lines.append("    u_raw = tl.load(u_ptr + offs_full, mask=load_mask, other=0.0)  "
                 "# (k=8, j=8, i=8, BE)")
    lines.append("    # EARLY PERMUTE: BE → slot 0   shape (BE, 8, 8, 8)")
    lines.append("    u_full = tl.permute(u_raw, (3, 0, 1, 2))")
    lines.append("")
    # Helper macros emitted inline below.
    # Split-half-of-4: take a tensor whose LAST axis = 4 (with lane 0 = real value,
    # lanes 1..3 = masked zeros). Extract lane 0 as a tensor with the last axis removed.
    # Sequence: reshape (...,4) → (...,2,2); split last (sub-lane) → 2 × (...,2),
    # take first (= lanes 0,1); split last (lane) → 2 × (...), take first (= lane 0).
    lines.append("    # ---- Split i axis (last): width 8 → low (i∈0..3) + high (i=4 in lane 0) ----")
    lines.append("    _t = tl.reshape(u_full, (BE, N_PAD, N_PAD, 2, N_MMA))   # (BE,8,8,2,4)")
    lines.append("    _t = tl.permute(_t, (0, 1, 2, 4, 3))                     # (BE,8,8,4,2)")
    lines.append("    u_xx_L, u_xx_H4 = tl.split(_t)                            # each (BE,8,8,4)")
    lines.append("    # Lane-0 of u_xx_H4 (size-4) → (BE, 8, 8)")
    lines.append("    _h = tl.reshape(u_xx_H4, (BE, N_PAD, N_PAD, 2, 2))       # (BE,8,8,2,2)")
    lines.append("    _h_a, _ = tl.split(_h)                                     # (BE,8,8,2)  ← inner=sub-lane; pair-0 = lanes (0,1)")
    lines.append("    # Wait: split removes LAST axis; we need to know which axis is which.")
    lines.append("    # In (BE,8,8,2_outer, 2_inner) reshape of size-4: outer=pair, inner=sub-lane.")
    lines.append("    # tl.split removes the LAST (inner) axis → 2 tensors of (BE,8,8,2_outer).")
    lines.append("    # So _h_a = sub-lane 0 across both pairs = lanes (0, 2). NOT what we want.")
    lines.append("    # Re-do: permute to put pair-axis LAST first, then split twice.")
    lines.append("    _h = tl.permute(_h, (0, 1, 2, 4, 3))                      # (BE,8,8,2_inner, 2_outer=pair)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,8,8,2_inner) ← pair 0 = lanes (0,1)")
    lines.append("    u_xx_H, _ = tl.split(_h_pair0)                             # (BE,8,8) ← sub-lane 0 = lane 0")
    lines.append("")
    lines.append("    # ---- Split j axis of both u_xx_L (i=c) and u_xx_H (i=t) ----")
    lines.append("    # j-split of u_xx_L: (BE,8,8,4) → low (BE,8,4,4), high4 (BE,8,4_pos_j_high,4_i)")
    lines.append("    _t = tl.reshape(u_xx_L, (BE, N_PAD, 2, N_MMA, N_MMA))    # (BE,8,2,4,4)")
    lines.append("    _t = tl.permute(_t, (0, 1, 3, 4, 2))                      # (BE,8,4_inner_j,4_i,2_outer_j)")
    lines.append("    u_xLL, u_xLH4 = tl.split(_t)                               # each (BE,8,4_pos_in_j_half,4_i)")
    lines.append("    # Lane-0 extract of axis 2 (= 4_pos_in_j_high_half) → (BE,8,4_i)")
    lines.append("    _h = tl.permute(u_xLH4, (0, 1, 3, 2))                     # (BE,8,4_i,4_pos_j_high) ← put target axis last")
    lines.append("    _h = tl.reshape(_h, (BE, N_PAD, N_MMA, 2, 2))            # (BE,8,4_i,2_outer,2_inner)")
    lines.append("    _h = tl.permute(_h, (0, 1, 2, 4, 3))                      # (BE,8,4_i,2_inner,2_outer)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,8,4_i,2_inner)")
    lines.append("    u_xLH, _ = tl.split(_h_pair0)                              # (BE,8,4_i)")
    lines.append("")
    lines.append("    # j-split of u_xx_H: (BE,8,8) → low (BE,8,4), high4 (BE,8,4_pos_j_high)")
    lines.append("    _t = tl.reshape(u_xx_H, (BE, N_PAD, 2, N_MMA))           # (BE,8,2,4)")
    lines.append("    _t = tl.permute(_t, (0, 1, 3, 2))                         # (BE,8,4_inner,2_outer)")
    lines.append("    u_xHL, u_xHH4 = tl.split(_t)                               # each (BE,8,4_pos_in_j_half)")
    lines.append("    # Lane-0 extract of last axis → (BE,8)")
    lines.append("    _h = tl.reshape(u_xHH4, (BE, N_PAD, 2, 2))               # (BE,8,2_outer,2_inner)")
    lines.append("    _h = tl.permute(_h, (0, 1, 3, 2))                         # (BE,8,2_inner,2_outer)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,8,2_inner)")
    lines.append("    u_xHH, _ = tl.split(_h_pair0)                              # (BE,8)")
    lines.append("")
    lines.append("    # ---- Split k axis on each of the 4 jL/jH × iL/iH tiles ----")
    lines.append("    # k-split of u_xLL: (BE,8,4,4) → ccc (BE,4,4,4), tcc (BE,4,4)")
    lines.append("    _t = tl.reshape(u_xLL, (BE, 2, N_MMA, N_MMA, N_MMA))     # (BE,2,4,4,4)")
    lines.append("    _t = tl.permute(_t, (0, 2, 3, 4, 1))                      # (BE,4_pos_k,4_j,4_i,2_outer_k)")
    lines.append("    u_ccc, u_xLL_H4 = tl.split(_t)                             # each (BE,4_pos_k,4_j,4_i)")
    lines.append("    # Lane-0 of u_xLL_H4 along axis 1 (= 4_pos_k_high) → (BE,4_j,4_i)")
    lines.append("    _h = tl.permute(u_xLL_H4, (0, 2, 3, 1))                   # (BE,4_j,4_i,4_pos_k_high)")
    lines.append("    _h = tl.reshape(_h, (BE, N_MMA, N_MMA, 2, 2))            # (BE,4_j,4_i,2_outer,2_inner)")
    lines.append("    _h = tl.permute(_h, (0, 1, 2, 4, 3))                      # (BE,4_j,4_i,2_inner,2_outer)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,4_j,4_i,2_inner)")
    lines.append("    u_tcc, _ = tl.split(_h_pair0)                              # (BE,4,4)")
    lines.append("")
    lines.append("    # k-split of u_xLH (= i-low, j=tail → ctc & ttc)")
    lines.append("    _t = tl.reshape(u_xLH, (BE, 2, N_MMA, N_MMA))            # (BE,2,4,4)")
    lines.append("    _t = tl.permute(_t, (0, 2, 3, 1))                         # (BE,4_pos_k,4_i,2_outer_k)")
    lines.append("    u_ctc, u_xLH_H4 = tl.split(_t)                             # each (BE,4_pos_k,4_i)")
    lines.append("    _h = tl.permute(u_xLH_H4, (0, 2, 1))                      # (BE,4_i,4_pos_k_high)")
    lines.append("    _h = tl.reshape(_h, (BE, N_MMA, 2, 2))                   # (BE,4_i,2_outer,2_inner)")
    lines.append("    _h = tl.permute(_h, (0, 1, 3, 2))                         # (BE,4_i,2_inner,2_outer)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,4_i,2_inner)")
    lines.append("    u_ttc, _ = tl.split(_h_pair0)                              # (BE,4)")
    lines.append("")
    lines.append("    # k-split of u_xHL (= i=tail, j-low → cct & tct)")
    lines.append("    _t = tl.reshape(u_xHL, (BE, 2, N_MMA, N_MMA))            # (BE,2,4,4)")
    lines.append("    _t = tl.permute(_t, (0, 2, 3, 1))                         # (BE,4_pos_k,4_i,2_outer_k)")
    lines.append("    u_cct, u_xHL_H4 = tl.split(_t)                             # each (BE,4_pos_k,4_i)")
    lines.append("    _h = tl.permute(u_xHL_H4, (0, 2, 1))                      # (BE,4_i,4_pos_k_high)")
    lines.append("    _h = tl.reshape(_h, (BE, N_MMA, 2, 2))                   # (BE,4_i,2_outer,2_inner)")
    lines.append("    _h = tl.permute(_h, (0, 1, 3, 2))                         # (BE,4_i,2_inner,2_outer)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,4_i,2_inner)")
    lines.append("    u_tct, _ = tl.split(_h_pair0)                              # (BE,4)")
    lines.append("")
    lines.append("    # k-split of u_xHH: (BE,8) → ctt (BE,4), ttt (BE,)")
    lines.append("    _t = tl.reshape(u_xHH, (BE, 2, N_MMA))                   # (BE,2,4)")
    lines.append("    _t = tl.permute(_t, (0, 2, 1))                            # (BE,4_pos_k,2_outer_k)")
    lines.append("    u_ctt, u_xHH_H4 = tl.split(_t)                             # each (BE,4_pos_k)")
    lines.append("    # Lane-0 of u_xHH_H4 along its only axis → (BE,)")
    lines.append("    _h = tl.reshape(u_xHH_H4, (BE, 2, 2))                    # (BE,2_outer,2_inner)")
    lines.append("    _h = tl.permute(_h, (0, 2, 1))                            # (BE,2_inner,2_outer)")
    lines.append("    _h_pair0, _ = tl.split(_h)                                 # (BE,2_inner)")
    lines.append("    u_ttt, _ = tl.split(_h_pair0)                              # (BE,)")
    lines.append("")
    # Emit per-tag offset and mask vars so the existing store code can reuse them.
    lines.append("    # ---- Per-tag offset/mask vars (for the store path) ----")
    lines.extend(_gen_per_tag_offsets())
    return "\n".join(lines)


def _gen_per_tag_offsets() -> List[str]:
    """Emit `offs_<tag>` and `mask_<tag>` for each of the 8 split tensors,
    matching the existing store layout (BE in the LAST slot)."""
    lines = []
    for k_t, j_t, i_t in TRIPLES:
        spatial_dims = []
        if k_t == "c":
            spatial_dims.append("k")
        if j_t == "c":
            spatial_dims.append("j")
        if i_t == "c":
            spatial_dims.append("i")
        n_alive = len(spatial_dims)
        rank = n_alive + 1
        e_pos = n_alive

        def slot_for_axis(pos: int) -> str:
            slots = ["None"] * rank
            slots[pos] = ":"
            return "[" + ", ".join(slots) + "]"

        axis_position = {}
        cursor = 0
        for ax in ["k", "j", "i"]:
            tag = {"k": k_t, "j": j_t, "i": i_t}[ax]
            if tag == "c":
                axis_position[ax] = cursor
                cursor += 1
        parts = []
        parts.append(f"block_base{slot_for_axis(e_pos)}")
        if k_t == "c":
            parts.append(f"k_c{slot_for_axis(axis_position['k'])} * N * N * BLOCK_SIZE_DATA")
        else:
            parts.append("4 * N * N * BLOCK_SIZE_DATA")
        if j_t == "c":
            parts.append(f"j_c{slot_for_axis(axis_position['j'])} * N * BLOCK_SIZE_DATA")
        else:
            parts.append("4 * N * BLOCK_SIZE_DATA")
        if i_t == "c":
            parts.append(f"i_c{slot_for_axis(axis_position['i'])} * BLOCK_SIZE_DATA")
        else:
            parts.append("4 * BLOCK_SIZE_DATA")
        parts.append(f"e_inblock{slot_for_axis(e_pos)}")
        s = f"{k_t}{j_t}{i_t}"
        lines.append(f"    offs_{s} = (" + " + ".join(parts) + ")")
        lines.append(f"    mask_{s} = mask_e{slot_for_axis(e_pos)}")
    # Also need k_c, j_c, i_c — defined here for clarity
    lines.insert(0, "    k_c = tl.arange(0, N_MMA)")
    lines.insert(1, "    j_c = tl.arange(0, N_MMA)")
    lines.insert(2, "    i_c = tl.arange(0, N_MMA)")
    return lines
    # Build offsets for each of the 8 split tensors.
    # axes order: (k, j, i, e)
    for k_t, j_t, i_t in TRIPLES:
        name = f"u_{k_t}{j_t}{i_t}"
        # Shape: BE × (4 if c else nothing) × ...
        # We always end with e-axis innermost in the loaded shape; final order: (axes... , BE).
        # For load we use 4D offsets where each "tail" axis collapses to a single index 4.
        # Build offset expression
        # k contribution
        if k_t == "c":
            k_expr = "k_c[:, None, None, None] * N * N * BLOCK_SIZE_DATA"
            k_dim = True
        else:
            k_expr = "4 * N * N * BLOCK_SIZE_DATA"
            k_dim = False
        if j_t == "c":
            j_expr = "j_c[None, :, None, None] * N * BLOCK_SIZE_DATA" if k_dim else \
                     "j_c[:, None, None] * N * BLOCK_SIZE_DATA"
            j_dim = True
        else:
            j_expr = "4 * N * BLOCK_SIZE_DATA"
            j_dim = False
        # Figure out broadcasting shape for the i and e parts depending on which axes are "alive"
        # We always end up with the e axis innermost; the spatial axes that are alive carry a dim.
        # Easiest: build expression manually for each (k,j) combo.

        # Determine the spatial-axes shape (the broadcasted axes)
        spatial_dims = []
        if k_t == "c":
            spatial_dims.append("k")
        if j_t == "c":
            spatial_dims.append("j")
        if i_t == "c":
            spatial_dims.append("i")
        # We'll always have BE on the right end as a column for indexing into e_inblock.

        # Number of "spatial" axes that are alive.  For LOAD we keep BE in the
        # LAST slot (matches BXYZE coalesced memory layout); immediately after
        # the load we permute BE to the FIRST slot for downstream compute.
        # This early/late permute eliminates shared-mem bank conflicts in MMA
        # (same trick as the p=3 `opt` kernel in triton_BXYZE.py).
        n_alive = len(spatial_dims)
        rank = n_alive + 1

        def slot_for_axis(pos: int) -> str:
            slots = ["None"] * rank
            slots[pos] = ":"
            return "[" + ", ".join(slots) + "]"

        # Load offsets: spatial axes at positions 0..n_alive-1 (canonical k,j,i
        # ordering); BE at the LAST slot (n_alive).
        e_pos = n_alive
        axis_position = {}
        cursor = 0
        for ax in ["k", "j", "i"]:
            tag = {"k": k_t, "j": j_t, "i": i_t}[ax]
            if tag == "c":
                axis_position[ax] = cursor
                cursor += 1

        # Build offset expression
        parts = []
        # block_base broadcasts only over the e axis
        parts.append(f"block_base{slot_for_axis(e_pos)}")
        # k contribution
        if k_t == "c":
            parts.append(f"k_c{slot_for_axis(axis_position['k'])} * N * N * BLOCK_SIZE_DATA")
        else:
            parts.append("4 * N * N * BLOCK_SIZE_DATA")
        # j contribution
        if j_t == "c":
            parts.append(f"j_c{slot_for_axis(axis_position['j'])} * N * BLOCK_SIZE_DATA")
        else:
            parts.append("4 * N * BLOCK_SIZE_DATA")
        # i contribution
        if i_t == "c":
            parts.append(f"i_c{slot_for_axis(axis_position['i'])} * BLOCK_SIZE_DATA")
        else:
            parts.append("4 * BLOCK_SIZE_DATA")
        # e_inblock
        parts.append(f"e_inblock{slot_for_axis(e_pos)}")

        offs_var = f"offs_{k_t}{j_t}{i_t}"
        mask_var = f"mask_{k_t}{j_t}{i_t}"
        lines.append(f"    {offs_var} = (" + " + ".join(parts) + ")")
        lines.append(f"    {mask_var} = mask_e{slot_for_axis(e_pos)}")
        # Load with BE in the last slot (BXYZE coalesced), then EARLY PERMUTE to
        # bring BE to slot 0 for compute. Skip the permute when no spatial axes
        # are alive (rank 1: nothing to permute).
        loaded_shape = ", ".join([d for d in spatial_dims] + ["BE"])
        if n_alive == 0:
            lines.append(f"    {name} = tl.load(u_ptr + {offs_var}, mask={mask_var}, other=0.0)  "
                         f"# shape ({loaded_shape})")
        else:
            lines.append(f"    _raw = tl.load(u_ptr + {offs_var}, mask={mask_var}, other=0.0)  "
                         f"# loaded shape ({loaded_shape}), BE last")
            perm_to_be_first = tuple([n_alive] + list(range(n_alive)))
            compute_shape = ", ".join(["BE"] + [d for d in spatial_dims])
            lines.append(f"    {name} = tl.permute(_raw, {perm_to_be_first})  "
                         f"# early permute → shape ({compute_shape})")
        lines.append("")
    return "\n".join(lines)


def _gen_phase_contract(
    phase_label: str,
    contract_axis: str,         # 'i', 'j', or 'k' — which axis is contracted
    in_prefix: str,             # input tensor name prefix, e.g. 'u'
    out_M_prefix: str,          # output M-side prefix, e.g. 'xM'
    out_D_prefix: str | None,   # output D-side prefix, or None to skip
    keep_M: bool = True,
    keep_D: bool = True,
) -> str:
    """Generate code for one stacked contraction over one input set.

    The input is 8 split tensors {in_prefix}_<k><j><i> (where each char ∈ {c,t}).
    The contracted axis ('i', 'j', or 'k') is collapsed: its 'c' slice (4 values)
    is the MMA input, the 't' slice (1 value) is the K-tail.

    Outputs (per non-skipped side):
      8 split tensors {out_prefix}_<a><b><c'>  where c' is the same axis label
      ('i' becomes "i'", etc.) but the indexing convention is identical.

    For non-contracted axes (call them α, β), the input has 4 tag combos
    (αβ ∈ {cc, ct, tc, tt}). For each, we have an MMA-shaped piece and a tail
    piece along the contracted axis. The output for that (α, β) combo has its
    own 'c' and 't' along the new output axis (replacing the contracted axis).

    The two non-contracted axes are kept exactly as-is — they retain their c/t
    tags. The contracted axis is REPLACED in the output by the new output-axis tag.
    """
    # Determine non-contracted axes (preserving order k,j,i)
    all_axes = ["k", "j", "i"]
    if contract_axis not in all_axes:
        raise ValueError(contract_axis)
    other = [a for a in all_axes if a != contract_axis]
    # Determine output axis name (i'→i', j'→j', k'→k') — but in tag-naming we
    # just use the same letter at the same positional slot as the contracted axis.
    # In the tag-strings (k,j,i) order, the contracted axis position is fixed.
    contract_pos = all_axes.index(contract_axis)

    lines = []
    lines.append("")
    lines.append("    # " + "=" * 60)
    lines.append(f"    # {phase_label}: contract over {contract_axis}")
    lines.append("    # " + "=" * 60)

    # For each (α-tag, β-tag) combo we do ONE stacked MMA + K-tail + output-row tail.
    # That produces:
    #   out_{...}_{α-tag}{β-tag}{c}: from MMA top-4 / bot-4
    #   out_{...}_{α-tag}{β-tag}{t}: from scalar output-row dot
    # The output's tag-tuple is built by inserting the new axis tag at contract_pos.

    for α_tag in TAGS:
        for β_tag in TAGS:
            # Reconstruct input tag triple: insert contracted-axis tag at contract_pos
            def make_triple(contract_tag: str) -> Tuple[str, str, str]:
                tags = []
                others_iter = iter([α_tag, β_tag])
                for ax in all_axes:
                    if ax == contract_axis:
                        tags.append(contract_tag)
                    else:
                        tags.append(next(others_iter))
                return tuple(tags)  # type: ignore

            in_c_triple = make_triple("c")    # MMA input (4 values along contracted axis)
            in_t_triple = make_triple("t")    # K-tail input (1 value)

            in_c_name = f"{in_prefix}_{''.join(in_c_triple)}"
            in_t_name = f"{in_prefix}_{''.join(in_t_triple)}"

            # Input shapes: input tensor has BE + alive non-contracted axes (each of size 4 if c, absent if t)
            # PLUS the contracted-axis dim (size 4 for c-input, size absent for t-input — t means scalar=1).
            # Examples:
            #   contract='i', α=k, β=j, both 'c': in_ccc shape (BE, k=4, j=4, i=4); in_cct shape (BE, k=4, j=4)
            #   α='t', β='c' (k=tail, j=core): in_tcc shape (BE, j=4, i=4); in_tct shape (BE, j=4)
            #   α='t', β='t': in_ttc shape (BE, i=4); in_ttt shape (BE,)

            # The MMA wants A of shape (M, N_MMA) where M flattens all non-contracted axes.
            # For α=c, β=c: M = BE * N_MMA * N_MMA.
            # For α=c, β=t: M = BE * N_MMA.
            # For α=t, β=t: M = BE.
            n_alive = (1 if α_tag == "c" else 0) + (1 if β_tag == "c" else 0)
            if n_alive == 2:
                m_expr = "BE * N_MMA * N_MMA"
                in_c_shape_post = f"(BE, N_MMA, N_MMA, N_MMA)"
                in_t_shape_post = f"(BE, N_MMA, N_MMA)"
                # Output reshape: (BE, A, B, 2, N_MMA)
                acc_reshape = "(BE, N_MMA, N_MMA, 2, N_MMA)"
                acc_permute = "(0, 1, 2, 4, 3)"
                split_out_shape = "(BE, N_MMA, N_MMA, N_MMA)"
                # full-input shape for output-row tail sum
                sum_axis = "3"
                row_broadcast = "[None, None, None, :]"
                full_for_row_shape = "(BE, N_MMA, N_MMA, 5)"  # constructed via cat below
            elif n_alive == 1:
                m_expr = "BE * N_MMA"
                in_c_shape_post = f"(BE, N_MMA, N_MMA)"
                in_t_shape_post = f"(BE, N_MMA)"
                acc_reshape = "(BE, N_MMA, 2, N_MMA)"
                acc_permute = "(0, 1, 3, 2)"
                split_out_shape = "(BE, N_MMA, N_MMA)"
                sum_axis = "2"
                row_broadcast = "[None, None, :]"
                full_for_row_shape = "(BE, N_MMA, 5)"
            else:  # n_alive == 0
                m_expr = "BE"
                in_c_shape_post = f"(BE, N_MMA)"
                in_t_shape_post = f"(BE,)"
                acc_reshape = "(BE, 2, N_MMA)"
                acc_permute = "(0, 2, 1)"
                split_out_shape = "(BE, N_MMA)"
                sum_axis = "1"
                row_broadcast = "[None, :]"
                full_for_row_shape = "(BE, 5)"

            tag_str = f"{α_tag}{β_tag}"
            lines.append("")
            lines.append(f"    # ---- slab (α={α_tag}, β={β_tag}): inputs {in_c_name} + {in_t_name} ----")

            # The input tensor in_c has canonical layout (BE, [k?], [j?], [i?])
            # where each axis is present iff its tag is 'c'.  For a contraction
            # to work via tl.dot's inner-dim semantics, we need the CONTRACTED
            # axis as the innermost (last) dim.  Build a permute order that
            # rotates the contracted axis to the end of the spatial axes,
            # keeping BE first and the non-contracted axes in canonical order.
            # Then after the MMA, we'll permute the output to put the NEW axis
            # (which replaces the contracted axis) back into its canonical slot.

            # Spatial axes present in in_c (in canonical order). The contracted
            # axis IS present in in_c (it carries the 'c' = 4-wide slice).
            in_c_axes = []          # list of axis labels ('k','j','i') present in in_c
            for ax in all_axes:
                tag = {"k": in_c_triple[0], "j": in_c_triple[1], "i": in_c_triple[2]}[ax]
                if tag == "c":
                    in_c_axes.append(ax)

            # in_c spatial-axis order is in_c_axes; full rank is 1+len(in_c_axes).
            # Build pre-permute that moves contracted axis to the end.
            contract_idx_in_c = in_c_axes.index(contract_axis)
            # Pre-permute: (0,) + in_c_axes positions reordered: keep non-contracted axes in canonical order, then contracted last.
            pre_perm_axes = [ax for ax in in_c_axes if ax != contract_axis] + [contract_axis]
            # Triton permute indices (in axis-position space). Position of BE = 0, in_c spatial axes start at 1.
            pre_perm_indices = [0] + [in_c_axes.index(ax) + 1 for ax in pre_perm_axes]

            if pre_perm_indices != list(range(len(pre_perm_indices))):
                lines.append(f"    _in_c_perm = tl.permute({in_c_name}, {tuple(pre_perm_indices)})")
                in_c_perm_name = "_in_c_perm"
            else:
                in_c_perm_name = in_c_name

            lines.append(f"    _in_c_flat = tl.reshape({in_c_perm_name}, ({m_expr}, N_MMA))")
            # K-tail input (no contracted dim) — non-contracted axes only.
            # in_t has the same non-contracted axes as in_c minus the contracted one.
            in_t_axes = [ax for ax in in_c_axes if ax != contract_axis]
            # Its current order is canonical (since make_triple preserved canonical order).
            # We need shape (m, ) where m = BE * prod(N_MMA for each alive non-contracted axis).
            if n_alive == 0:
                lines.append(f"    _in_t_flat = {in_t_name}")
            else:
                lines.append(f"    _in_t_flat = tl.reshape({in_t_name}, ({m_expr},))")
            # MMA
            lines.append(f"    _acc = tl.dot(_in_c_flat, O_low4_T)")
            lines.append(f"    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]")
            # Reshape & split:
            #   _acc shape (m, 8). The 'm' axis has the non-contracted axes in their pre_perm order
            #   (i.e. with contracted axis removed; canonical order for the others).
            #   The width-8 splits into (2 = type) × (4 = new-axis-core).
            #
            #   reshape to (BE, [non-contracted-axes...], 2, N_MMA)
            #   then permute the type axis out: (BE, [non-c-axes...], N_MMA, 2)
            #   then split → (BE, [non-c-axes...], N_MMA).  The N_MMA dim represents
            #   the new (output) axis along which we contracted.
            lines.append(f"    _acc_4d = tl.reshape(_acc, {acc_reshape})")
            lines.append(f"    _acc_p = tl.permute(_acc_4d, {acc_permute})")
            lines.append(f"    _top, _bot = tl.split(_acc_p)")
            # _top, _bot shape: (BE, [non-contracted-axes in canonical order], N_MMA=new-axis-core)
            # We need to permute this so the new axis ends up in its canonical slot (where the contracted
            # axis used to be in the (k, j, i) ordering).
            # Current axis labels of _top: ('BE',) + tuple(non-contracted axes in canonical order) + (contract_axis_NEW,)
            # Desired: BE + canonical (k, j, i) order with new-axis at its original slot.
            current_axes = ["BE"] + [ax for ax in in_c_axes if ax != contract_axis] + [contract_axis]
            desired_axes = ["BE"] + [ax if ax != contract_axis else contract_axis for ax in in_c_axes]
            # post-permute: indices in current_axes that produce desired_axes
            post_perm = [current_axes.index(ax) for ax in desired_axes]
            if post_perm != list(range(len(post_perm))):
                lines.append(f"    _top = tl.permute(_top, {tuple(post_perm)})")
                lines.append(f"    _bot = tl.permute(_bot, {tuple(post_perm)})")

            # Output-row tail (along the new axis = scalar slice contract_axis_NEW = 't').
            # Compute via the full 5-row scalar dot over the contracted axis:
            #   row = sum(in_c * row_c[broadcast over non-contracted axes], axis=contract_axis_innermost)
            #         + in_t * row_t
            # We can use the already-permuted in_c_perm (where contract axis is innermost) — easiest.
            row_broadcast_perm = "[None" + ", None" * (n_alive) + ", :]"  # for (BE, [non-c], contract) rank
            lines.append(f"    _row_M_lo = tl.sum({in_c_perm_name} * mrow_c{row_broadcast_perm}, axis={n_alive + 1 - 1})")
            # the axis index above is the position of the contracted axis in in_c_perm = last = n_alive+1-1 = n_alive (since rank = n_alive+1)
            # but with BE at 0 and n_alive non-contracted axes, the contracted axis is at position n_alive+1-1? Let me just hard-code:
            # rank of in_c_perm = 1 (BE) + (n_alive+1) (non-c axes + contract) - wait the in_c tensor has rank = 1 + len(in_c_axes) = 1 + (n_alive+1) = n_alive+2.
            # in_c_perm rank also n_alive+2; last axis is the contracted one. Index = n_alive+1.
            # Whoops — I had the wrong index. Let me regenerate properly.
            # Replace the just-emitted line.
            lines.pop()  # remove wrong axis index line
            contract_axis_idx_in_perm = n_alive + 1  # in (BE, non-c..., contract) — contract is at index 1+n_alive
            lines.append(f"    _row_M_lo = tl.sum({in_c_perm_name} * mrow_c{row_broadcast_perm}, axis={contract_axis_idx_in_perm})")
            lines.append(f"    _row_M = _row_M_lo + {in_t_name} * mrow_t")
            lines.append(f"    _row_D_lo = tl.sum({in_c_perm_name} * drow_c{row_broadcast_perm}, axis={contract_axis_idx_in_perm})")
            lines.append(f"    _row_D = _row_D_lo + {in_t_name} * drow_t")

            # Now write outputs:
            #   out_M_{<core triple>} = _top                  (axis-tag-c at contract_pos)
            #   out_M_{<tail triple>} = _row_M                (axis-tag-t at contract_pos)
            # Same for D side (if keep_D).
            out_c_triple = make_triple("c")  # core along new axis
            out_t_triple = make_triple("t")  # tail along new axis
            if keep_M:
                lines.append(f"    {out_M_prefix}_{''.join(out_c_triple)} = _top")
                lines.append(f"    {out_M_prefix}_{''.join(out_t_triple)} = _row_M")
            if keep_D and out_D_prefix is not None:
                lines.append(f"    {out_D_prefix}_{''.join(out_c_triple)} = _bot")
                lines.append(f"    {out_D_prefix}_{''.join(out_t_triple)} = _row_D")
    return "\n".join(lines)


def _gen_sum_block(a_prefix: str, b_prefix: str, out_prefix: str) -> str:
    """Element-wise sum of two split-tensor representations: out = a + b."""
    lines = []
    lines.append("")
    lines.append(f"    # ---- {out_prefix} = {a_prefix} + {b_prefix} ----")
    for triple in TRIPLES:
        s = "".join(triple)
        lines.append(f"    {out_prefix}_{s} = {a_prefix}_{s} + {b_prefix}_{s}")
    return "\n".join(lines)


def _gen_store_block(prefix: str) -> str:
    """Store the final result back to BXYZE memory using same offset patterns
    as load. Compute keeps BE in slot 0; stores need BE in the LAST slot to
    match the BXYZE offset layout (LATE PERMUTE)."""
    lines = []
    lines.append("")
    lines.append("    # ====================================================")
    lines.append(f"    # Store result {prefix} into BXYZE output (late permute)")
    lines.append("    # ====================================================")
    for k_t, j_t, i_t in TRIPLES:
        s = f"{k_t}{j_t}{i_t}"
        offs_var = f"offs_{s}"
        mask_var = f"mask_{s}"
        # Count alive spatial axes for this tag triple
        n_alive_here = sum(1 for t in (k_t, j_t, i_t) if t == "c")
        if n_alive_here == 0:
            # rank 1: just (BE,) — no permute needed
            lines.append(f"    tl.store(out_ptr + {offs_var}, {prefix}_{s}, mask={mask_var})")
        else:
            # Compute layout: (BE, spatial...).  Need: (spatial..., BE).
            perm_to_be_last = tuple(list(range(1, n_alive_here + 1)) + [0])
            lines.append(f"    _out = tl.permute({prefix}_{s}, {perm_to_be_last})")
            lines.append(f"    tl.store(out_ptr + {offs_var}, _out, mask={mask_var})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def generate_kernel_source() -> str:
    parts = [_HEADER]
    parts.append(_gen_load_block())
    # Phase 1: contract i over u → xM (top), xD (bot)
    parts.append(_gen_phase_contract(
        phase_label="Phase 1",
        contract_axis="i",
        in_prefix="u",
        out_M_prefix="xM",
        out_D_prefix="xD",
        keep_M=True, keep_D=True,
    ))
    # Phase 2a: contract j over xM → yMM (top), yMD (bot)
    parts.append(_gen_phase_contract(
        phase_label="Phase 2a",
        contract_axis="j",
        in_prefix="xM",
        out_M_prefix="yMM",
        out_D_prefix="yMD",
        keep_M=True, keep_D=True,
    ))
    # Phase 2b: contract j over xD → yDM (top only)
    parts.append(_gen_phase_contract(
        phase_label="Phase 2b",
        contract_axis="j",
        in_prefix="xD",
        out_M_prefix="yDM",
        out_D_prefix=None,
        keep_M=True, keep_D=False,
    ))
    # u_sum = yMD + yDM
    parts.append(_gen_sum_block("yMD", "yDM", "usum"))
    # Phase 3a: contract k over usum → z1M (top only)
    parts.append(_gen_phase_contract(
        phase_label="Phase 3a",
        contract_axis="k",
        in_prefix="usum",
        out_M_prefix="z1M",
        out_D_prefix=None,
        keep_M=True, keep_D=False,
    ))
    # Phase 3b: contract k over yMM → z2D (bot only)
    parts.append(_gen_phase_contract(
        phase_label="Phase 3b",
        contract_axis="k",
        in_prefix="yMM",
        out_M_prefix=None,   # we don't store the top half
        out_D_prefix="z2D",
        keep_M=False, keep_D=True,
    ))
    parts.append(_gen_sum_block("z1M", "z2D", "res"))
    parts.append(_gen_store_block("res"))
    return "\n".join(parts) + "\n"


def main():
    src = generate_kernel_source()
    import os
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "_generated_mixed_p4_kernel.py")
    with open(out_path, "w") as f:
        f.write(src)
    print(f"wrote {out_path} ({len(src.splitlines())} lines)")


if __name__ == "__main__":
    main()
