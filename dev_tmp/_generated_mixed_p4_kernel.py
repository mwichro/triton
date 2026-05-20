"""Auto-generated Triton kernel: mixed MMA + CUDA-core Laplacian for p=4 (N=5).

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
    O_low4_T = tl.load(O_low4_ptr + rO[None, :] * N_MMA + cO[:, None])   # (4, 8)
    O_col4 = tl.load(O_col4_ptr + rO)                                    # (8,)
    # M_row4, D_row4: full 5 values, loaded into (4,)+scalar pieces
    mrow_c = tl.load(M_row4_ptr + tl.arange(0, N_MMA))     # M[4, 0..3]   (4,)
    mrow_t = tl.load(M_row4_ptr + 4)                       # M[4, 4]      scalar
    drow_c = tl.load(D_row4_ptr + tl.arange(0, N_MMA))     # (4,)
    drow_t = tl.load(D_row4_ptr + 4)                       # scalar


    # =================================================================
    # 8 per-tag tl.load calls, each immediately early-permuted to BE-first.
    # =================================================================
    k_c = tl.arange(0, N_MMA)
    j_c = tl.arange(0, N_MMA)
    i_c = tl.arange(0, N_MMA)

    offs_ccc = (block_base[None, None, None, :] + k_c[:, None, None, None] * N * N * BLOCK_SIZE_DATA + j_c[None, :, None, None] * N * BLOCK_SIZE_DATA + i_c[None, None, :, None] * BLOCK_SIZE_DATA + e_inblock[None, None, None, :])
    mask_ccc = mask_e[None, None, None, :]
    _raw = tl.load(u_ptr + offs_ccc, mask=mask_ccc, other=0.0)
    u_ccc = tl.permute(_raw, (3, 0, 1, 2))  # BE → slot 0
    offs_cct = (block_base[None, None, :] + k_c[:, None, None] * N * N * BLOCK_SIZE_DATA + j_c[None, :, None] * N * BLOCK_SIZE_DATA + 4 * BLOCK_SIZE_DATA + e_inblock[None, None, :])
    mask_cct = mask_e[None, None, :]
    _raw = tl.load(u_ptr + offs_cct, mask=mask_cct, other=0.0)
    u_cct = tl.permute(_raw, (2, 0, 1))  # BE → slot 0
    offs_ctc = (block_base[None, None, :] + k_c[:, None, None] * N * N * BLOCK_SIZE_DATA + 4 * N * BLOCK_SIZE_DATA + i_c[None, :, None] * BLOCK_SIZE_DATA + e_inblock[None, None, :])
    mask_ctc = mask_e[None, None, :]
    _raw = tl.load(u_ptr + offs_ctc, mask=mask_ctc, other=0.0)
    u_ctc = tl.permute(_raw, (2, 0, 1))  # BE → slot 0
    offs_ctt = (block_base[None, :] + k_c[:, None] * N * N * BLOCK_SIZE_DATA + 4 * N * BLOCK_SIZE_DATA + 4 * BLOCK_SIZE_DATA + e_inblock[None, :])
    mask_ctt = mask_e[None, :]
    _raw = tl.load(u_ptr + offs_ctt, mask=mask_ctt, other=0.0)
    u_ctt = tl.permute(_raw, (1, 0))  # BE → slot 0
    offs_tcc = (block_base[None, None, :] + 4 * N * N * BLOCK_SIZE_DATA + j_c[:, None, None] * N * BLOCK_SIZE_DATA + i_c[None, :, None] * BLOCK_SIZE_DATA + e_inblock[None, None, :])
    mask_tcc = mask_e[None, None, :]
    _raw = tl.load(u_ptr + offs_tcc, mask=mask_tcc, other=0.0)
    u_tcc = tl.permute(_raw, (2, 0, 1))  # BE → slot 0
    offs_tct = (block_base[None, :] + 4 * N * N * BLOCK_SIZE_DATA + j_c[:, None] * N * BLOCK_SIZE_DATA + 4 * BLOCK_SIZE_DATA + e_inblock[None, :])
    mask_tct = mask_e[None, :]
    _raw = tl.load(u_ptr + offs_tct, mask=mask_tct, other=0.0)
    u_tct = tl.permute(_raw, (1, 0))  # BE → slot 0
    offs_ttc = (block_base[None, :] + 4 * N * N * BLOCK_SIZE_DATA + 4 * N * BLOCK_SIZE_DATA + i_c[:, None] * BLOCK_SIZE_DATA + e_inblock[None, :])
    mask_ttc = mask_e[None, :]
    _raw = tl.load(u_ptr + offs_ttc, mask=mask_ttc, other=0.0)
    u_ttc = tl.permute(_raw, (1, 0))  # BE → slot 0
    offs_ttt = (block_base[:] + 4 * N * N * BLOCK_SIZE_DATA + 4 * N * BLOCK_SIZE_DATA + 4 * BLOCK_SIZE_DATA + e_inblock[:])
    mask_ttt = mask_e[:]
    u_ttt = tl.load(u_ptr + offs_ttt, mask=mask_ttt, other=0.0)

    # ============================================================
    # Phase 1: contract over i
    # ============================================================

    # ---- slab (α=c, β=c): inputs u_ccc + u_cct ----
    _in_c_flat = tl.reshape(u_ccc, (BE * N_MMA * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(u_cct, (BE * N_MMA * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA, N_MMA))
    _top = _top_4d
    _bot = _bot_4d
    _row_M_lo = tl.sum(u_ccc * mrow_c[None, None, None, :], axis=3)
    _row_M = _row_M_lo + u_cct * mrow_t
    _row_D_lo = tl.sum(u_ccc * drow_c[None, None, None, :], axis=3)
    _row_D = _row_D_lo + u_cct * drow_t
    xM_ccc = _top
    xM_cct = _row_M
    xD_ccc = _bot
    xD_cct = _row_D

    # ---- slab (α=c, β=t): inputs u_ctc + u_ctt ----
    _in_c_flat = tl.reshape(u_ctc, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(u_ctt, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA))
    _top = _top_4d
    _bot = _bot_4d
    _row_M_lo = tl.sum(u_ctc * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + u_ctt * mrow_t
    _row_D_lo = tl.sum(u_ctc * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + u_ctt * drow_t
    xM_ctc = _top
    xM_ctt = _row_M
    xD_ctc = _bot
    xD_ctt = _row_D

    # ---- slab (α=t, β=c): inputs u_tcc + u_tct ----
    _in_c_flat = tl.reshape(u_tcc, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(u_tct, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA))
    _top = _top_4d
    _bot = _bot_4d
    _row_M_lo = tl.sum(u_tcc * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + u_tct * mrow_t
    _row_D_lo = tl.sum(u_tcc * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + u_tct * drow_t
    xM_tcc = _top
    xM_tct = _row_M
    xD_tcc = _bot
    xD_tct = _row_D

    # ---- slab (α=t, β=t): inputs u_ttc + u_ttt ----
    _in_c_flat = tl.reshape(u_ttc, (BE, N_MMA))
    _in_t_flat = u_ttt
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA))
    _top = _top_4d
    _bot = _bot_4d
    _row_M_lo = tl.sum(u_ttc * mrow_c[None, :], axis=1)
    _row_M = _row_M_lo + u_ttt * mrow_t
    _row_D_lo = tl.sum(u_ttc * drow_c[None, :], axis=1)
    _row_D = _row_D_lo + u_ttt * drow_t
    xM_ttc = _top
    xM_ttt = _row_M
    xD_ttc = _bot
    xD_ttt = _row_D

    # ============================================================
    # Phase 2a: contract over j
    # ============================================================

    # ---- slab (α=c, β=c): inputs xM_ccc + xM_ctc ----
    _in_c_perm = tl.permute(xM_ccc, (0, 1, 3, 2))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(xM_ctc, (BE * N_MMA * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 1, 3, 2))
    _bot = tl.permute(_bot_4d, (0, 1, 3, 2))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, None, :], axis=3)
    _row_M = _row_M_lo + xM_ctc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, None, :], axis=3)
    _row_D = _row_D_lo + xM_ctc * drow_t
    yMM_ccc = _top
    yMM_ctc = _row_M
    yMD_ccc = _bot
    yMD_ctc = _row_D

    # ---- slab (α=c, β=t): inputs xM_cct + xM_ctt ----
    _in_c_flat = tl.reshape(xM_cct, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(xM_ctt, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA))
    _top = _top_4d
    _bot = _bot_4d
    _row_M_lo = tl.sum(xM_cct * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + xM_ctt * mrow_t
    _row_D_lo = tl.sum(xM_cct * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + xM_ctt * drow_t
    yMM_cct = _top
    yMM_ctt = _row_M
    yMD_cct = _bot
    yMD_ctt = _row_D

    # ---- slab (α=t, β=c): inputs xM_tcc + xM_ttc ----
    _in_c_perm = tl.permute(xM_tcc, (0, 2, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(xM_ttc, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 2, 1))
    _bot = tl.permute(_bot_4d, (0, 2, 1))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + xM_ttc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + xM_ttc * drow_t
    yMM_tcc = _top
    yMM_ttc = _row_M
    yMD_tcc = _bot
    yMD_ttc = _row_D

    # ---- slab (α=t, β=t): inputs xM_tct + xM_ttt ----
    _in_c_flat = tl.reshape(xM_tct, (BE, N_MMA))
    _in_t_flat = xM_ttt
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA))
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA))
    _top = _top_4d
    _bot = _bot_4d
    _row_M_lo = tl.sum(xM_tct * mrow_c[None, :], axis=1)
    _row_M = _row_M_lo + xM_ttt * mrow_t
    _row_D_lo = tl.sum(xM_tct * drow_c[None, :], axis=1)
    _row_D = _row_D_lo + xM_ttt * drow_t
    yMM_tct = _top
    yMM_ttt = _row_M
    yMD_tct = _bot
    yMD_ttt = _row_D

    # ============================================================
    # Phase 2b: contract over j
    # ============================================================

    # ---- slab (α=c, β=c): inputs xD_ccc + xD_ctc ----
    _in_c_perm = tl.permute(xD_ccc, (0, 1, 3, 2))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(xD_ctc, (BE * N_MMA * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 1, 3, 2))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, None, :], axis=3)
    _row_M = _row_M_lo + xD_ctc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, None, :], axis=3)
    _row_D = _row_D_lo + xD_ctc * drow_t
    yDM_ccc = _top
    yDM_ctc = _row_M

    # ---- slab (α=c, β=t): inputs xD_cct + xD_ctt ----
    _in_c_flat = tl.reshape(xD_cct, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(xD_ctt, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _top = _top_4d
    _row_M_lo = tl.sum(xD_cct * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + xD_ctt * mrow_t
    _row_D_lo = tl.sum(xD_cct * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + xD_ctt * drow_t
    yDM_cct = _top
    yDM_ctt = _row_M

    # ---- slab (α=t, β=c): inputs xD_tcc + xD_ttc ----
    _in_c_perm = tl.permute(xD_tcc, (0, 2, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(xD_ttc, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 2, 1))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + xD_ttc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + xD_ttc * drow_t
    yDM_tcc = _top
    yDM_ttc = _row_M

    # ---- slab (α=t, β=t): inputs xD_tct + xD_ttt ----
    _in_c_flat = tl.reshape(xD_tct, (BE, N_MMA))
    _in_t_flat = xD_ttt
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA))
    _top = _top_4d
    _row_M_lo = tl.sum(xD_tct * mrow_c[None, :], axis=1)
    _row_M = _row_M_lo + xD_ttt * mrow_t
    _row_D_lo = tl.sum(xD_tct * drow_c[None, :], axis=1)
    _row_D = _row_D_lo + xD_ttt * drow_t
    yDM_tct = _top
    yDM_ttt = _row_M

    # ---- usum = yMD + yDM ----
    usum_ccc = yMD_ccc + yDM_ccc
    usum_cct = yMD_cct + yDM_cct
    usum_ctc = yMD_ctc + yDM_ctc
    usum_ctt = yMD_ctt + yDM_ctt
    usum_tcc = yMD_tcc + yDM_tcc
    usum_tct = yMD_tct + yDM_tct
    usum_ttc = yMD_ttc + yDM_ttc
    usum_ttt = yMD_ttt + yDM_ttt

    # ============================================================
    # Phase 3a: contract over k
    # ============================================================

    # ---- slab (α=c, β=c): inputs usum_ccc + usum_tcc ----
    _in_c_perm = tl.permute(usum_ccc, (0, 2, 3, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(usum_tcc, (BE * N_MMA * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 3, 1, 2))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, None, :], axis=3)
    _row_M = _row_M_lo + usum_tcc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, None, :], axis=3)
    _row_D = _row_D_lo + usum_tcc * drow_t
    z1M_ccc = _top
    z1M_tcc = _row_M

    # ---- slab (α=c, β=t): inputs usum_cct + usum_tct ----
    _in_c_perm = tl.permute(usum_cct, (0, 2, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(usum_tct, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 2, 1))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + usum_tct * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + usum_tct * drow_t
    z1M_cct = _top
    z1M_tct = _row_M

    # ---- slab (α=t, β=c): inputs usum_ctc + usum_ttc ----
    _in_c_perm = tl.permute(usum_ctc, (0, 2, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(usum_ttc, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA, N_MMA))
    _top = tl.permute(_top_4d, (0, 2, 1))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + usum_ttc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + usum_ttc * drow_t
    z1M_ctc = _top
    z1M_ttc = _row_M

    # ---- slab (α=t, β=t): inputs usum_ctt + usum_ttt ----
    _in_c_flat = tl.reshape(usum_ctt, (BE, N_MMA))
    _in_t_flat = usum_ttt
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _top_4d = tl.reshape(_top_flat, (BE, N_MMA))
    _top = _top_4d
    _row_M_lo = tl.sum(usum_ctt * mrow_c[None, :], axis=1)
    _row_M = _row_M_lo + usum_ttt * mrow_t
    _row_D_lo = tl.sum(usum_ctt * drow_c[None, :], axis=1)
    _row_D = _row_D_lo + usum_ttt * drow_t
    z1M_ctt = _top
    z1M_ttt = _row_M

    # ============================================================
    # Phase 3b: contract over k
    # ============================================================

    # ---- slab (α=c, β=c): inputs yMM_ccc + yMM_tcc ----
    _in_c_perm = tl.permute(yMM_ccc, (0, 2, 3, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(yMM_tcc, (BE * N_MMA * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA, N_MMA))
    _bot = tl.permute(_bot_4d, (0, 3, 1, 2))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, None, :], axis=3)
    _row_M = _row_M_lo + yMM_tcc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, None, :], axis=3)
    _row_D = _row_D_lo + yMM_tcc * drow_t
    z2D_ccc = _bot
    z2D_tcc = _row_D

    # ---- slab (α=c, β=t): inputs yMM_cct + yMM_tct ----
    _in_c_perm = tl.permute(yMM_cct, (0, 2, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(yMM_tct, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA))
    _bot = tl.permute(_bot_4d, (0, 2, 1))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + yMM_tct * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + yMM_tct * drow_t
    z2D_cct = _bot
    z2D_tct = _row_D

    # ---- slab (α=t, β=c): inputs yMM_ctc + yMM_ttc ----
    _in_c_perm = tl.permute(yMM_ctc, (0, 2, 1))
    _in_c_flat = tl.reshape(_in_c_perm, (BE * N_MMA, N_MMA))
    _in_t_flat = tl.reshape(yMM_ttc, (BE * N_MMA,))
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE * N_MMA, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA, N_MMA))
    _bot = tl.permute(_bot_4d, (0, 2, 1))
    _row_M_lo = tl.sum(_in_c_perm * mrow_c[None, None, :], axis=2)
    _row_M = _row_M_lo + yMM_ttc * mrow_t
    _row_D_lo = tl.sum(_in_c_perm * drow_c[None, None, :], axis=2)
    _row_D = _row_D_lo + yMM_ttc * drow_t
    z2D_ctc = _bot
    z2D_ttc = _row_D

    # ---- slab (α=t, β=t): inputs yMM_ctt + yMM_ttt ----
    _in_c_flat = tl.reshape(yMM_ctt, (BE, N_MMA))
    _in_t_flat = yMM_ttt
    _acc = tl.dot(_in_c_flat, O_low4_T)
    _acc = _acc + _in_t_flat[:, None] * O_col4[None, :]
    _acc_2d = tl.reshape(_acc, (BE, 2, N_MMA))
    _acc_2d_p = tl.permute(_acc_2d, (0, 2, 1))
    _top_flat, _bot_flat = tl.split(_acc_2d_p)
    _bot_4d = tl.reshape(_bot_flat, (BE, N_MMA))
    _bot = _bot_4d
    _row_M_lo = tl.sum(yMM_ctt * mrow_c[None, :], axis=1)
    _row_M = _row_M_lo + yMM_ttt * mrow_t
    _row_D_lo = tl.sum(yMM_ctt * drow_c[None, :], axis=1)
    _row_D = _row_D_lo + yMM_ttt * drow_t
    z2D_ctt = _bot
    z2D_ttt = _row_D

    # ---- res = z1M + z2D ----
    res_ccc = z1M_ccc + z2D_ccc
    res_cct = z1M_cct + z2D_cct
    res_ctc = z1M_ctc + z2D_ctc
    res_ctt = z1M_ctt + z2D_ctt
    res_tcc = z1M_tcc + z2D_tcc
    res_tct = z1M_tct + z2D_tct
    res_ttc = z1M_ttc + z2D_ttc
    res_ttt = z1M_ttt + z2D_ttt

    # ====================================================
    # Store result res into BXYZE output (late permute)
    # ====================================================
    _out = tl.permute(res_ccc, (1, 2, 3, 0))
    tl.store(out_ptr + offs_ccc, _out, mask=mask_ccc)
    _out = tl.permute(res_cct, (1, 2, 0))
    tl.store(out_ptr + offs_cct, _out, mask=mask_cct)
    _out = tl.permute(res_ctc, (1, 2, 0))
    tl.store(out_ptr + offs_ctc, _out, mask=mask_ctc)
    _out = tl.permute(res_ctt, (1, 0))
    tl.store(out_ptr + offs_ctt, _out, mask=mask_ctt)
    _out = tl.permute(res_tcc, (1, 2, 0))
    tl.store(out_ptr + offs_tcc, _out, mask=mask_tcc)
    _out = tl.permute(res_tct, (1, 0))
    tl.store(out_ptr + offs_tct, _out, mask=mask_tct)
    _out = tl.permute(res_ttc, (1, 0))
    tl.store(out_ptr + offs_ttc, _out, mask=mask_ttc)
    tl.store(out_ptr + offs_ttt, res_ttt, mask=mask_ttt)
