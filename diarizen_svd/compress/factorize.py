#!/usr/bin/env python3
"""Whitened / weighted SVD of one weight matrix and the low-rank factor pair built from it.

Three factorization rules are implemented.  They differ only in which calibration statistics enter
the SVD; the rank allocation (allocate.py) and the factor construction (``to_factors``) are shared.

  input_whitened_svd   SVD(W L_x), C_x = L_x L_x^T           SVD-LLM  [Wang et al., ICLR 2025]
  two_sided_svd        SVD(L_g^T W L_x), C_g = L_g L_g^T      OBD-LLM  [Li et al., arXiv:2604.00821],
                                                              GFWSVD   [Chekalina et al., arXiv:2505.17974]
  fisher_weighted_svd  SVD(D W), D = diag(sqrt(I_row))        FWSVD    [Hsu et al., ICLR 2022]

The two-sided rule is the closed-form minimizer of the weighted Frobenius error
||L_g^T (W - W_r) L_x||_F, which under the K-FAC model H ~= C_x (x) C_g equals the second-order loss
increase of the truncation [Friedland & Torokhti, SIAM J. Matrix Anal. Appl. 29(2):656-659, 2007;
Allen et al., J. Amer. Statist. Assoc. 109(505):145-159, 2014].  With C_g = I it reduces to SVD-LLM.

Every rule returns a ``Spectrum`` whose left factor is already "un-whitened" on the output side, so
the truncated pair is always  A = U_r sqrt(S_r),  B = sqrt(S_r) V_r^T R^{-1}  with R the right
whitening factor (L_x, or None for the identity).  The square-root split of the singular values
between the two factors follows ASVD [Yuan et al., arXiv:2312.05821] / SVD-LLM.
"""
from typing import NamedTuple, Optional
import torch


class Spectrum(NamedTuple):
    U: torch.Tensor           # (d_out, n)  output-side factor, de-whitened (L_g^{-T} U~ or D^{-1} U~ or U)
    S: torch.Tensor           # (n,)        singular values of the whitened matrix, descending
    Vh: torch.Tensor          # (n, d_in)   right singular vectors of the whitened matrix
    R: Optional[torch.Tensor] # (d_in, d_in) right whitening factor L_x (lower-triangular) or None (= I)
    shape: tuple              # dense weight shape: (d_out, d_in) or (C_out, C_in, K)


def damped_cholesky(C: torch.Tensor, eps: float, max_tries: int = 8):
    """Cholesky of C + ridge*I with ridge = eps * mean(diag C); the ridge is raised x10 until it succeeds.
    ``eps`` = 1e-3 here (0.1 % of the mean diagonal).  OBD-LLM uses 10 %; SVD-LLM's paper is silent."""
    ridge = eps * C.diagonal().mean().clamp_min(1e-30)
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(C + ridge * torch.eye(C.shape[0], device=C.device, dtype=C.dtype)), float(ridge)
        except Exception:
            ridge = ridge * 10
    raise RuntimeError("Cholesky failed after damping escalation")


def input_whitened_svd(W: torch.Tensor, C_x: torch.Tensor, eps: float = 1e-3) -> Spectrum:
    """SVD-LLM: whiten the input side only (float32, as in the reference implementation)."""
    shape = tuple(W.shape)
    W2 = W.reshape(shape[0], -1).float()
    Lx, _ = damped_cholesky(C_x.float(), eps, max_tries=5)
    U, S, Vh = torch.linalg.svd(W2 @ Lx, full_matrices=False)
    return Spectrum(U, S, Vh, Lx, shape)


def two_sided_svd(W: torch.Tensor, C_x: torch.Tensor, C_g: torch.Tensor, eps: float = 1e-3):
    """OBD-LLM / Friedland-Torokhti: whiten both sides, SVD, un-whiten the left factor (float64).

    W ~= L_g^{-T} U~_r S~_r V~_r^T L_x^{-1};  the component loss of dropping index i is S~_i^2 / 2.
    Returns (Spectrum with U = L_g^{-T} U~, ridge actually used for C_g)."""
    shape = tuple(W.shape)
    W2 = W.reshape(shape[0], -1)
    Lx, _ = damped_cholesky(C_x.float(), eps, max_tries=5)               # same L_x as SVD-LLM
    Lg, ridge = damped_cholesky(C_g.to(torch.float64), eps)
    Wt = Lg.T @ (W2.to(torch.float64) @ Lx.to(torch.float64))            # L_g^T W L_x
    Ut, St, Vht = torch.linalg.svd(Wt, full_matrices=False)
    U = torch.linalg.solve_triangular(Lg.T, Ut, upper=True)               # L_g^{-T} U~
    return Spectrum(U.float(), St.float(), Vht.float(), Lx, shape), ridge


def fisher_weighted_svd(W: torch.Tensor, I_row: torch.Tensor, clamp: float = 1e-6) -> Spectrum:
    """FWSVD as published: SVD(D W) with D = diag(sqrt(I_row / mean)), no input statistics.

    W ~= D^{-1} U~_r S~_r V~_r^T.  Rows with I_row < clamp * mean are clamped (the reference code
    guards against zero-Fisher rows the same way).  The per-matrix normalization by the mean keeps
    the singular values comparable across matrices for the shared budget allocation."""
    shape = tuple(W.shape)
    W2 = W.reshape(shape[0], -1).to(torch.float64)
    Ic = I_row.to(torch.float64).clamp_min(clamp * I_row.to(torch.float64).mean())
    d = (Ic / Ic.mean()).sqrt()
    Ut, St, Vht = torch.linalg.svd(d.unsqueeze(1) * W2, full_matrices=False)
    return Spectrum((Ut / d.unsqueeze(1)).float(), St.float(), Vht.float(), None, shape)


def to_factors(sp: Spectrum, r: int):
    """Truncate to rank r and return the factor pair (A: d_out x r, B: r x d_in), W ~= A @ B.

    A = U_r sqrt(S_r),  B = sqrt(S_r) V_r^T R^{-1}  (R^{-1} applied by a triangular solve)."""
    sq = sp.S[:r].sqrt()
    A = (sp.U[:, :r] * sq.unsqueeze(0)).contiguous()
    Vr = sp.Vh[:r, :]
    if sp.R is not None:
        Vr = torch.linalg.solve_triangular(sp.R.T, Vr.T, upper=True).T   # V_r^T L_x^{-1}
    B = (sq.unsqueeze(1) * Vr).contiguous()
    return A, B


def equal_norm_resplit(A: torch.Tensor, B: torch.Tensor):
    """Re-split the pair so that ||A||_F = ||B||_F while A @ B is unchanged (gauge freedom of the pair).

    s = sqrt(||A|| / ||B||),  A <- A / s,  B <- B * s.  This is the equal-norm convention of
    PiSSA [Meng et al., NeurIPS 2024] and RefLoRA [Zhang et al., NeurIPS 2025].  It is needed here
    because the de-whitening inverse L_g^{-T} lands on A alone and, for a small-class discriminative
    loss, makes ||A|| / ||B|| ~ 1e7; recovery fine-tuning from that split diverges (paper, Fig. 3)."""
    nA, nB = A.double().norm(), B.double().norm()
    s = (nA / nB).sqrt().item()
    return (A / s).to(A.dtype), (B * s).to(B.dtype), s


PAIRS = (("lr_A.weight", "lr_B.weight"), ("conv_A.weight", "conv_B.weight"))


def resplit_state_dict(sd):
    """:func:`equal_norm_resplit` applied to every factor pair of a whole checkpoint, in place.

    Used by ``scripts/resplit_factors.py`` to re-split a checkpoint that was built with
    ``--no_resplit`` (or by older code) without recomputing the factorization.  Returns
    (state_dict, ||A||/||B|| before, ||A||/||B|| after, max relative change of ||A|| ||B||)."""
    before, after, maxrel = [], [], 0.0
    for ka in list(sd):
        for sa, sb in PAIRS:
            if not ka.endswith(sa):
                continue
            kb = ka[: -len(sa)] + sb
            if kb not in sd:
                continue
            A, B = sd[ka].double(), sd[kb].double()
            nA, nB = A.norm().item(), B.norm().item()
            s = (nA / nB) ** 0.5
            A2, B2 = A / s, B * s
            before.append(nA / nB)
            after.append(A2.norm().item() / B2.norm().item())
            # the composition is bilinear in (A, B): (A/s)(sB) = AB exactly, for linear and conv alike
            maxrel = max(maxrel, abs((A2.norm() * B2.norm() - A.norm() * B.norm()).item()) / max(1e-30, nA * nB))
            sd[ka], sd[kb] = A2.to(sd[ka].dtype), B2.to(sd[kb].dtype)
    return sd, before, after, maxrel


def reconstruct(sp: Spectrum, r: int) -> torch.Tensor:
    """Dense rank-r reconstruction (for diagnostics)."""
    A, B = to_factors(sp, r)
    return A @ B
