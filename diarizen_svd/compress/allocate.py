#!/usr/bin/env python3
"""Global rank allocation under one parameter budget (water-filling over all compressed matrices).

Every rank-one component i of matrix m carries
    value  = S_{m,i}^2            (the whitened singular value squared: under the two-sided rule this
                                   is twice the second-order loss increase of discarding it)
    price  = cost_m = d_out + d_in (parameters added by one more rank; C_out + C_in*K for a conv)
and the budget  sum_m r_m cost_m <= P_full / rho - P_fixed  is filled greedily in decreasing order of
value / price.  For a linear price this greedy order is the Lagrangian relaxation of the knapsack and
is optimal for the stated value model.
"""
from collections import defaultdict


def rank_cost(shape) -> int:
    """Parameters per unit rank: d_out + d_in  (conv: C_out + C_in * K, i.e. the im2col width)."""
    return shape[0] + shape[1] if len(shape) == 2 else shape[0] + shape[1] * shape[2]


def dense_params(shape) -> int:
    return shape[0] * shape[1] if len(shape) == 2 else shape[0] * shape[1] * shape[2]


def rank_cap(shape, n_sv: int) -> int:
    """Largest rank for which the factorization is smaller than the dense matrix (break-even)."""
    return min(dense_params(shape) // rank_cost(shape), n_sv)


def water_fill(spectra: dict, P_full: int, P_fixed: int, ratio: float, floor: int = 1, round_to: int = 8):
    """Greedy budget allocation.

    spectra : {key: Spectrum}  (only .S and .shape are used)
    P_full  : parameters of the dense encoder
    P_fixed : parameters that are NOT factorized (conv0, norms, biases, positional conv, projections)
    ratio   : target compression ratio rho;  matrix budget = P_full / rho - P_fixed
    floor   : minimum rank reserved for every matrix before water-filling (1 = no floor)
    round_to: linear-map ranks are rounded to a multiple of this (kernel-friendly); conv ranks are not
    Returns {key: rank}.
    """
    budget = P_full / ratio - P_fixed
    lo = {k: min(floor, rank_cap(sp.shape, sp.S.numel())) for k, sp in spectra.items()}
    budget -= sum(lo[k] * rank_cost(spectra[k].shape) for k in spectra)
    if budget < 0:
        raise ValueError(f"ratio {ratio} infeasible: the rank floor alone exceeds the budget")
    comps = []
    for k, sp in spectra.items():
        s2 = (sp.S ** 2).tolist()
        cst = rank_cost(sp.shape)
        cap = rank_cap(sp.shape, sp.S.numel())
        comps.extend((s2[i] / cst, k) for i in range(lo[k], cap))
    comps.sort(key=lambda z: -z[0])
    add = defaultdict(int)
    for _, k in comps:
        c = rank_cost(spectra[k].shape)
        if budget >= c:
            add[k] += 1
            budget -= c
    ranks = {}
    for k, sp in spectra.items():
        r = lo[k] + add[k]
        if k[0] == "lin" and round_to > 1:
            r = max(round_to, int(round(r / round_to)) * round_to)
        mx = min(sp.shape[0], sp.shape[1]) if len(sp.shape) == 2 else min(sp.shape[0], sp.shape[1] * sp.shape[2])
        ranks[k] = min(r, mx)
    return ranks


def lowrank_params(spectra: dict, ranks: dict) -> int:
    return sum(ranks[k] * rank_cost(sp.shape) for k, sp in spectra.items())
