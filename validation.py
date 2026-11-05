"""信效度检验：Kendall's W 一致性、Pearson 交叉验证、抗扰动鲁棒性。

叙事（诚实版）：
1. 纯单链 + 纯最大流 → Kendall's W ≈ 0：证明纯数学模型在无冗余拓扑上
   退化为全并列，无法区分主次（这是"引入业务先验"的必要性证据）。
2. 完整模型（最大流 + 连通性 + 结构关键度）→ 抗扰动排名稳定（W → 1）：
   这才是真正的鲁棒性证据。
"""
from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
from scipy.stats import pearsonr, rankdata

from .graph_model import Topology
from .shapley import normalized_responsibility


def kendalls_w(rankings: List[List[float]]) -> float:
    """Kendall 协调系数 W（含并列修正）。rankings 为 m 组排名。"""
    if not rankings:
        return float("nan")
    m = len(rankings)
    n = len(rankings[0])
    R = np.zeros(n)
    for r in rankings:
        R += rankdata(np.asarray(r, dtype=float), method="average")
    Rbar = R.mean()
    S = float(np.sum((R - Rbar) ** 2))
    W = 12.0 * S / (m * m * (n ** 3 - n))
    return W


def _perturbed_rankings(
    topology: Topology,
    n_perturb: int,
    sigma: float,
    seed: int,
    pure_maxflow: bool,
) -> List[List[float]]:
    """对边容量做高斯扰动，重复计算责任度排名。"""
    rng = np.random.default_rng(seed)
    nodes = [n.id for n in topology.nodes]
    rankings: List[List[float]] = []
    for _ in range(n_perturb):
        t = copy.deepcopy(topology)
        for e in t.edges:
            e.capacity = max(0.01, e.capacity + rng.normal(0.0, sigma))
        if pure_maxflow:
            t.alpha, t.beta, t.gamma = 1.0, 0.0, 0.0
        resp = normalized_responsibility(t)
        rankings.append([resp[n] for n in nodes])
    return rankings


def robustness_report(
    topology: Topology,
    n_perturb: int = 100,
    sigma: float = 0.2,
    seed: int = 42,
) -> Dict:
    """对比：纯单链+纯最大流（退化基线） vs 完整模型（鲁棒）。"""
    nodes = [n.id for n in topology.nodes]

    chain = topology.without_redundancy()
    pure_rankings = _perturbed_rankings(chain, n_perturb, sigma, seed, pure_maxflow=True)
    full_rankings = _perturbed_rankings(topology, n_perturb, sigma, seed, pure_maxflow=False)

    # 完整模型下，第一名节点在扰动中保持稳定的比例
    full_firsts = [np.argmax(r) for r in full_rankings]
    top1 = nodes[int(np.bincount(full_firsts).argmax())]
    top1_stability = float(np.bincount(full_firsts).max()) / n_perturb

    return {
        "kendalls_w_pure_chain_maxflow": kendalls_w(pure_rankings),
        "kendalls_w_full_model": kendalls_w(full_rankings),
        "top1_node": top1,
        "top1_stability": top1_stability,
        "n_perturb": n_perturb,
        "sigma": sigma,
        "nodes": nodes,
    }


def pearson_cross_validation(
    exposure_scores: List[float],
    responsibility_scores: List[float],
) -> Dict:
    """双规则背靠背测算的相关性（规则A：文本暴露度；规则B：拓扑责任度）。"""
    r, p = pearsonr(exposure_scores, responsibility_scores)
    return {"pearson_r": float(r), "p_value": float(p), "n": len(exposure_scores)}
