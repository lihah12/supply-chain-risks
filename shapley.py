"""精确 Shapley 值：供应链节点责任度归因。

对 n 个节点，枚举全部 2^n 个子集、按组合权重累加边际贡献，
得到每个节点的精确 Shapley 责任度（n 较小时比蒙特卡罗更严谨）。
"""
from __future__ import annotations

import math
from typing import Callable, Dict, FrozenSet, Iterable

from .graph_model import Topology


def exact_shapley(
    node_ids: Iterable[str],
    v: Callable[[FrozenSet[str]], float],
) -> Dict[str, float]:
    """精确 Shapley 值。

    边际贡献公式：
        φ_i = Σ_{S ⊆ N\\{i}} [ |S|! (n-|S|-1)! / n! ] * [ v(S∪{i}) - v(S) ]
    """
    nodes = list(node_ids)
    n = len(nodes)
    sh = {node: 0.0 for node in nodes}

    for bits in range(1 << n):
        s = frozenset(nodes[i] for i in range(n) if bits & (1 << i))
        s_size = len(s)
        if s_size == n:  # 全集没有可加入的节点
            continue
        v_s = v(s)
        weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / math.factorial(n)
        for node in nodes:
            if node in s:
                continue
            v_su = v(s | {node})
            sh[node] += weight * (v_su - v_s)
    return sh


def responsibility(topology: Topology) -> Dict[str, float]:
    """各节点责任度（Shapley 值，即边际破坏力的期望）。"""
    return exact_shapley([n.id for n in topology.nodes], topology.characteristic)


def normalized_responsibility(topology: Topology) -> Dict[str, float]:
    """归一化责任度（最大为 1），与暴露度同尺度，便于四象限对比。

    用"相对最关键节点"而非"和为1"：和为1会让最高责任节点也只有 ~1/n，
    无法用固定阈值 0.5 划出高低。最大归一化下最高责任节点=1.0。
    """
    sh = responsibility(topology)
    mx = max(sh.values()) if sh else 0.0
    if mx <= 0:
        return {k: 0.0 for k in sh}
    return {k: v / mx for k, v in sh.items()}


def responsibility_decomposition(topology: Topology) -> Dict[str, Dict[str, float]]:
    """按特征函数三部分分解责任度，解释"为什么这个节点排第一"。

    利用 Shapley 值的线性性：
        φ(v) = alpha·φ(maxflow) + beta·φ(connectivity) + gamma·φ(criticality)
    其中 criticality 是加性项，其 Shapley 恰为该节点归一化关键度本身。
    全部除以最大 total，使"合计"列与 normalized_responsibility 同尺度。
    """
    nodes = [n.id for n in topology.nodes]
    sh_flow = exact_shapley(nodes, topology.max_flow)
    sh_conn = exact_shapley(nodes, topology.connectivity)
    sh_crit = exact_shapley(nodes, topology.structural_criticality)
    raw = {
        n: topology.alpha * sh_flow[n] + topology.beta * sh_conn[n] + topology.gamma * sh_crit[n]
        for n in nodes
    }
    max_total = max(raw.values()) if raw else 0.0
    if max_total <= 0:
        max_total = 1.0
    out: Dict[str, Dict[str, float]] = {}
    for n in nodes:
        flow = topology.alpha * sh_flow[n] / max_total
        conn = topology.beta * sh_conn[n] / max_total
        crit = topology.gamma * sh_crit[n] / max_total
        out[n] = {
            "flow": flow,
            "connectivity": conn,
            "criticality": crit,
            "total": flow + conn + crit,
        }
    return out
