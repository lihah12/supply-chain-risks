"""暴露度 × 责任度 二维矩阵，四象限分类。"""
from __future__ import annotations

from typing import Dict, List

QUADRANTS = {
    "显性核心": "高暴露-高责任：需重点防御，应急预案优先保障",
    "隐性关键": "低暴露-高责任：潜伏炸弹，防微杜渐，补冗余/压力测试",
    "局部扰动": "高暴露-低责任：虚假警报，避免应急资源错配",
    "常规节点": "低暴露-低责任：常规监控即可",
}


def classify(
    exposure: Dict[str, float],
    responsibility: Dict[str, float],
    threshold: float = 0.5,
) -> List[Dict]:
    """将节点映射到四象限。

    要求 exposure / responsibility 均已归一化到 [0,1] 且量纲一致。
    返回每个节点的坐标、象限与建议。
    """
    rows = []
    for nid in exposure:
        e = float(exposure[nid])
        r = float(responsibility.get(nid, 0.0))
        if e >= threshold and r >= threshold:
            quad = "显性核心"
        elif e < threshold and r >= threshold:
            quad = "隐性关键"
        elif e >= threshold and r < threshold:
            quad = "局部扰动"
        else:
            quad = "常规节点"
        rows.append(
            {
                "node": nid,
                "exposure": e,
                "responsibility": r,
                "quadrant": quad,
                "advice": QUADRANTS[quad],
            }
        )
    return rows
