"""RAG 决策解释引擎：在强约束 JSON 上下文中让 DeepSeek 做业务翻译。

核心防幻觉原则：LLM 只做"基于事实的解释"，不参与任何数学计算；
上下文是打包好的强约束 JSON，Prompt 明确禁止编造 JSON 之外的数字与事实。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .llm import DeepSeekClient

GUARD = (
    "你只能基于下方【上下文】中的 JSON 数据回答，严禁编造任何不在其中的数字、"
    "节点或事实；若数据未覆盖某问题，请明确说『该数据未覆盖』。"
)

EXPLAIN_SYSTEM = (
    "你是供应链风险管理专家，面向管理层输出决策建议。" + GUARD + "\n"
    "请输出：①整体风险判断 ②四象限解读（重点点出『隐性关键/潜伏炸弹』节点）"
    "③按 红/橙/黄/蓝 四级给出应急策略。"
)


def build_context(
    topology,
    responsibility: Dict[str, float],
    exposure: Dict[str, float],
    matrix_rows: List[Dict[str, Any]],
    validation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """把底层拓扑、Shapley 责任度、暴露度打包成强约束 JSON。"""
    nodes = [
        {"id": n.id, "name": n.name, "role": n.role, "criticality": n.criticality}
        for n in topology.nodes
    ]
    edges = [{"src": e.src, "dst": e.dst, "channel": e.channel} for e in topology.edges]
    ranking = sorted(responsibility, key=responsibility.get, reverse=True)
    ctx = {
        "topology": {"nodes": nodes, "edges": edges},
        "responsibility": {k: round(v, 4) for k, v in responsibility.items()},
        "exposure": {k: round(v, 4) for k, v in exposure.items()},
        "matrix": matrix_rows,
        "responsibility_ranking": ranking,
    }
    if validation is not None:
        ctx["validation"] = {
            "kendalls_w_full_model": round(validation.get("kendalls_w_full_model", 0.0), 4),
            "top1_node": validation.get("top1_node"),
        }
    return ctx


def explain(client: DeepSeekClient, context: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": f"【上下文】\n{_dumps(context)}\n\n请生成决策建议。"},
    ]
    return client.chat(messages, temperature=0.2)


def answer(client: DeepSeekClient, context: Dict[str, Any], question: str) -> str:
    system = "你是供应链风险管理助手。" + GUARD
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【上下文】\n{_dumps(context)}\n\n问题：{question}"},
    ]
    return client.chat(messages, temperature=0.1)


def _dumps(obj: Dict[str, Any]) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False, indent=2)
