"""供应链风险全景测度系统 —— 多智能体博弈归因。

三层硬解耦：
- 前端多智能体（Extractor/Feedback/Revision）：非结构化文本 -> 暴露度
- 后端图论/合作博弈（最大流 + 连通性 + 结构关键度）：精确 Shapley 责任度
- DeepSeek RAG 解释引擎：强约束 JSON 下生成决策建议

运行：streamlit run app.py
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.exposure import assess_corpus, compute_node_exposure, load_corpus  # noqa: E402
from src.graph_model import Topology  # noqa: E402
from src.llm import DeepSeekClient, resolve_api_key  # noqa: E402
from src.matrix import classify  # noqa: E402
from src.rag import answer, build_context, explain  # noqa: E402
from src.shapley import normalized_responsibility, responsibility_decomposition  # noqa: E402
from src.validation import pearson_cross_validation, robustness_report  # noqa: E402

# ------------------------------------------------------------------ 调色板
# 来自已验证的参考色板（dataviz skill），状态色用于四象限，顺序蓝用于量级。
QUADRANT_COLORS = {
    "显性核心": "#d03b3b",  # critical
    "隐性关键": "#ec835a",  # serious
    "局部扰动": "#fab219",  # warning
    "常规节点": "#0ca30c",  # good
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"

st.set_page_config(page_title="供应链风险全景测度", page_icon="📊", layout="wide")


# ------------------------------------------------------------------ 缓存（默认数据插槽）
@st.cache_data(show_spinner=False)
def load_topology() -> Topology:
    return Topology.from_json(str(ROOT / "config" / "topology.json"))


@st.cache_data(show_spinner=False)
def load_corpus_data() -> dict:
    return load_corpus(str(ROOT / "config" / "corpus.json"))


@st.cache_resource(show_spinner=False)
def get_client(api_key: str):
    if not api_key:
        return None
    return DeepSeekClient(api_key)


# ------------------------------------------------------------------ 默认数据与导入辅助
# 默认示例数据用于未上传时的兜底，也作为模板下载源。
DEFAULT_TOPOLOGY = load_topology()
DEFAULT_CORPUS = load_corpus_data()

_TOPO_TPL = (ROOT / "config" / "topology.json").read_text(encoding="utf-8")
_CORPUS_TPL = (ROOT / "config" / "corpus.json").read_text(encoding="utf-8")


def _read_text(uploaded) -> str:
    return uploaded.getvalue().decode("utf-8")


def infer_ref_date(corpus: dict) -> str:
    """暴露度时间衰减的参考日期：优先用语料 time_window 终点，否则取事件最晚日期。"""
    tw = corpus.get("time_window") or [None, None]
    if tw and tw[1]:
        return str(tw[1])
    dates = [e.get("date") for e in corpus.get("events", []) if e.get("date")]
    return max(dates) if dates else "2026-09-30"


def parse_events_csv(text: str) -> dict:
    """把事件 CSV 解析成语料 dict。

    必需列：date, node, severity, title
    可选列：id, source, category, scope
    time_window 由日期范围自动推断。
    """
    reader = csv.DictReader(io.StringIO(text))
    required = {"date", "node", "severity"}
    if not required.issubset({c.lower() for c in (reader.fieldnames or [])}):
        raise ValueError(f"CSV 缺少必需列，需要：{sorted(required)}（实际表头：{reader.fieldnames}）")
    events: list = []
    dates: list = []
    for i, row in enumerate(reader, 1):
        date = (row.get("date") or "").strip()
        node = (row.get("node") or "").strip()
        try:
            severity = float(row.get("severity") or 0)
        except (TypeError, ValueError):
            severity = 0.0
        events.append(
            {
                "id": (row.get("id") or f"U{i:03d}").strip(),
                "date": date,
                "source": (row.get("source") or "").strip(),
                "node": node,
                "category": (row.get("category") or "").strip(),
                "severity": max(0.0, min(1.0, severity)),
                "title": (row.get("title") or "").strip(),
            }
        )
        if date:
            dates.append(date)
    if not events:
        raise ValueError("CSV 中未解析到任何事件行")
    time_window = [min(dates), max(dates)] if dates else ["2000-01-01", "2000-01-01"]
    return {"time_window": time_window, "events": events}


def make_csv_template() -> str:
    """由默认语料生成一份 CSV 模板，方便用户照格式填写。"""
    buf = io.StringIO()
    cols = ["id", "date", "source", "node", "category", "severity", "title"]
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for e in DEFAULT_CORPUS.get("events", []):
        w.writerow({k: e.get(k, "") for k in cols})
    return buf.getvalue()


def resolve_inputs(topo_file, corpus_file):
    """返回 (topology, corpus, status)。未上传则回退默认示例数据。"""
    topology = DEFAULT_TOPOLOGY
    corpus = DEFAULT_CORPUS
    status = {"topo": "默认示例拓扑", "corpus": "默认示例语料"}
    if topo_file is not None:
        try:
            topology = Topology.from_dict(json.loads(_read_text(topo_file)))
            status["topo"] = f"已导入：{topo_file.name}"
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ 拓扑文件解析失败：{e}")
    if corpus_file is not None:
        try:
            raw = _read_text(corpus_file)
            if corpus_file.name.lower().endswith(".csv"):
                corpus = parse_events_csv(raw)
            else:
                corpus = json.loads(raw)
                if not isinstance(corpus, dict) or "events" not in corpus:
                    raise ValueError("语料 JSON 必须包含 'events' 数组")
            status["corpus"] = f"已导入：{corpus_file.name}"
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ 语料文件解析失败：{e}")
    return topology, corpus, status

# 侧边栏
with st.sidebar:
    st.title("📊 供应链风险全景测度")
    st.caption("基于 LLM 的多智能体博弈归因")

    api_key = resolve_api_key()
    if not api_key:
        manual = st.text_input("DeepSeek API Key（可选）", type="password", help="不填则用确定性规则兜底")
        api_key = manual.strip() or None
    if api_key:
        st.success("✅ 已接入 DeepSeek")
    else:
        st.info("ℹ️ 未配置 Key，暴露度走确定性规则兜底（可离线复现）")

    st.divider()
    threshold = st.slider("四象限阈值", 0.1, 0.9, 0.5, 0.05)

    st.divider()
    st.subheader("📥 导入自有数据")
    st.caption("上传你自己的拓扑或事件语料，即时替换默认示例进行分析（可只传其一）。")
    topo_file = st.file_uploader(
        "供应链拓扑 (topology.json)",
        type=["json"],
        help="节点/边/容量/关键度，schema 同 config/topology.json（source/sink/α/β/γ 可选）",
    )
    corpus_file = st.file_uploader(
        "风险事件语料 (corpus.json / .csv)",
        type=["json", "csv"],
        help="JSON 同 config/corpus.json；CSV 需含列 date,node,severity,title（可选 id,source,category）",
    )
    _c1, _c2, _c3 = st.columns(3)
    with _c1:
        st.download_button("⬇️ 拓扑模板", data=_TOPO_TPL, file_name="topology_template.json")
    with _c2:
        st.download_button("⬇️ 语料模板(JSON)", data=_CORPUS_TPL, file_name="corpus_template.json")
    with _c3:
        st.download_button("⬇️ 语料模板(CSV)", data=make_csv_template(), file_name="corpus_template.csv")

# ------------------------------------------------------------------ 计算（支持导入自有数据）
topology, corpus, input_status = resolve_inputs(topo_file, corpus_file)
node_ids = [n.id for n in topology.nodes]
node_names = {n.id: n.name for n in topology.nodes}
responsibility = normalized_responsibility(topology)
decomp = responsibility_decomposition(topology)
validation = robustness_report(topology)

# ------------------------------------------------------------------ 顶部
st.title("供应链风险全景测度系统")
st.caption(
    "硬解耦三层：前端多智能体读文字算暴露度 · 后端图论博弈算责任度 · DeepSeek 强约束 RAG 做解释。"
)
st.caption(
    f"特征函数权重 α={topology.alpha} β={topology.beta} γ={topology.gamma}　|　"
    f"数据来源：拓扑【{input_status['topo']}】 · 语料【{input_status['corpus']}】"
)

# 暴露度（LLM 或兜底）
use_llm = False
if api_key:
    use_llm = st.toggle("启用 DeepSeek 多智能体分析（否则用确定性兜底）", value=False)

if use_llm and api_key:
    with st.spinner("多智能体提取→批判→修正中…"):
        client = get_client(api_key)
        event_results = assess_corpus(corpus, client)
else:
    event_results = assess_corpus(corpus, None)

# 语料节点与拓扑对齐校验：拓扑里不存在的节点相关事件直接剔除
_bad_nodes = {e["node"] for e in event_results if e["node"] not in node_ids}
if _bad_nodes:
    st.warning(f"⚠️ 语料中节点 {', '.join(sorted(_bad_nodes))} 不在拓扑内，相关事件已剔除。")
    event_results = [e for e in event_results if e["node"] in node_ids]

ref_date = infer_ref_date(corpus)
exposure = compute_node_exposure(event_results, node_ids, ref_date=ref_date)

matrix_rows = classify(exposure, responsibility, threshold=threshold)
context = build_context(topology, responsibility, exposure, matrix_rows, validation)

# ------------------------------------------------------------------ 标签页
tab_matrix, tab_attrib, tab_exposure, tab_rag, tab_valid = st.tabs(
    ["📊 全景矩阵", "🔢 责任归因", "📰 暴露度分析", "🤖 智能决策解释", "✅ 信效度检验"]
)

# ---- 全景矩阵 ----
with tab_matrix:
    st.subheader("暴露度 × 责任度 四象限矩阵")
    st.caption("横轴暴露度（读文字），纵轴责任度（算网络）。两维独立才见'潜伏炸弹'。")

    fig = go.Figure()
    for quad, color in QUADRANT_COLORS.items():
        sub = [r for r in matrix_rows if r["quadrant"] == quad]
        if not sub:
            continue
        fig.add_trace(
            go.Scatter(
                x=[r["exposure"] for r in sub],
                y=[r["responsibility"] for r in sub],
                mode="markers+text",
                name=quad,
                text=[f"{r['node']} {node_names[r['node']]}" for r in sub],
                textposition="top center",
                textfont=dict(size=11, color=INK),
                marker=dict(size=16, color=color, line=dict(width=2, color=SURFACE)),
            )
        )
    fig.add_hline(y=threshold, line_dash="dash", line_color=AXIS)
    fig.add_vline(x=threshold, line_dash="dash", line_color=AXIS)
    fig.update_layout(
        height=520,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, Segoe UI, sans-serif", color=INK),
        xaxis=dict(title="暴露度 →", range=[-0.05, 1.05], gridcolor=GRID, zeroline=False),
        yaxis=dict(title="责任度 →", range=[-0.05, 1.05], gridcolor=GRID, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 节点明细")
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "节点": f"{r['node']} {node_names[r['node']]}",
                "暴露度": round(r["exposure"], 3),
                "责任度": round(r["responsibility"], 3),
                "象限": r["quadrant"],
                "建议": r["advice"],
            }
            for r in sorted(matrix_rows, key=lambda r: -r["responsibility"])
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

# ---- 责任归因 ----
with tab_attrib:
    st.subheader("节点责任度（精确 Shapley 值）")
    st.caption("对全部 2^6 子集按组合权重累加边际贡献，非蒙特卡罗采样。")

    order = sorted(node_ids, key=lambda x: responsibility[x], reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[f"{n} {node_names[n]}" for n in order],
            y=[responsibility[n] for n in order],
            marker_color=BLUE,
            text=[f"{responsibility[n]:.3f}" for n in order],
            textposition="outside",
            textfont=dict(color=INK),
        )
    )
    fig.update_layout(
        height=380,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, Segoe UI, sans-serif", color=INK),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=dict(color=INK)),
        yaxis=dict(title="责任度", gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 为什么它排第一？—— 责任度三因子分解")
    st.caption("Shapley 线性性：责任度 = α·流量损失 + β·连通性惩罚 + γ·结构关键度")
    decomp_order = sorted(node_ids, key=lambda x: decomp[x]["total"], reverse=True)
    import pandas as pd

    ddf = pd.DataFrame(
        [
            {
                "节点": f"{n} {node_names[n]}",
                "流量损失": round(decomp[n]["flow"], 4),
                "连通性惩罚": round(decomp[n]["connectivity"], 4),
                "结构关键度": round(decomp[n]["criticality"], 4),
                "合计": round(decomp[n]["total"], 4),
            }
            for n in decomp_order
        ]
    )
    st.dataframe(ddf, use_container_width=True, hide_index=True)

# ---- 暴露度分析 ----
with tab_exposure:
    st.subheader("节点暴露度（文本 → 分数）")
    st.caption(f"引擎：{'DeepSeek 多智能体' if use_llm else '确定性规则兜底（可离线复现）'}")

    eorder = sorted(node_ids, key=lambda x: exposure[x], reverse=True)
    fig = go.Figure(
        go.Bar(
            x=[f"{n} {node_names[n]}" for n in eorder],
            y=[exposure[n] for n in eorder],
            marker_color=ORANGE,
            text=[f"{exposure[n]:.3f}" for n in eorder],
            textposition="outside",
            textfont=dict(color=INK),
        )
    )
    fig.update_layout(
        height=380,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, Segoe UI, sans-serif", color=INK),
        xaxis=dict(gridcolor=GRID, zeroline=False),
        yaxis=dict(title="暴露度", gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED)),
        margin=dict(l=60, r=20, t=20, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 事件级证据（可审计）")
    import pandas as pd

    edf = pd.DataFrame(
        [
            {
                "事件": e["event_id"],
                "日期": e["date"],
                "来源": e["source"],
                "节点": f"{e['node']} {node_names[e['node']]}",
                "严重度": round(e["severity"], 3),
                "范围": e["scope"],
                "引擎": e["engine"],
                "证据锚点": e["evidence"],
            }
            for e in event_results
        ]
    )
    st.dataframe(edf, use_container_width=True, hide_index=True)

# ---- 智能决策解释 ----
with tab_rag:
    st.subheader("DeepSeek RAG 决策解释")
    st.caption("LLM 只在强约束 JSON 上下文内做业务翻译，不做数学计算，禁止编造数字。")

    if not api_key:
        st.warning("请先在侧边栏配置 DeepSeek API Key。")
    else:
        client = get_client(api_key)
        if st.button("生成管理层决策建议", type="primary"):
            with st.spinner("生成中…"):
                try:
                    text = explain(client, context)
                    st.session_state["explain_text"] = text
                except Exception as e:
                    st.error(f"调用失败：{e}")

        if "explain_text" in st.session_state:
            st.markdown("### 决策建议")
            st.markdown(st.session_state["explain_text"])

        st.divider()
        q = st.chat_input("针对某个节点或风险提问…")
        if q:
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
            st.session_state["messages"].append({"role": "user", "content": q})
            with st.spinner("回答中…"):
                try:
                    a = answer(client, context, q)
                except Exception as e:
                    a = f"调用失败：{e}"
            st.session_state["messages"].append({"role": "assistant", "content": a})

        for m in st.session_state.get("messages", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

# ---- 信效度检验 ----
with tab_valid:
    st.subheader("全链路信效度与抗扰动检验")
    val = validation

    st.markdown("#### 检验一：Kendall's W 协调系数（诚实版叙事）")
    c1, c2 = st.columns(2)
    c1.metric("纯单链 + 纯最大流（退化基线）", f"{val['kendalls_w_pure_chain_maxflow']:.4f}")
    c2.metric("完整模型（最大流+连通性+关键度）", f"{val['kendalls_w_full_model']:.4f}")
    st.caption(
        "解释：W≈0 表示纯数学模型在无冗余单链上完全并列、无法区分主次——这是"
        "必须引入业务先验的必要性证据；完整模型的 W 趋近 1 才是真正的排名鲁棒性证据。"
        f"扰动 {val['n_perturb']} 次后，第一名 {val['top1_node']} "
        f"（{node_names[val['top1_node']]}）稳定保持 {val['top1_stability']*100:.0f}%。"
    )

    st.markdown("#### 检验二：两维独立性检验（暴露度 vs 责任度）")
    exp_scores = [e["severity"] for e in event_results]
    resp_scores = [responsibility[e["node"]] for e in event_results]
    if len(exp_scores) >= 2:
        pr = pearson_cross_validation(exp_scores, resp_scores)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson r", f"{pr['pearson_r']:.4f}")
        c2.metric("p 值", f"{pr['p_value']:.4f}")
        c3.metric("样本 n", pr["n"])
    else:
        st.info("有效事件不足 2 条，无法计算两维相关性（请检查语料节点是否与拓扑对齐）。")
    st.caption(
        "解释：暴露度与责任度不呈显著正相关（甚至略负），说明这是两条相对独立的"
        "信息维度——这正是二维矩阵能分离出『低暴露-高责任』隐性关键节点的前提。"
        "下游消费端（零售/配送）更易上新闻，上游生产端（工厂/牧场）结构上更关键，"
        "二者错位才构成真正的风险盲区。"
    )

st.divider()
st.caption(
    "数据驱动：拓扑与语料均为可配置数据；可在左侧『导入自有数据』上传你的 JSON/CSV 直接分析，"
    "或替换 config 下的三个插槽，底层引擎不变。"
)
