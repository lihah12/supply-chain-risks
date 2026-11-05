# 供应链风险全景测度系统

基于 **LLM 多智能体博弈归因** 的供应链风险测度系统。核心是三层**硬解耦**：

| 层        | 职责                    | 技术                                   |
| -------- | --------------------- | ------------------------------------ |
| 前端多智能体   | 把非结构化舆情/气象文本转成**暴露度** | Extractor → Feedback（四维批判）→ Revision |
| 后端博弈归因   | 算每个节点失效的**责任度**       | 精确 Shapley 值（最大流 + 连通性惩罚 + 结构关键度）    |
| RAG 解释引擎 | 在强约束 JSON 下做**业务解释**  | DeepSeek，禁编造数字                       |

**设计原则：LLM 绝不参与数学计算。** 后端是纯 Python 图论/博弈模型，LLM 只在强约束上下文中"翻译"结果，从根上规避数值幻觉。

## 目录结构

```
app.py                  # Streamlit 主界面（5 个标签页）
config/topology.json    # 供应链拓扑（节点/边/容量/关键度）—— 插槽 1
config/corpus.json      # 舆情事件语料 —— 插槽 2
src/llm.py             # DeepSeek 客户端（OpenAI 兼容）+ Key 解析
src/agents.py          # Extractor / Feedback / Revision 多智能体
src/exposure.py        # 事件级严重度 → 节点暴露度聚合
src/graph_model.py     # 有向网络 + 最大流 + 连通性 + 特征函数
src/shapley.py         # 精确 Shapley 值 + 责任度分解
src/matrix.py          # 暴露度×责任度 四象限分类
src/rag.py             # 强约束 RAG 解释引擎
src/validation.py      # Kendall's W 鲁棒性 + 两维独立性检验
```

## 本地运行

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # 填入你的 DeepSeek Key
streamlit run app.py
```

> 没有 API Key 也能跑：暴露度会走**确定性规则兜底**（语料中的锚点严重度 + 时间衰减），  
> 全部结果可离线复现。配置 Key 后打开侧边栏的「启用 DeepSeek 多智能体分析」开关即可切换。

## 拿到免费永久链接（Streamlit Community Cloud）

我无法替你注册账号/推送代码，这一步需要你自己的 GitHub 账号，操作如下：

1. **推到 GitHub**：把本目录初始化为 git 仓库并 push 到一个公开/私有 repo（`requirements.txt` 已在仓库里）。
2. 打开 <https://share.streamlit.io>，用 GitHub 登录。
3. 点 **New app** → 选你的 repo、分支、主文件填 `app.py` → **Deploy**。
4. 部署成功后进入 **App settings → Secrets**，添加：
   ```toml
   DEEPSEEK_API_KEY = "sk-你的key"
   ```
5. 保存即自动重启，你会得到一个形如 `https://你的app.streamlit.app` 的**永久公网链接**。

> 备选：Hugging Face Spaces（同样免费、支持 Streamlit，把仓库推到 Space 即可）。

## 在页面内上传自有数据（免改代码）

除了替换仓库里的 `config/*.json`，你也可以直接在已部署的页面里上传自己的数据，实时分析：

1. 打开左侧边栏的 **📥 导入自有数据**。
2. 上传 **供应链拓扑**（`topology.json`，schema 同 `config/topology.json`）和/或 **风险事件语料**（`corpus.json` 或 `.csv`）。
   - 只传其一即可，未上传的维度沿用默认示例数据。
   - 拓扑若缺 `source/sink/α/β/γ`，会按默认 `source=N1`、`sink=N6`、`α=0.5 β=0.2 γ=0.3` 兜底。
3. 不确定格式？点 **⬇️ 拓扑模板 / 语料模板(JSON) / 语料模板(CSV)** 下载示例文件照填即可。

### 语料 CSV 格式

| 列                            | 是否必需 | 说明                   |
| ---------------------------- | ---- | -------------------- |
| `date`                       | 必需   | 事件日期 `YYYY-MM-DD`    |
| `node`                       | 必需   | 拓扑中的节点编号，如 `N2`      |
| `severity`                   | 必需   | 严重度 0~1（0=无影响，1=灾难性） |
| `title`                      | 必需   | 事件描述文本               |
| `id` / `source` / `category` | 可选   | 事件编号 / 来源 / 类别       |

`time_window` 由日期范围自动推断，无需手写。

> **节点对齐校验**：语料里引用了拓扑中不存在的节点时，相关事件会被剔除并给出 ⚠️ 提示，不会崩溃。

## 换数据，不换引擎

换一家企业，**不改一行底层算法代码**，只需替换三个 JSON 插槽：

1. **拓扑**（`config/topology.json`）：节点清单、流向、相对容量比例、结构关键度。
2. **语料**（`config/corpus.json`）：目标产品的公开投诉/召回/气象事件。
3. **大模型 Prompt**（`src/rag.py` 顶部）：把系统提示词从"你是液态奶专家"换成对应行业专家。

最大流 + 精确 Shapley + 四象限引擎完全通用。

## 数据与合规

- 拓扑容量是**相对归一化映射**，不代表企业绝对机密货量。
- 语料为**结构化硬编码锚点**，完全可复现，绝不随机抓取/插值伪造。
- 示例拓扑（伊利液态奶 6 节点）与语料为演示用种子数据，替换为真实公开数据即可用于正式测度。
