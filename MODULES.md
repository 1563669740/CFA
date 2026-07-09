# CFA-Score 模块功能详解（v2.2）

---

## 目录

- [0. 项目概述与核心概念](#0-项目概述与核心概念)
- [1. models.py — 数据模型](#1-modelspy--数据模型)
- [2. deepseek.py — DeepSeek API 客户端](#2-deepseekpy--deepseek-api-客户端)
- [3. knowledge.py — 知识加载工具函数](#3-knowledgepy--知识加载工具函数)
- [4. adapter.py — LLM 适配器接口](#4-adapterpy--llm-适配器接口)
- [5. extractor.py — 规则锚点抽取器](#5-extractorpy--规则锚点抽取器)
- [6. retriever.py — 混合稀疏召回器（v2.2 新增）](#6-retrieverpy--混合稀疏召回器v22-新增)
- [7. semantic_index.py — 语义别名索引与候选召回](#7-semantic_indexpy--语义别名索引与候选召回)
- [8. llm_extractor.py — LLM 语义锚点抽取器](#8-llm_extractorpy--llm-语义锚点抽取器)
- [9. anchor_verifier.py — 锚点校验器](#9-anchor_verifierpy--锚点校验器)
- [10. intent_router.py — 意图路由器（v2.1 新增）](#10-intent_routerpy--意图路由器v21-新增)
- [11. engine.py — CFA-Score 核心引擎](#11-enginepy--cfa-score-核心引擎)
- [12. sanitizer.py — 安全改写器](#12-sanitizerpy--安全改写器)
- [13. gateway.py — 安全网关核心（v2.1 新增）](#13-gatewaypy--安全网关核心v21-新增)
- [14. server.py — HTTP API 服务（v2.1 新增）](#14-serverpy--http-api-服务v21-新增)
- [15. cli.py — 命令行入口](#15-clipy--命令行入口)
- [16. main.py — 批量检测主入口](#16-mainpy--批量检测主入口)
- [17. __init__.py — 公开 API 导出](#17-initpy--公开-api-导出)
- [18. ui/ — 前端工作台（React + Vite）](#18-ui--前端工作台react--vite)
- [19. 配置体系](#19-配置体系)
- [20. 测试体系](#20-测试体系)
- [21. 模块间数据流全景图](#21-模块间数据流全景图)
- [22. v2.2 变更记录](#22-v22-变更记录)

---

## 0. 项目概述与核心概念

### 0.1 项目定位

CFA-Score 是一个**输出侧防泄密网关**。它的核心问题是：

> 大模型在回答用户问题时，即使不直接输出受限字段，也可能通过组合多个公开信息锚点，让攻击者**间接还原**出受限事实。CFA-Score 自动检测并拦截这种风险。

### 0.2 组合事实还原（CFA）原理

```
               用户输入（模糊，缩小候选）        模型输出（补充细节，进一步压缩）
                     ↓                                  ↓
 事实池（多条记录） → 通过锚点逐步过滤 → 最终候选 ≤ k 条 → 受限字段被间接还原
```

**医疗场景示例**：
1. 事实池有 8 条患者记录，含患者姓名、科室、病房类型、诊断、用药、医保等
2. 用户输入："心内科 ICU 的那个心梗病人" → 抽取 `ward_type=ICU`（候选 8→5）、`department=心内科`（5→4）、`condition_summary` 含"心梗"（4→2） → 候选压缩到 2 条（P004、P005）
3. 模型输出："建议继续双抗，并结合 β 受体阻滞剂治疗" → "双抗"对应 `medication=替格瑞洛+阿司匹林+美托洛尔` → 候选压缩到 1 条（P004）
4. **最终**：P004 的 `diagnosis`（诊断结果）、`medication`（用药方案）、`insurance_level`（医保类型）被间接还原

### 0.3 三种检测模式

| 模式 | 抽取方式 | 安全改写 | 快速程度 | 召回率 |
|------|---------|---------|---------|--------|
| **模式1（Rule Only）** | 仅规则抽取（确定性） | 规则改写 | 最快 | 基准 |
| **模式2（Rule + LLM）** | 规则 + LLM 语义抽取 | 规则改写 | 较慢 | 更高 |
| **模式3（+ 二次检测）** | 模式2 + LLM 改写 + 再检测 | LLM改写+兜底 | 最慢 | 最高+兜底 |

### 0.4 四个预置场景

| 场景 ID | 领域 | 受限字段示例 | 事实池规模 |
|---------|------|-------------|-----------|
| `healthcare` | 医疗健康 | 诊断结果、用药方案、医保类型 | 8 条患者记录 |
| `finance` | 金融信贷 | 贷款金额、利率、信用评级、抵押物 | 10 条贷款记录 |
| `aerospace` | 航天测控 | 组件版本、风险状态、处置状态 | 5 条资产记录 |
| `meetings` | 会议管理 | 受限内容、涉密标识 | 4 条会议记录 |

### 0.5 技术栈

- **后端**：Python 3.10+，纯标准库（无第三方依赖）
- **前端**：React 18 + Vite 5，纯标准 fetch（无 axios）
- **LLM**：DeepSeek API（OpenAI 兼容接口）
- **测试**：unittest + pytest

---

## 1. models.py — 数据模型

**文件位置**：`cfa_score/models.py`  
**技术依赖**：`dataclasses`（Python 标准库）`hashlib`, `re`, `json`, `unicodedata`, `collections.Counter`, `math`  
**作用**：定义所有数据的结构，是全项目各模块通信的"共同语言"。

### 1.1 AssetFact — 受限事实池中的一条记录

```python
@dataclass(frozen=True)
class AssetFact:
    id: str                        # 唯一标识，如 "P001"、"G003"、"L001"、"M001"
    system_name: str               # 显示名称（通用字段名，实际含义由场景决定）
    business_domain: str           # 业务域/科室
    environment: str               # 环境/病房类型
    function_category: str         # 功能分类
    component_version: str         # 组件版本
    risk_status: str               # 风险状态
    disposition_status: str        # 处置状态
    remote_entry: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)
```

**`extra` 字段机制**：最初 8 个固定字段为航天测控场景设计。医疗/金融/会议场景的字段名完全不同（如 `patient_name`、`loan_amount`、`meeting_topic`），通过 `extra` 字典让同一个 `AssetFact` 适配任意场景——policy 中的 `field_order` 指向不在 8 个固定字段中的字段时，从 `extra` 取值。

**关键方法**：
| 方法 | 签名 | 行为 |
|------|------|------|
| `get(field_name)` | `(str) → str` | 优先从具名属性取值，其次从 `extra` 字典取值。空值返回 `""` |
| `display_name(display_field)` | `(str) → str` | 返回人类可读名称，依次尝试 `display_field` → `system_name` → `name` → `title` → `display_name` → `id` |
| `from_dict(data)` | `(dict) → AssetFact` | 从 JSON 字典反序列化。known 字段直接映射，其余放入 `extra` |

**注意事项**：
- `frozen=True`：AssetFact 不可变，保证事实池数据在运行期间不被修改
- `from_dict` 中 `system_name` 如果为空，会自动尝试从 `name`/`title`/`display_name`/`id` 回退
- 使用 `field(default_factory=dict)` 避免默认参数共享引用问题

### 1.2 SemanticFieldAlias — 语义别名信息

```python
@dataclass
class SemanticFieldAlias:
    components: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    partial_clues: List[str] = field(default_factory=list)
    possible_inferences: List[Dict[str, str]] = field(default_factory=list)
    partial_match_policy: str = "any_component_or_alias"
```

**各字段含义**：
- `components`：字段值的组成部分。如 `"替格瑞洛 90mg bid + 阿司匹林 100mg qd + 美托洛尔 25mg bid"` → `["替格瑞洛", "阿司匹林", "美托洛尔"]`
- `aliases`：同义表达。如 `"ICU"` → `["重症监护", "重症监护室", "ICU病房"]`
- `partial_clues`：部分线索。如 `"替格瑞洛联合阿司匹林双抗方案"` → `["双抗", "抗血小板", "联合抗血小板"]`
- `possible_inferences`：可能通过公开知识推断的信息。如 `[{"target_field": "risk_status", "target_value": "高危", "source": "CVE-2017-0144"}]`
- `partial_match_policy`：部分匹配策略，默认 `"any_component_or_alias"`

**数据来源**：离线由 LLM 生成，存储在 `config/healthcare_semantic_aliases.json` 中。结构为：

```json
{
  "semantic_aliases": {
    "medication": {
      "替格瑞洛 90mg bid + 阿司匹林 100mg qd + 美托洛尔 25mg bid": {
        "components": ["替格瑞洛", "阿司匹林", "美托洛尔"],
        "aliases": ["替格瑞洛", "替格瑞洛联合阿司匹林", "双抗"],
        "partial_clues": ["双抗", "抗血小板", ...],
        "possible_inferences": [],
        "partial_match_policy": "any_component_or_alias"
      }
    }
  }
}
```

### 1.3 CandidateValue — 候选召回结果

```python
@dataclass
class CandidateValue:
    field_name: str
    canonical_value: str
    score: float
    source: str = "hybrid_sparse"          # v2.2 新增
    matched_terms: List[str] = field(default_factory=list)    # v2.2 新增
    score_breakdown: Dict[str, float] = field(default_factory=dict)  # v2.2 新增
```

向后兼容：所有 v2.2 新增字段都有默认值，旧代码创建 `CandidateValue(field_name, canonical_value, score)` 仍然有效。

`score_breakdown` 示例：
```python
{"alias_raw": 6.5, "alias_norm": 0.867, "bm25_raw": 2.3, "bm25_norm": 0.697, "ngram": 0.45, "field_hint": 1.0}
```

### 1.4 FieldPolicy — 字段策略（检测规则配置）

```python
@dataclass(frozen=True)
class FieldPolicy:
    # ---- 核心字段分类 ----
    protected_fields: List[str]             # 受限字段
    identifier_fields: List[str]            # 强标识字段（v2.2 精简）
    quasi_identifier_fields: List[str]      # 准标识字段（v2.2 新增）
    field_order: List[str]                  # 字段处理优先级
    field_labels: Dict[str, str]            # 字段中文标签
    field_weights: Dict[str, float]         # 字段评分权重

    # ---- 匹配配置 ----
    field_aliases: Dict[str, Dict[str, List[str]]]   # 别名映射
    public_rules: List[Dict[str, Any]]               # 公开知识推理规则
    uniqueness_k: int = 1                            # 唯一还原阈值
    display_field: str = "system_name"               # 显示名称字段

    # ---- 安全改写配置 ----
    safe_replacements: Dict[str, str] = field(default_factory=dict)
    safe_hint: str = ""

    # ---- LLM 相关配置 ----
    semantic_aliases: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    match_type_weights: Dict[str, float] = field(default_factory=dict)
    llm_extraction_fields: List[str] = field(default_factory=list)
    llm_confidence_threshold: float = 0.4
    llm_max_accepted_values: int = 30
```

**三类字段分类（v2.2 核心设计）**：

| 分类 | 字段 | 作用 | 医疗场景示例 |
|------|------|------|-------------|
| `protected_fields` | 受限字段 | 被还原即构成风险 | `diagnosis`, `medication`, `insurance_level` |
| `identifier_fields` | 强标识 | 直接可定位记录 | `patient_name` |
| `quasi_identifier_fields` | 准标识 | 组合后可定位 | `department`, `ward_type`, `condition_summary` |

**关键方法**：
| 方法 | 签名 | 行为 |
|------|------|------|
| `label(field_name)` | `(str) → str` | 返回中文标签，找不到则返回原字段名 |
| `weight(field_name)` | `(str) → float` | 返回字段权重，默认 0.05 |
| `match_type_weight(mt)` | `(str) → float` | 返回匹配类型权重，优先 policy 配置，否则用 `DEFAULT_MATCH_TYPE_WEIGHTS` |
| `get_semantic_aliases(fn)` | `(str) → Dict[str, SemanticFieldAlias]` | 返回解析后的语义别名字典 |
| `from_dict(data)` | `(dict) → FieldPolicy` | 从 JSON 反序列化 |

### 1.5 Anchor — 从文本中抽取的一个事实线索

```python
@dataclass
class Anchor:
    # ---- 基础信息 ----
    id: str                          # 全局唯一 ID
    field_name: str                  # 对应事实池的字段名
    field_label: str                 # 中文标签
    text: str                        # 原文中的证据文本
    canonical_value: str             # 对应的事实池规范值
    start: int                       # 原文起始字符位置
    end: int                         # 原文结束字符位置
    anchor_type: str                 # 类型标签
    protected: bool                  # 是否为受限字段
    source: str = "output"           # "input" 或 "output"
    inferred: bool = False           # 推断锚点

    # ---- 推断锚点相关 ----
    evidence: str = ""
    source_anchor_id: Optional[str] = None

    # ---- v2.0 LLM 语义抽取字段 ----
    match_type: str = "exact"        # exact/alias/semantic/partial/inferred/ambiguous
    confidence: float = 1.0          # 0.0-1.0
    llm_reason: str = ""
    accepted_values: List[str] = field(default_factory=list)
```

**ID 生成规则**：
- 规则抽取器：`"A" + SHA256(field_name|value|start|end|inferred|source|source_anchor_id)[:10]`（确定性，可复现）
- LLM 抽取器：`"LLM" + 递增计数器的 4 位零填充`，如 `LLM0001`

**关键方法**：
| 方法 | 签名 | 行为 |
|------|------|------|
| `effective_canonical_value()` | `() → str` | 优先返回 `canonical_value`，否则返回单个/多个 `accepted_values` 的描述 |
| `match_symbol()` | `() → str` | exact/alias 返回 `"="`，其他返回 `"≈"` |
| `to_dict()` | `() → Dict[str, Any]` | 序列化为字典（含 `accepted_values`） |

### 1.6 ReductionStep — 还原链中的一步

```python
@dataclass
class ReductionStep:
    field_name: str
    field_label: str
    anchor_text: str
    canonical_value: str
    before_count: int
    after_count: int
    remaining_asset_ids: List[str]
    match_symbol: str = "="      # "=" 或 "≈"
```

示例输出：`病房类型=ICU: 8 → 5 (P002,P003,P004,P005,P007)`

### 1.7 RiskFinding — 一次风险发现

```python
@dataclass
class RiskFinding:
    target_asset_id: str
    target_asset_name: str
    risk_level: str              # CRITICAL/HIGH/MEDIUM/LOW
    score: float                 # 0-100
    reason: str
    restored_fact: str
    anchors: List[Anchor]
    reduction_chain: List[ReductionStep]
    minimal_combinations: List[List[str]]
    key_anchor_ids: List[str] = field(default_factory=list)
    key_anchor_summary: List[str] = field(default_factory=list)
```

风险等级判定：≥85→CRITICAL, ≥70→HIGH, ≥45→MEDIUM, <45→LOW

### 1.8 AnalysisResult — 完整检测结果

```python
@dataclass
class AnalysisResult:
    raw_answer: str
    anchors: List[Anchor]
    findings: List[RiskFinding]
    x_replaced_answer: str      # X 替换版
    safe_answer: str             # 安全泛化版
    user_input: str = ""
    model_output: str = ""
    secondary_check_performed: bool = False
    secondary_safe_answer: str = ""
    secondary_findings: List[RiskFinding] = field(default_factory=list)
```

`to_dict()` 方法将 `anchors`、`findings`、`reduction_chain` 递归序列化为 JSON 兼容格式。

### 1.9 GatewayResponse — 网关对外返回

```python
@dataclass
class GatewayResponse:
    request_id: str
    answer: str                  # ← 调用方唯一应该使用的文本
    raw_answer: str              # LLM 原始输出（仅前端 diff 展示用）
    risk_detected: bool
    risk_level: str
    score: float
    safe_answer_used: str        # 使用的答案策略标签
    findings_count: int
    findings_summary: List[Dict[str, Any]] = field(default_factory=list)
    intent: str = ""
    routed_scenario: str = ""
    answer_strategy: str = ""    # cfa_gated/general_answer/weather_answer/need_city_prompt
```

### 1.10 Intent — 意图分类结果

```python
@dataclass
class Intent:
    domain: str                  # domain_healthcare/domain_finance/.../general_chat
    confidence: float
    reason: str
    matched_keywords: List[str] = field(default_factory=list)
```

### 1.11 DEFAULT_MATCH_TYPE_WEIGHTS — 全局常量

```python
DEFAULT_MATCH_TYPE_WEIGHTS = {
    "exact": 1.00, "alias": 0.90, "semantic": 0.80,
    "partial": 0.65, "inferred": 0.70, "ambiguous": 0.45,
}
```

各 policy 可通过 `match_type_weights` 覆盖。

---

## 2. deepseek.py — DeepSeek API 客户端

**文件位置**：`cfa_score/deepseek.py`  
**技术依赖**：`urllib.request` + `json`（纯 Python 标准库，无需 pip install）

### 2.1 DeepSeekConfig

```python
@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    timeout_seconds: int = 60
```

### 2.2 关键函数详解

**`load_dotenv(path: str | Path = ".env") → None`**  
自实现的 `.env` 加载器。逐行读取 `KEY=VALUE` 格式，跳过注释和空行，不覆盖已存在的环境变量。

**`config_from_env(env_path) → DeepSeekConfig`**  
1. 调用 `load_dotenv(env_path)`  
2. 读取 `DEEPSEEK_API_KEY`（必须设置，否则抛出 `RuntimeError`）  
3. 读取可选 `DEEPSEEK_BASE_URL`（默认 `https://api.deepseek.com`）  
4. 读取可选 `DEEPSEEK_MODEL`（默认 `deepseek-chat`）

**`DeepSeekClient`** 类：
- `chat(messages, temperature=0.2, max_tokens=512) → str`  
  1. 构造 JSON payload：`{"model": ..., "messages": ..., "temperature": ..., "max_tokens": ...}`  
  2. POST 到 `{base_url}/chat/completions`，Bearer 认证  
  3. 解析 `data["choices"][0]["message"]["content"]` 并返回  
  4. HTTP 错误（如 401/429/500）→ `RuntimeError`  
  5. URL 错误（网络不通）→ `RuntimeError`

### 2.3 错误处理

| 异常类型 | 触发条件 | 错误信息格式 |
|----------|---------|-------------|
| `RuntimeError` | API Key 未设置 | `"DEEPSEEK_API_KEY is not set..."` |
| `RuntimeError` | HTTP 非 200 | `"DeepSeek request failed with HTTP {code}: {detail}"` |
| `RuntimeError` | URL 不可达 | `"DeepSeek request failed: {reason}"` |
| `RuntimeError` | 响应 JSON 结构异常 | `"Unexpected DeepSeek response shape: {body}"` |

---

## 3. knowledge.py — 知识加载工具函数

**文件位置**：`cfa_score/knowledge.py`  
**技术依赖**：`json` + `pathlib` + `dataclasses.replace`

| 函数 | 签名 | 功能 | 异常 |
|------|------|------|------|
| `load_assets(path)` | `(str\|Path) → List[AssetFact]` | 加载事实池 JSON 列表 | 非列表 → `ValueError` |
| `load_policy(path)` | `(str\|Path) → FieldPolicy` | 加载字段策略 JSON 对象 | 非对象 → `ValueError` |
| `load_public_knowledge(path)` | `(str\|Path) → List[dict]` | 加载公开知识规则，支持列表和 `{"rules": [...]}` 两种格式 | 格式不符 → `ValueError` |
| `load_semantic_aliases(path)` | `(str\|Path) → dict` | 加载语义别名 JSON，自动提取 `semantic_aliases` 键 | 非对象 → `ValueError` |
| `merge_public_knowledge(policy, rules)` | `(FieldPolicy, List[dict]) → FieldPolicy` | 将外部规则合并到 FieldPolicy（使用 `dataclasses.replace`） | 无，空规则直接返回原 policy |
| `dump_json(data, path)` | `(Any, str\|Path) → None` | 写入 JSON 文件（ensure_ascii=False, indent=2） | IO 异常透传 |

**编码处理**：所有 `read_text` 使用 `encoding="utf-8-sig"`，兼容带 BOM 的 UTF-8 文件。

---

## 4. adapter.py — LLM 适配器接口

**文件位置**：`cfa_score/adapter.py`  
**技术依赖**：`abc` + `pathlib` + `deepseek.py` + `models.py`

### 4.1 LLMAdapter（抽象基类）

```python
class LLMAdapter(ABC):
    @abstractmethod
    def generate(self, user_input: str, context: dict[str, Any] | None = None) -> str:
        ...
```

### 4.2 DeepSeekAdapter

构造函数接受 4 个参数：
- `config: DeepSeekConfig | None` — 直接传入配置对象
- `env_path: str | Path = ".env"` — .env 文件路径
- `system_prompt: str | None` — 自定义系统提示词
- `model: str | None` — 覆盖模型名称

**`generate(user_input, context) → str`**：
1. 从 context 中提取 `fact_pool`、`public_knowledge`、`policy`
2. 构造 user message：`【用户问题】` + `【公开漏洞情报】` + `【内部资产台账】` + `"请基于以上信息回答用户的问题。"`
3. 调用 `self._client.chat(messages)`

**设计意图**：将**整个事实池直接注入 prompt**（Prompt Stuffing）——不是 RAG。论文实验阶段故意这么做，确保 LLM 充分看到事实池以暴露 CFA 风险。

### 4.3 格式化辅助函数

**`_format_public_knowledge(rules) → str`**：格式化规则为 "当 {field} 为 {values} 时，{implies}" 格式。

**`_format_fact_pool(assets, policy) → str`**：按 `field_order` 格式化每条事实，格式为 "  [ID] display_name，field_label=value，..."。

**`generate_answer(user_input, public_knowledge_rules, fact_pool, env_path, policy) → str`**：遗留兼容函数，内部创建 `DeepSeekAdapter` 并调用 `generate()`。

---

## 5. extractor.py — 规则锚点抽取器

**文件位置**：`cfa_score/extractor.py`  
**技术依赖**：`hashlib` + `re`  
**作用**：模式 1 的核心，模式 2/3 的基础层。从 user_input 和 model_output 中精确抽取与事实池字段值匹配的文本。

### 5.1 RuleBasedAnchorExtractor

**两个入口**：
- `extract(raw_answer, assets, source="output") → List[Anchor]` — 单文本抽取
- `extract_segments(segments, assets) → List[Anchor]` — 多文本抽取（engine.py 实际调用）

`segments` 格式：`[("input", user_input), ("output", model_output)]`

### 5.2 抽取三步流程

**第一步：精确值匹配**
1. 遍历 `field_order` 中的每个字段
2. 收集事实池中该字段的所有值，**按长度降序排列**（长值先匹配，防止短值"吃掉"长值的一部分）
3. 在文本中查找精确匹配位置（使用词边界感知正则 `_literal_pattern`）
4. 去重 key：`(source, field_name, value, start, end)`

**第二步：别名匹配**
1. 遍历 `field_aliases`，将别名短语映射到规范值
2. 对每个 canonical_value + aliases，按长度降序在文本中查找
3. `canonical_value` 始终是规范值（如 `"ICU"`），`text` 是原文证据（如 `"重症监护室"`）

**第三步：公开规则推断**
1. 对已抽取的锚点应用 `public_rules` 推理
2. 生成 `inferred=True` 的推断锚点，记录 `source_anchor_id` 和 `evidence`
3. 去重 key：`(source, target_field, target_value, start, end, True)`

### 5.3 辅助方法

| 方法 | 功能 |
|------|------|
| `_make_anchor(field_name, text, canonical_value, start, end, inferred, ...)` | 统一的 Anchor 构造器，自动计算 `protected` 和 `anchor_type` |
| `_stable_anchor_id(field_name, value, start, end, inferred, source, source_anchor_id)` | 确定性 ID：`"A" + SHA256(参数拼接)[:10]` |
| `_anchor_type(field_name)` | 根据字段名返回中文类型标签 |
| `_literal_pattern(needle)` | 词边界感知正则。`"xz-utils 5.6.1"` 不匹配 `"xz-utils 5.6.10"` |
| `_find_all(text, needle)` | 返回所有匹配位置的 `[(start, end, matched_text), ...]` |
| `_deduplicate_overlaps(anchors)` | 去重完全相同的锚点，保留嵌套锚点（如"航天测控生产区"和"生产区"） |
| `_infer_from_public_rules(anchors)` | 应用 `public_rules` 推理生成推断锚点 |

---

## 6. retriever.py — 混合稀疏召回器（v2.2 新增）

**文件位置**：`cfa_score/retriever.py`  
**技术依赖**：`math` + `re` + `unicodedata` + `collections.Counter`（纯 Python 标准库）  
**作用**：在原有 alias 召回基础上，增加 BM25 + 中文 char n-gram + Field Hint 四路融合召回。

### 6.1 设计动机

原有召回仅依赖人工编写的 `semantic_aliases` 做字符串包含匹配，存在盲区：
- 无法捕捉 token 级别的部分重合（"做了支架的病人" vs "急性心梗介入术后"）
- 口语表达和合成词无法匹配到别名
- 过多依赖人工别名维护

### 6.2 核心函数

**`normalize_text(text: str) → str`**
1. `unicodedata.normalize("NFKC", text)` — 全角→半角归一化
2. `.lower()` — 英文小写
3. `β` → `"beta"`，`＋` → `"+"`
4. 多个空白合并为一个空格
5. **保留版本号**（`5.6.1`、`CVE-2017-0144`、`xz-utils` 不会被破坏）
6. **不删除中文字符**

**`word_tokens(text: str) → List[str]`**
- 英文单词：`[a-zA-Z]+`
- 数字序列：`\d+(?:\.\d+)*`（如 `5.6.1`）
- 技术 token：`[a-zA-Z0-9_.-]+`（含 `xz-utils`、`CVE-2017-0144`）
- 中文连续片段 → 2-gram + 3-gram
- 单字符跳过

示例：`"建议继续双抗治疗，版本为 xz-utils 5.6.1"`
→ `["xz", "utils", "xz-utils", "5.6.1", "建议", "议继", "继续", "续双", "双抗", "抗治", "治疗", ...]`

**`char_ngrams(text: str, ns=(2,3)) → List[str]`** — 仅对中文连续字符生成 n-gram

### 6.3 CandidateDocument

```python
@dataclass
class CandidateDocument:
    field_name: str
    canonical_value: str
    text: str              # field_name + label + value + aliases + components + clues 拼接
    tokens: List[str]      # word_tokens 结果
    token_counts: Counter  # 词频统计
    length: int            # token 数量
    source_terms: Dict[str, List[str]]  # 各来源词条（调试用）
```

### 6.4 HybridSparseRetriever

**构造函数**：
1. 从事实池和 `alias_lookup` 构建所有 `CandidateDocument`（按 `(field_name, value)` 去重）
2. 统计每个 token 的 document frequency（`_doc_freqs`）
3. 计算平均文档长度 `avgdl`
4. 预计算 IDF 缓存

### 6.5 四路融合评分

**1. Alias 分数（`_alias_score`）**：
| 匹配类型 | 加分 |
|----------|------|
| `canonical_value` 直接出现在文本中 | +3.5 |
| `alias` 命中 | +3.0 |
| `component` 命中 | +2.0 |
| `partial_clue` 命中 | +1.0 |

**2. BM25 分数（`_bm25_score`）**：
```
BM25(q, d) = Σ_{t∈q} IDF(t) × TF(t,d) × (k1+1) / (TF(t,d) + k1×(1-b+b×len(d)/avgdl))
```
参数：`k1=1.2`, `b=0.75`  
IDF：`log(1 + (N-df+0.5)/(df+0.5))`，未见 token 默认 0.1

**3. 中文 n-gram 分数（`_ngram_score`）**：
```
ngram_score = Σ_{t∈(q∩d)} IDF(t) / Σ_{t∈q} IDF(t)
```
范围 0-1。query 和 doc token 集合的交集 IDF 占比。

**4. Field Hint 分数（`_field_hint_score`）**：
内置 `FIELD_HINTS` 字典，如 `"medication" → ["用药", "治疗", "处方", "双抗", ...]`。命中 1 个 0.5，≥2 个 1.0。

**最终融合**：
```python
final = 0.45 × squash(alias_raw) + 0.30 × squash(bm25_raw) + 0.20 × ngram_val + 0.05 × hint_val
squash(score) = score / (score + 1.0)   # [0, +∞) → [0, 1]
```

### 6.6 候选截断策略

```python
def retrieve(text, top_k=40, max_per_field=8, min_score=0.08):
```
1. 所有文档评分，过滤 < `min_score`
2. 按 final_score 降序排序
3. **每字段最多保留 `max_per_field=8` 个**（防止单一字段占满全局 top-k）
4. 合并所有字段候选
5. 全局再取 `top_k`

**为什么做 per-field 截断**：CFA 检测依赖多字段组合还原链。如果一个字段候选过多，会挤掉其他字段，导致组合还原链失效。

### 6.7 调试能力

`self.last_stats` 保存最近一次检索统计（不打印到控制台）：
```python
{"query_length": 45, "candidate_count": 32, "empty_candidate": False,
 "top_fields": ["medication", "ward_type", "department"],
 "max_score": 0.87, "avg_score": 0.34, "source": "hybrid_sparse"}
```

---

## 7. semantic_index.py — 语义别名索引与候选召回

**文件位置**：`cfa_score/semantic_index.py`  
**技术依赖**：`retriever.HybridSparseRetriever`

### 7.1 SemanticIndex

构造时三步初始化：
1. 构建 `_alias_lookup`：`(field_name, canonical_value) → SemanticFieldAlias` 字典
2. 构建 `_valid_values`：`field_name → set(所有可能的值)` 用于校验
3. **初始化 `_hybrid_retriever = HybridSparseRetriever(policy, assets, alias_lookup)`**（v2.2）

### 7.2 公开方法

| 方法 | 签名 | 功能 | 调用场景 |
|------|------|------|---------|
| `retrieve_candidates` | `(text, top_k=30) → List[CandidateValue]` | 委托给 `_hybrid_retriever.retrieve()` | `LLMSemanticAnchorExtractor.extract()` 候选召回 |
| `retrieve_candidates_for_field` | `(text, field_name, top_k=15) → List[CandidateValue]` | 单字段过滤 | 按需 |
| `build_candidate_text` | `(candidates, max_per_field=10) → Dict[str, List[str]]` | 分组为 `{field_name: [value1, ...]}` | LLM prompt 构建 |
| `get_valid_values` | `(field_name) → set` | 某字段在事实池中的所有值 | `AnchorVerifier` 校验 |
| `get_aliases_for_value` | `(field_name, canonical_value) → SemanticFieldAlias` | 查别名 | 按需 |
| `contains_value` | `(field_name, value) → bool` | 值是否存在 | 校验 |

**向后兼容**：所有公开接口签名未变。`CandidateValue` 扩展了 3 个有默认值的新字段，不影响现有代码。

---

## 8. llm_extractor.py — LLM 语义锚点抽取器

**文件位置**：`cfa_score/llm_extractor.py`  
**技术依赖**：`json` + `re` + `deepseek.DeepSeekClient`  
**作用**：调用 LLM 从 user_input 和 model_output 中抽取规则抽取器漏掉的同义表达、概括、暗示类锚点。

### 8.1 LLMSemanticAnchorExtractor

```python
class LLMSemanticAnchorExtractor:
    def __init__(self, client: DeepSeekClient, policy: FieldPolicy, semantic_index: SemanticIndex):
        self._client = client
        self._policy = policy
        self._index = semantic_index
        self._anchor_counter = 0
```

### 8.2 Prompt 模板详解

**System Prompt** 关键设计点：
- "你的任务不是回答用户问题，也不是判断是否泄密"（防止 LLM 做风险判断，只做结构化抽取）
- "如果表达是模糊的，必须输出 accepted_values，而不是强行选择一个 canonical_value"（关键安全设计：宁可多候选也不误判）
- "不允许编造事实池中不存在的字段值"（防幻觉）
- "只输出 JSON，不要输出解释性文字，不要用 markdown 代码块包裹"（便于解析）

**User Prompt** 包含四个部分：
1. 字段策略说明（字段列表 + 受限/标识标签）
2. SemanticIndex 召回的候选值（`build_candidate_text` 格式化）
3. 用户输入原文
4. 模型输出原文

### 8.3 抽取流程

```
Step 1: combined_text = user_input + " " + model_output
Step 2: candidates = _index.retrieve_candidates(combined_text, top_k=max_candidates)
Step 3: 如 candidates 为空 → 直接返回 []（无需调用 LLM，节省 API 费用）
Step 4: 构建 prompt → 调用 LLM（temperature=0.1, max_tokens=1500）
Step 5: 三层容错 JSON 解析
        ├─ json.loads(raw) 直接解析
        ├─ 正则提取 ```json ... ``` 块
        └─ 正则提取第一个 { ... } 对象
Step 6: _convert_to_anchors() 转换并做第一层校验
```

### 8.4 Anchor 转换逻辑

`_convert_to_anchors()` 对 LLM 每一条输出做第一层校验：
1. `field_name` 必须在 `policy.field_order` 中 → 否则丢弃
2. `source` 必须是 `"input"` 或 `"output"` → 否则默认 `"output"`
3. `confidence` ≥ `policy.llm_confidence_threshold`（默认 0.4）→ 否则丢弃
4. `accepted_values` 过滤到事实池中真实存在的值
5. `canonical_value` 无效（不在事实池）→ 移入 `accepted_values` 或丢弃
6. 至少有一个有效映射 → 否则丢弃
7. `accepted_values` 数量 > `llm_max_accepted_values`（默认 30）→ 截断

### 8.5 容错解析

`_parse_json_response(raw) → Optional[Dict[str, Any]]` 三层回退：
1. `json.loads(raw)` — 直接解析
2. 正则 `re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)` — 提取 Markdown 代码块
3. 正则 `re.search(r'\{[\s\S]*\}', raw)` — 提取任意 JSON 对象

---

## 9. anchor_verifier.py — 锚点校验器

**文件位置**：`cfa_score/anchor_verifier.py`  
**技术依赖**：`semantic_index.SemanticIndex`

### 9.1 7 条校验规则

| 规则 | 检查内容 | 处理 |
|------|---------|------|
| 1 | `field_name` 在 `policy.field_order` 中 | 丢弃 |
| 2 | `source_text` 在对应 source 文本中可找到（推断锚点豁免此检查） | 丢弃（先精确匹配，再大小写不敏感匹配） |
| 3 | `canonical_value` 和 `accepted_values` 在事实池中真实存在 | 修正：无效 canonical 移入 accepted，过滤后的 accepted 全无效则丢弃 |
| 4 | 至少有一个有效映射（canonical 或 accepted） | 丢弃 |
| 5 | `confidence` ≥ 阈值 | 丢弃 |
| 6 | `accepted_values` 数量 ≤ 上限 | 超限 → 降级 match_type 为 `"ambiguous"` |
| 7 | `protected` 标记以 policy 为准 | 覆盖（不信 LLM 输出） |

### 9.2 校验通过后的输出

返回一个新的 Anchor 对象，其中 `accepted_values`、`canonical_value`、`match_type`、`source_text`、`protected` 均已纠错。

### 9.3 批量校验

`verify_all(llm_anchors, user_input, model_output) → List[Anchor]` 对每个 LLM anchor 调用 `verify()`，仅返回通过校验的。

---

## 10. intent_router.py — 意图路由器（v2.1 新增）

**文件位置**：`cfa_score/intent_router.py`  
**技术依赖**：`re` + `dataclasses`

### 10.1 意图域

```python
DOMAIN_HEALTHCARE  = "domain_healthcare"
DOMAIN_FINANCE     = "domain_finance"
DOMAIN_AEROSPACE   = "domain_aerospace"
DOMAIN_MEETINGS    = "domain_meetings"
GENERAL_WEATHER    = "general_weather"
GENERAL_CHAT       = "general_chat"
AMBIGUOUS          = "ambiguous"
```

### 10.2 分类机制

`classify_intent(text) → Intent`：

1. 遍历 6 组关键词模式（医疗/金融/航天/会议/天气/闲聊）
2. 每组内使用乘法组合器：`domain_score = 1 - Π(1 - weight_i)`
   - 单个强匹配（weight=0.95）→ score=0.95
   - 两个中等匹配（0.8 + 0.8）→ score=1-0.2×0.2=0.96
3. 天气+时间关键词组合额外 boost 1.5 倍
4. 最高分 ≥ 0.55 → 归类到对应领域
5. 否则 → `GENERAL_CHAT`（confidence=0.3）

### 10.3 辅助函数

| 函数 | 签名 | 功能 |
|------|------|------|
| `is_domain_intent(intent)` | `(Intent) → bool` | 是否为业务领域内意图 |
| `map_intent_to_scenario(intent)` | `(Intent) → str` | 映射到场景 ID |
| `get_general_system_prompt(intent)` | `(Intent) → str` | 返回通用模式的 system prompt |

### 10.4 关键词模式（共约 70 个）

医疗场景 20 个模式（含 ICU、抗凝、替格瑞洛等），金融场景 13 个（含 LPR、抵押、征信等），航天场景 9 个（含 CVE、XZ、生产区等），会议场景 6 个，天气场景 5 个，闲聊场景 5 个。

---

## 11. engine.py — CFA-Score 核心引擎

**文件位置**：`cfa_score/engine.py`  
**技术依赖**：`itertools` + 前 10 个模块  
**作用**：编排检测流程、计算 CFA-Score、生成改写。整个项目的**大脑**。

### 11.1 ExtractionMode

```python
class ExtractionMode:
    RULE_ONLY = "rule_only"          # 模式1
    RULE_PLUS_LLM = "rule_plus_llm"  # 模式2/3（区别在 do_secondary_check 参数）
```

### 11.2 AnchorMerger

```python
class AnchorMerger:
    @staticmethod
    def merge(rule_anchors, llm_anchors) -> List[Anchor]:
```

**合并策略**：
1. 规则锚点优先加入（精确字符串匹配，确定性高）
2. LLM 锚点补充：跳过已在规则锚点中的 `(field_name, canonical_value)` 对
3. 最终排序：source（input 优先）→ start → end → field_name → inferred

### 11.3 CFAScoreEngine

**构造函数**：
```python
def __init__(self, assets, policy, *, mode=RULE_ONLY, deepseek_client=None):
```
始终创建 `rule_extractor`、`sanitizer`、`semantic_index`。仅 mode 为 `RULE_PLUS_LLM` 且 `deepseek_client` 非空时才创建 LLM 组件。

**模式切换**：
- `enable_mode_2(deepseek_client)` — 升级到模式2
- `enable_mode_3(deepseek_client)` — 升级到模式3（`_llm_rewriter_client` 已设置）

### 11.4 analyze() 主流程

```
Step 1. segments = [("input", user_input), ("output", model_output)]（如有 user_input）
Step 2. rule_anchors = rule_extractor.extract_segments(segments, assets)
Step 3. [Mode 2/3] llm_anchors = _llm_extractor.extract() → _verifier.verify_all()
Step 4. all_anchors = AnchorMerger.merge(rule_anchors, llm_anchors)
Step 5. findings = _build_findings(all_anchors)
Step 6. x_replaced = sanitizer.make_x_replaced(model_output, findings)
Step 7. safe = sanitizer.make_safe_answer(model_output, findings)
Step 8. [Mode 3] secondary: LLM 改写 → 再检测 → 残留风险则用兜底回答
Step 9. 返回 AnalysisResult
```

### 11.5 _build_findings() 子流程

对每个事实（asset）执行：

1. **筛选匹配锚点**（`_anchors_matching_asset`）：选出所有字段值匹配该事实的锚点
2. **检查"敏感还原形态"**（`_has_sensitive_restoration_shape`）：必须同时满足"有定位器"+"有输出侧受限锚点"
3. **计算还原链**（`_reduction_chain`）：逐步过滤候选，每步记录 before/after 数量
4. **唯一性判断**：`final_count > policy.uniqueness_k`（默认 k=1）→ 不算风险（候选不唯一）
5. **计算评分**（`_score`）
6. **LOW 等级跳过**（不纳入最终报告）
7. **寻找最小锚点组合**（`_minimal_unique_anchor_sets`）
8. **构造 RiskFinding**

### 11.6 _has_sensitive_restoration_shape()（v2.2 重写）

**原有逻辑**：
```python
has_identifier = fields ∩ identifier_fields      # 仅允许 identifier_fields
has_output_protected = 至少1个输出侧非推断 protected 锚点
return has_identifier AND has_output_protected
```

**v2.2 新逻辑**：
```python
all_locator_fields = identifier_fields ∪ quasi_identifier_fields
has_locator = fields ∩ all_locator_fields
has_output_protected = 至少1个输出侧非推断 protected 锚点
return has_locator AND has_output_protected
```

**修复原因**：用户输入"心内科 ICU 的那个心梗病人"——没有患者姓名（强标识），但 `department=心内科` + `ward_type=ICU` + `condition_summary` 含"心梗"这三个准标识组合，已经将候选从 8 条压缩到 2 条。旧逻辑因为缺少 identifier 直接跳过，形成 **false negative**。

### 11.7 _score() 评分函数

```python
score = 100 × min(1.0,
    0.38 × uniqueness_score        # 唯一性：final_count=1→1.0
  + 0.34 × min(1.0, protected_weight)
  + 0.16 × min(1.0, identifier_weight)  # v2.2: 含 quasi_identifier_fields
  + inferred_bonus (0.05)          # 有推断锚点
  + version_bonus (0.08)          # 有版本号锚点
  + status_bonus (0.08)           # 有处置状态锚点
)
```

**effective_weight** = `field_weight × match_type_weight × confidence`

**等级判定**：≥85→CRITICAL, ≥70→HIGH, ≥45→MEDIUM, <45→LOW

### 11.8 _minimal_unique_anchor_sets() 最小锚点组合

穷举所有 r 个锚点组合（r = 2..5），找到能将候选唯一压缩到 1 条的最小组合。使用键集合子集剪枝优化（防止重复组合）。

### 11.9 _llm_safe_rewrite() LLM 安全改写

仅模式 3 调用：
1. 收集所有 MEDIUM/HIGH/CRITICAL 输出锚点
2. 构造改写 prompt："请改写下面的模型回答，使其不再暴露或暗示以下风险锚点..."
3. LLM 改写（temperature=0.3, max_tokens=512）
4. 对改写结果重新跑 CFA（_build_findings）
5. 有残留风险 → 兜底回答 `_FALLBACK_SAFE_ANSWER`
6. 无残留 → 返回改写文本

### 11.10 _FALLBACK_SAFE_ANSWER

```python
"该问题涉及可能受限的内部信息。请通过授权系统查询，或联系具备相应权限的人员处理。"
```

---

## 12. sanitizer.py — 安全改写器

**文件位置**：`cfa_score/sanitizer.py`  
**技术依赖**：`re`

### 12.1 AnswerSanitizer

```python
class AnswerSanitizer:
    def __init__(self, policy: Optional[FieldPolicy] = None):
```
接受可选 policy。若有 policy，使用其中的 `safe_replacements` 和 `safe_hint`；否则使用内置默认值。

### 12.2 两个输出版本

**`make_x_replaced(raw_answer, findings) → str`**：
- 敏感锚点文本 → `"X"`（极端脱敏，只保留句子结构）
- 航天场景额外硬编码已知改写规则
- 末尾追加授权查询提示

**`make_safe_answer(raw_answer, findings) → str`**：
- 敏感锚点文本 → policy.safe_replacements 中的泛化词（如"双抗"→"相关治疗方案"、"AA+"→"相关评级"）
- 无 findings 时仍然调用 `_replace_asset_ids()` 替换裸 ID
- 末尾追加 `safe_hint`

### 12.3 _dangerous_output_anchors() 危险锚点筛选

筛选条件：
1. 仅 MEDIUM/HIGH/CRITICAL 等级
2. 仅 `source="output"`（不改写用户输入，只改写输出）
3. 仅非推断锚点
4. 替换范围：`protected_fields` + `identifier_fields` + `display_field`
5. 排除已泛化的安全词（"受影响版本"、"处置未完成"等）
6. 按文本长度降序排列（长文本先替换，防止短文本破坏长文本中的部分）

### 12.4 _replace_asset_ids()

```python
pattern = re.compile(r'(?<![A-Za-z0-9])([A-Z]{1,3}\d{2,4})(?![A-Za-z0-9])')
# P004 → [P类编号], G003 → [G类编号], L001 → [L类编号]
```
无论有没有 findings 都会被调用，确保即使 CFA 未检测到风险也不泄漏事实 ID。

---

## 13. gateway.py — 安全网关核心（v2.1 新增）

**文件位置**：`cfa_score/gateway.py`  
**技术依赖**：`uuid` + 前 12 个模块  

### 13.1 CFAGateway

```python
class CFAGateway:
    def __init__(self, env_path=".env", base_dir=None):
```

初始化：`_scenario_cache`（缓存已加载的场景）、`_system_prompts`（4 个场景的 LLM 提示词）。

### 13.2 handle_chat() 完整链路

```
Step 0: 意图分类（scenario="auto"/"general" 时触发）
        ├─ 领域 → 映射场景 ID
        └─ 非领域 → _handle_general_chat()，跳过 CFA 管线
Step 1: _load_scenario(scenario) → (assets, policy)（缓存后免重复读文件）
Step 2: _call_llm_with_prompt() → raw_answer（注入领域边界约束）
Step 3: _run_cfa() → AnalysisResult
Step 4: _select_final_answer() → 最终安全回答
Step 5: _build_response() → GatewayResponse
```

### 13.3 handle_analyze() 仅检测链路

跳过 Step 2（不调用 LLM），直接对传入的 model_output 进行 CFA 检测。

### 13.4 _select_final_answer() 最终回答选择

| 条件 | answer | safe_answer_used 标签 |
|------|--------|----------------------|
| 二次检测 + 有残留风险 | 兜底回答 | `"fallback"` |
| 二次检测 + 无残留 | LLM 改写 | `"secondary_safe_answer"` |
| 无二次检测 + 有风险 | 规则改写 | `"cfa_safe_answer"` |
| 无二次检测 + 无风险 | 原样返回（raw_answer） | `"raw_answer"` |

### 13.5 _load_scenario() 场景加载与缓存

```python
self._scenario_cache: Dict[str, dict] = {}
```
首次加载读 JSON 文件并缓存到 `_scenario_cache`，后续直接命中缓存。

### 13.6 系统提示词

每个场景预设了**故意不安全的**系统提示词——告诉 LLM "可以提及具体诊断、用药方案、金额、评级等细节"。

**领域边界约束**（防止 LLM 越界回答）：
```
【重要边界约束】
你只能回答与当前业务领域直接相关的问题。如果用户的问题
与你的业务领域无关，请以通用AI助手的身份回答...
```

---

## 14. server.py — HTTP API 服务（v2.1 新增）

**文件位置**：`cfa_score/server.py`  
**技术依赖**：`http.server` + `json` + `argparse`（Python 标准库 HTTP server）

### 14.1 四个端点

| 方法 | 路径 | 请求体 | 响应 | 说明 |
|------|------|--------|------|------|
| POST | `/api/cfa-chat` | `{"user_input": "...", "scenario": "...", "mode": "...", "secondary_check": bool}` | `GatewayResponse` JSON | 完整链路 |
| POST | `/api/cfa-analyze` | `{"user_input": "...", "model_output": "...", "scenario": "...", "mode": "...", "secondary_check": bool}` | `GatewayResponse` JSON | 仅检测 |
| GET | `/api/health` | 无 | `{"status": "ok", "service": "cfa-gateway"}` | 健康检查 |
| GET | `/api/scenarios` | 无 | `{"scenarios": [{id, label}, ...]}` | 场景列表 |

### 14.2 CFAServer

**两种启动方式**：
- `serve()` — 阻塞模式，`Ctrl+C` 时 `shutdown()` 优雅退出
- `serve_non_blocking()` — 守护线程中启动，返回 `HTTPServer` 对象

**启动命令**：
```bash
python -m cfa_score.server                          # 0.0.0.0:8080
python -m cfa_score.server --port 9000              # 自定义端口
python -m cfa_score.server --host 127.0.0.1         # 仅本地
```

### 14.3 错误响应格式

| HTTP 状态码 | 触发条件 | 响应体 |
|-------------|---------|--------|
| 400 | 缺少必填字段 | `{"error": "Missing required field: user_input"}` |
| 400 | JSON 解析失败 | `{"error": "Invalid JSON: ..."}` |
| 400 | 场景不存在 | `{"error": "Unknown scenario: 'xxx'..."}` |
| 500 | 内部异常 | `{"error": "CFA gateway error: ..."}` |
| 404 | 路径不存在 | `{"error": "Not found"}` |

### 14.4 响应头

- `Content-Type: application/json; charset=utf-8`
- `Content-Length: ...`
- `X-Request-ID: {request_id}`

---

## 15. cli.py — 命令行入口

**文件位置**：`cfa_score/cli.py`  
**技术依赖**：`argparse` + `json` + `sys`

### 15.1 命令

```bash
cfa-score analyze \
  --facts config/healthcare_assets.json \
  --policy config/healthcare_policy.json \
  --input examples/healthcare_user_input.txt \
  --public-knowledge config/healthcare_public_knowledge.json \
  --env .env \
  --print summary
```

### 15.2 参数

| 参数 | 必需 | 说明 |
|------|------|------|
| `--facts` | ✓ | 事实池 JSON 路径 |
| `--policy` | ✓ | 字段策略 JSON 路径 |
| `--input` | | 用户输入文本文件路径 |
| `--input-text` | | 用户输入文本（直接命令行传入） |
| `--model-output` | | 已存在的模型回答文件路径 |
| `--model-output-text` | | 已存在的模型回答文本 |
| `--public-knowledge` | | 公开知识规则 JSON |
| `--env` | | .env 文件路径（默认 `.env`） |
| `--out` | | 输出 JSON 报告路径 |
| `--print` | | 输出类型：`summary`/`json`/`safe`/`x` |

**模型回答来源优先级**：直接文本 > 文件读取 > DeepSeek 在线生成

---

## 16. main.py — 批量检测主入口

**文件位置**：`main.py`（项目根目录）  
**作用**：早期开发的批量检测脚本，通过修改脚本顶部的全局变量切换场景和模式。

### 16.1 全局配置变量

| 变量 | 可选值 | 说明 |
|------|--------|------|
| `SCENARIO` | `"aerospace"`/`"healthcare"`/`"finance"`/`"meetings"`/`"custom"` | 场景切换 |
| `EXTRACTION_MODE` | `ExtractionMode.RULE_ONLY`/`RULE_PLUS_LLM` | 抽取模式 |
| `DO_SECONDARY_CHECK` | `True`/`False` | 模式3开关 |
| `MODEL_OUTPUT_TEXT` / `MODEL_OUTPUT_PATH` | 字符串 | 已有模型回答 |
| `USE_DEEPSEEK` | `True`/`False` | 是否在线生成（以上两个为空时生效） |
| `REPORT_PATH` | `Path` | 报告输出路径（默认 `report.json`） |

### 16.2 自定义场景

当 `SCENARIO = "custom"` 时，可配置：
- `CUSTOM_FACTS_PATH`、`CUSTOM_POLICY_PATH`、`CUSTOM_PUBLIC_KNOWLEDGE_PATH`
- `CUSTOM_SEMANTIC_ALIASES_PATH`、`CUSTOM_USER_INPUT_FILE`、`CUSTOM_SYSTEM_PROMPT`

---

## 17. __init__.py — 公开 API 导出

```python
from .adapter import DeepSeekAdapter, LLMAdapter, generate_answer
from .anchor_verifier import AnchorVerifier
from .engine import CFAScoreEngine, ExtractionMode
from .gateway import CFAGateway, GatewayResponse
from .knowledge import load_assets, load_policy, load_public_knowledge, load_semantic_aliases, merge_public_knowledge
from .llm_extractor import LLMSemanticAnchorExtractor
from .semantic_index import SemanticIndex
```

导出 13 个符号，分为 4 组：引擎、网关、知识加载、LLM 辅助。

---

## 18. ui/ — 前端工作台（React + Vite）

### 18.1 技术栈

- **框架**：React 18.2（函数组件 + Hooks）
- **构建**：Vite 5.4
- **依赖**：仅 `react` + `react-dom`（无其他第三方库）
- **通信**：原生 `fetch`（无 axios）
- **代理**：Vite dev server proxy `/api` → `http://127.0.0.1:8080`

### 18.2 文件结构

```
ui/
├── index.html           # 入口 HTML（中文 title）
├── package.json         # npm 配置
├── vite.config.js       # Vite 配置（端口 3000，API 代理到 8080）
└── src/
    ├── main.jsx         # React 根节点挂载
    ├── App.jsx          # 主组件（约 450 行）
    ├── App.css          # 全局样式
    ├── api.js           # API 客户端（cfaChat, cfaAnalyze）
    ├── DiffView.jsx     # 文本对比组件（v2.2 新增）
    └── diffUtils.js     # LCS diff 算法（v2.2 新增）
```

### 18.3 App.jsx 组件结构

```
App
├── Toolbar（场景选择器、模式选择器、二次检测开关、清空/导出按钮）
└── Workspace
    ├── PanelLeft
    │   ├── Tabs（对话模式 / 分析模式）
    │   ├── InputArea（用户输入 + 模型输出 textarea）
    │   ├── ButtonRow（发送 / 清空）
    │   └── QuickExamples（示例快速填充）
    └── PanelRight
        └── OutputSection
            ├── RiskBanner（绿色安全 / 红色告警）
            ├── MetaGrid（CFA 得分、回答策略、风险发现数、请求 ID）
            ├── RoutingInfo（意图、路由、应答策略）
            ├── RiskDetailCards（每个发现的还原链详情）
            ├── DiffView（LLM 原始输出 vs CFA 安全回答对比）
            ├── RawAnswer / SafeAnswer（回退单卡片展示）
            └── RawJSON（折叠完整响应）
```

### 18.4 API 客户端（api.js）

| 函数 | 端点 | 请求体字段 |
|------|------|-----------|
| `cfaChat({user_input, scenario, mode, secondary_check})` | `POST /api/cfa-chat` | user_input, scenario, mode, secondary_check |
| `cfaAnalyze({user_input, model_output, scenario, mode, secondary_check})` | `POST /api/cfa-analyze` | user_input, model_output, scenario, mode, secondary_check |

错误处理：HTTP 非 200 → 读取 response body 中的 `.error`，或回退到 `HTTP {status}`。

### 18.5 DiffView 组件（v2.2 新增）

**diffUtils.js**：LCS（最长公共子序列）算法，对中文+混合文本进行 token 级差异计算。

**DiffView.jsx**：左右双栏对比视图。
- 🟢 绿色背景：相同内容
- 🔴 红色背景+删除线：仅在 LLM 原始输出中存在（CFA 已移除/改写）
- 🔵 蓝色背景+虚线下划线：仅在 CFA 安全回答中存在（脱敏替换后的内容）

### 18.6 启动命令

```bash
cd ui && npm install && npm run dev
```

---

## 19. 配置体系

### 19.1 .env 环境变量

```
DEEPSEEK_API_KEY=sk-your-api-key-here    # 必填
DEEPSEEK_BASE_URL=https://api.deepseek.com  # 可选
DEEPSEEK_MODEL=deepseek-chat              # 可选
```

### 19.2 Policy JSON 完整字段说明

```json
{
  "uniqueness_k": 1,                 // 唯一还原阈值（k≤此值才算唯一）
  "display_field": "patient_name",   // 用于显示名称的字段
  "protected_fields": [...],         // 受限字段（被还原即风险）
  "identifier_fields": [...],        // 强标识（v2.2 精简）
  "quasi_identifier_fields": [...],  // 准标识（v2.2 新增）
  "field_order": [...],              // 字段处理优先级
  "field_labels": {...},             // 字段中文标签
  "field_weights": {...},            // 字段评分权重
  "field_aliases": {...},            // 别名映射（三段式：field→canonical→[aliases]）
  "public_rules": [...],             // 公开知识推理规则
  "safe_replacements": {...},        // 安全替换词
  "safe_hint": "...",                // 安全提示语
  "match_type_weights": {...},       // 匹配类型权重
  "llm_confidence_threshold": 0.4,   // LLM 最低置信度
  "llm_max_accepted_values": 30,     // 候选值上限
  "llm_extraction_fields": [...]     // LLM 重点关注的字段
}
```

### 19.3 事实池 JSON 格式

```json
[
  {
    "id": "P001",
    "patient_name": "张伟",
    "department": "心内科",
    "ward_type": "普通病房",
    "condition_summary": "胸痛待查",
    "diagnosis": "冠心病（稳定型心绞痛）",
    "medication": "阿司匹林 100mg qd + 阿托伐他汀 20mg qn",
    "insurance_level": "城镇职工医保",
    "admission_date": "2024-11-15",
    "doctor": "李主任"
  },
  ...
]
```
（医疗场景数据共 8 条记录 P001-P008）

### 19.4 语义别名 JSON 格式

```json
{
  "description": "LLM离线生成的语义别名",
  "semantic_aliases": {
    "medication": {
      "替格瑞洛 90mg bid + 阿司匹林 100mg qd + 美托洛尔 25mg bid": {
        "components": ["替格瑞洛", "阿司匹林", "美托洛尔"],
        "aliases": ["替格瑞洛", "阿司匹林", "替格瑞洛联合阿司匹林", "双抗"],
        "partial_clues": ["双抗", "抗血小板"],
        "possible_inferences": [],
        "partial_match_policy": "any_component_or_alias"
      }
    }
  }
}
```

---

## 20. 测试体系

### 20.1 测试分类

| 测试类 | 测试数 | 覆盖范围 |
|--------|--------|---------|
| `CFAScoreEngineTest` | 7 | 航天场景模式1全链路、边界条件 |
| `AnchorModelTest` | 4 | Anchor 模型的字段和行为 |
| `FieldPolicyTest` | 3 | FieldPolicy 配置和权重计算 |
| `SemanticIndexTest` | 5 | 语义索引候选召回（原有测试） |
| `AnchorMergerTest` | 2 | 锚点合并去重逻辑 |
| `HealthcareScenarioTest` | 3 | 医疗场景规则抽取 |
| `ModeAndWeightTest` | 2 | 模式切换和权重影响 |
| `FieldPolicyNewFeaturesTest` | 2 | quasi_identifier_fields 加载 |
| `HybridRetrieverTest` | 9 | v2.2 混合检索器（详下） |
| `FinanceRetrieverTest` | 2 | 金融场景检索器 |

### 20.2 HybirdRetriever 9 个测试

| 测试 | 覆盖场景 | 关键断言 |
|------|---------|---------|
| `test_exact_canonical_value_recalled` | "ICU" 出现在文本中 | `assertIn("ICU", ward_cands)` |
| `test_alias_recalled` | "重症监护室" 映射到 "ICU" | `assertIn("ICU", ward_cands)` |
| `test_component_recalled` | "双抗方案" 召回 medication 候选 | `assertGreater(len(med_cands), 0)` |
| `test_chinese_ngram_recall` | "急性心梗" 通过 n-gram 重叠召回 | `assertIn("心内科", dept_cands)` + `assertGreater(len(cond_cands), 0)` |
| `test_version_token_preserved` | `5.6.1`、`xz-utils`、`β→beta` 归一化 | `assertIn("5.6.1", tokens)` |
| `test_irrelevant_text_produces_few_results` | "今天天气很好" 返回很少候选 | `assertLessEqual(len(candidates), 5)` |
| `test_field_balancing` | 每字段 ≤ 8 候选 | `assertLessEqual(count, 8)` |
| `test_candidate_value_has_new_fields` | CandidateValue 新增字段存在 | `hasattr(c, "source")` |
| `test_field_hint_boosts_medication` | "用药 双抗" 应召回 medication | `assertGreater(len(med_candidates), 0)` |

### 20.3 运行测试

```bash
python -m pytest tests/test_engine.py -v --tb=short    # 39 个测试
```

---

## 21. 模块间数据流全景图

```
                       ┌──────────────────────────┐
                       │      配置加载层           │
                       │                          │
                       │  config/*.json ──► knowledge.py ──► AssetFact[], FieldPolicy
                       │  .env          ──► deepseek.py  ──► DeepSeekClient
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │      意图路由层 (v2.1)    │
                       │                          │
                       │  intent_router.py        │
                       │    classify_intent()      │
                       │      ├─ 领域请求 → CFA    │
                       │      └─ 天气/闲聊 → 通用  │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │      生成层              │
                       │                          │
                       │  adapter.py              │
                       │    user_input + fact_pool │
                       │          → raw_answer    │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │      检测层              │
                       │                          │
                       │  engine.py (CFAScoreEngine)
                       │    ├─ extractor.py        │
                       │    │    精确匹配+别名+推断 → rule_anchors
                       │    │                       │
                       │    ├─ [Mode 2/3]           │
                       │    │   semantic_index.py   │
                       │    │     └─ retriever.py   │ (v2.2)
                       │    │         BM25+N-gram   │
                       │    │           → 候选值    │
                       │    │   llm_extractor.py    │
                       │    │     LLM 语义抽取       │
                       │    │   anchor_verifier.py  │
                       │    │     7条校验 ← verified
                       │    │                       │
                       │    ├─ AnchorMerger.merge() │
                       │    ├─ _build_findings()    │
                       │    │   逐个事实 → 还原链    │
                       │    │   → 唯一性判断(≤k)     │
                       │    │   → _score() 评分     │
                       │    │   → 最小锚点组合       │
                       │    └─ _llm_safe_rewrite()  │
                       │        [Mode 3] LLM改写    │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │      改写层              │
                       │                          │
                       │  sanitizer.py            │
                       │    make_x_replaced()      │
                       │    make_safe_answer()     │
                       │    _replace_asset_ids()   │
                       └──────────────────────────┘
                                    │
                                    ▼
                       ┌──────────────────────────┐
                       │      对外层 (v2.1)       │
                       │                          │
                       │  gateway.py (CFAGateway)  │
                       │    handle_chat()          │
                       │    handle_analyze()       │
                       │    _select_final_answer() │
                       │                          │
                       │  server.py (CFAServer)    │
                       │    POST /api/cfa-chat     │
                       │    POST /api/cfa-analyze  │
                       │    GET  /api/health       │
                       │    GET  /api/scenarios    │
                       │                          │
                       │  ui/ (React + Vite)       │
                       │    proxy /api → :8080     │
                       │    DiffView 对比展示      │
                       │                          │
                       │  cli.py (命令行)           │
                       │    cfa-score analyze      │
                       └──────────────────────────┘
```

**数据实体流转**：
```
user_input (str) + model_output (str)
  ↓ extractor.py + llm_extractor.py + anchor_verifier.py
Anchor[] (field_name, text, canonical_value, source, match_type, confidence)
  ↓ engine.py _build_findings()
RiskFinding[] (target_asset, risk_level, score, reduction_chain, anchors)
  ↓ sanitizer.py
safe_answer (str), x_replaced_answer (str)
  ↓ gateway.py + server.py
GatewayResponse { answer: str, risk_detected: bool, ... }
  ↓ 外部调用方
仅使用 answer 字段
```

---

## 22. v2.2 变更记录

| 日期 | 变更 | 影响文件 |
|------|------|---------|
| 2026-07-08 | 新增 `quasi_identifier_fields` 字段分类，修复 false negative | `models.py`, `engine.py`, 4 个 `config/*_policy.json` |
| 2026-07-08 | `_has_sensitive_restoration_shape()` 重写：接受准标识字段作为定位器 | `engine.py` |
| 2026-07-08 | `_score()` 累计 quasi_identifier 字段权重 | `engine.py` |
| 2026-07-08 | 新增 `HybridSparseRetriever`（BM25 + N-gram + Alias + Hint 四路融合） | `retriever.py`（新文件） |
| 2026-07-08 | `CandidateValue` 扩展 3 个新字段（向后兼容） | `models.py` |
| 2026-07-08 | `SemanticIndex` 集成 `HybridSparseRetriever` | `semantic_index.py` |
| 2026-07-08 | 新增 DiffView 文本对比组件 | `ui/src/DiffView.jsx`, `ui/src/diffUtils.js`（新文件） |
| 2026-07-08 | 新增 11 个混合检索器测试 | `tests/test_engine.py` |
| 2026-07-08 | 测试总数：28 → 39 | — |