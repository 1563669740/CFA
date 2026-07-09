# CFA-Score 通用模板：组合事实还原风险检测

这个项目用于检测：当用户输入和模型输出中的多个关键词/锚点组合起来，是否能够把已经写好的受限事实池压缩到唯一事实，并间接还原低权限用户不应直接获得的受限字段。

它不是只能检测 `CFA_Score_航天测控_XZ_组合事实还原风险_最终案例.docx` 里的案例。那个 docx 只是样例材料；真正通用的部分是：

- 受限事实池：`config/*.json` 中的事实记录；
- 字段策略：哪些字段是范围/识别锚点，哪些字段是受限字段；
- 关键词条/别名：自然语言说法如何映射到标准字段值；
- 推断规则：用 `policy.public_rules` 或 `config/public_knowledge.sample.json` 模拟公开知识推断；Answer 也可以切换为 DeepSeek 生成。

## PyCharm 右键运行

推荐直接右键运行项目根目录下的：

```text
main.py
```

**运行前需要先在项目根目录创建 `.env` 文件**（参考 `.env.example`）：

```text
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

默认会读取：

```text
examples/user_input.txt              # 用户输入（模拟低权限用户的提问）
config/assets.sample.json            # 样例受限事实池
config/policy.sample.json            # 样例字段策略
config/public_knowledge.sample.json  # 样例公开知识
```

Answer 由 DeepSeek API 根据以上信息在线生成。

运行后会在控制台打印：

- 抽取到的锚点数量；
- 是否命中唯一受限事实；
- 被还原出的受限字段；
- `key anchors`，即导致唯一还原的关键锚点组合；
- 安全改写后的回答。

完整 JSON 报告会写到：

```text
report.json
```

如果要换自己的数据，优先改 `main.py` 顶部这几个常量：

```python
FACTS_PATH = BASE_DIR / "config" / "assets.sample.json"
POLICY_PATH = BASE_DIR / "config" / "policy.sample.json"
PUBLIC_KNOWLEDGE_PATH = BASE_DIR / "config" / "public_knowledge.sample.json"
USER_INPUT_PATH = BASE_DIR / "examples" / "user_input.txt"
```

也可以直接把文本填到：

```python
USER_INPUT_TEXT = ""
```

非空时会优先使用这里的文本。

## 通用模板怎么填

可以参考：

```text
config/facts.template.json
config/policy.template.json
```

事实池可以是任意领域的数据，只要每条记录有唯一 `id`，并把字段写进 JSON。例如：

```json
{
  "id": "FACT-001",
  "name": "受限事实对象A",
  "scene": "范围关键词A",
  "scope": "业务范围A",
  "public_trait": "公开线索A",
  "restricted_fact": "受限事实值A",
  "restricted_status": "受限状态A"
}
```

策略文件里重点配置：

```json
{
  "display_field": "name",
  "protected_fields": ["restricted_fact", "restricted_status"],
  "identifier_fields": ["scene", "scope", "public_trait"],
  "field_order": ["scene", "scope", "public_trait", "restricted_fact", "restricted_status"]
}
```

含义：

- `protected_fields`：受限事实字段，低权限用户不能直接得到；
- `identifier_fields`：范围、类别、关键词条等锚点字段，本身可能不是秘密，但能缩小候选；
- `field_aliases`：把用户/模型里的自然语言关键词映射到标准值；
- `public_rules`：模拟 LLM 或公开知识推断，例如“公开线索A => 受限状态A”。

## 当前检测流程

1. 同时读取用户输入和模型输出。
2. 从两段文本中抽取锚点，并标注来源：`用户输入` 或 `模型输出`。
3. 用锚点逐步过滤受限事实池。
4. 如果组合后只剩唯一记录，并且组合中包含受限字段，就生成风险发现。
5. 输出 `key_anchor_ids` 和 `key_anchor_summary`，标注导致唯一还原的关键锚点。
6. 只对模型输出做 X 替换和安全改写，不改写用户原始输入。

## LLM 介入方式

Answer 统一由 DeepSeek API 在线生成。DeepSeek 接收用户提问、公开漏洞情报、内部资产台账，扮演企业 AI 运维助手角色来生成回答。客户端读取项目根目录 `.env` 中的配置。

锚点抽取仍使用基于规则的确定性抽取器（精确匹配 + 别名映射 + 公开知识推断）：

- `field_aliases`：关键词/别名识别；
- `public_rules` / `public_knowledge.sample.json`：根据公开知识或上下文进行推断。

核心评分和唯一还原逻辑不依赖外部模型；如果以后要做真实 NER/锚点抽取，只需要让抽取器输出与 `Anchor` 等价的结构：字段名、标准值、原文片段、来源、是否推断。

## 测试

仍然可以用测试确认模板逻辑：

```bash
python -m unittest discover -s tests
```

目前测试覆盖了：

- 原样例风险检测；
- 用户输入 + 模型输出组合检测；
- 关键锚点标注；
- 版本号边界，避免 `5.6.10` 被误识别成 `5.6.1`；
- 推断锚点不能跨事实记录误匹配；
- 锚点 ID 稳定可复现。

## 公开知识和 DeepSeek Answer 生成

现在项目把四个实验输入显式分开：

```text
用户输入 Input              examples/user_input.txt
公开知识 Public Knowledge   config/public_knowledge.sample.json
内部知识库事实池 Fact Pool   config/assets.sample.json
LLM 输出 Answer             由 DeepSeek 在线生成
```

DeepSeek 被提示为"企业内部的AI运维助手"，接收用户问题 + 公开漏洞情报 + 内部资产台账，生成自然的专业回答。CFA-Score 随后检测该回答与用户输入组合后是否能唯一还原受限事实。

运行前确保 `.env` 已配置：

```text
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

右键运行 `main.py` 即可。也可以用 CLI：

```bash
python -m cfa_score.cli analyze \
  --input examples/user_input.txt \
  --facts config/assets.sample.json \
  --policy config/policy.sample.json \
  --public-knowledge config/public_knowledge.sample.json \
  --env .env
```

注意：DeepSeek 生成模式会把 Fact Pool 发送给外部模型，请只用于可外发的实验数据。


