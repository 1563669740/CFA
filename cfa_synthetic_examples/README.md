# CFA 合成测试样例

本目录全部为虚构测试数据，不对应任何真实人员、项目、系统或机构。

文件：
- `secret_library.jsonl`：保密库记录
- `internal_kb.jsonl`：与保密记录一一对应的内部知识库文档
- `test_cases.jsonl`：正向泄密测试与负向误拦截测试

建议：
1. `secret_keywords` 同时放实体、数值、事件名和常见别名。
2. `attack_paraphrases` 必须逐条定制，避免所有记录共用同一模板。
3. `negative_samples` 选取同领域但不涉及具体秘密的近邻问题。
4. 测试时同时统计秘密召回率、误拦截率和安全回答可用性。
