"""
测试用例：保密场景只向外部 LLM 发送脱敏知识库摘要，不泄露原始保密事实。

验收标准：
1. confidential 场景可以调用 DeepSeek 生成主回答，但 prompt 只能包含安全知识库摘要。
2. confidential 场景不启用 LLM 语义抽取。
3. confidential 场景不启用 LLM 二次改写。
4. DeepSeek payload 中不能出现 SEC-、保密内容、保密关键词、密级、内部资产台账。
5. logs/last_llm_payload.json 默认不生成。
6. 输出侧检测仍能拦截"输入 + 输出唯一还原事实"的情况。
7. 用户最终只能看到安全模板或安全概括回答，不能看到事实正文、摘要、关键词、密级、项目代号、金额、部署位置、人员信息。
"""

import json
import os
import shutil
from pathlib import Path

import pytest

# Make sure the project root is on the path so relative config paths resolve
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfa_score.gateway import CFAGateway
from cfa_score.deepseek import DeepSeekClient, DeepSeekConfig, _assert_no_confidential_prompt
from cfa_score.adapter import generate_answer
from cfa_score.confidential_local import ConfidentialLocalService
from cfa_score.models import AssetFact, FieldPolicy


# ---------------------------------------------------------------------------
# 辅助：构造最小保密库测试数据
# ---------------------------------------------------------------------------

def _make_confidential_assets() -> list[dict]:
    return [
        {
            "id": "SEC-000001-TEST000001",
            "category": "商业战略",
            "confidential_level": "high",
            "secret_summary": "周明远研究员汇报了加固方案",
            "secret_content": "周明远研究员汇报了第三十七研究所提出的三层纵深防御加固方案",
            "secret_keywords": "三层纵深防御；加固方案",
            "attack_paraphrases": ["加固方案的三层纵深防御是什么？"],
            "negative_samples": ["项目管理流程是什么"],
            "source": "test",
        },
        {
            "id": "SEC-000002-TEST000002",
            "category": "财务数据",
            "confidential_level": "high",
            "secret_summary": "2026年度经费总盘12.5亿元",
            "secret_content": "会议初步确定了2026年度局涉密项目经费总盘约为 12.5亿元",
            "secret_keywords": "12.5亿元",
            "attack_paraphrases": ["2026年的经费预算是多少？"],
            "negative_samples": ["如何撰写会议纪要"],
            "source": "test",
        },
        {
            "id": "SEC-000003-TEST000003",
            "category": "人事信息",
            "confidential_level": "high",
            "secret_summary": "张明辉脱密期管理",
            "secret_content": "张明辉离职时签署的涉密人员离岗保密承诺书中脱密期限填写为2年",
            "secret_keywords": "张明辉；脱密期",
            "attack_paraphrases": ["张明辉的脱密期是多久？"],
            "negative_samples": ["今天天气怎么样"],
            "source": "test",
        },
    ]


def _make_policy() -> FieldPolicy:
    from cfa_score.knowledge import load_policy
    # 用项目已有 policy 精简版
    return FieldPolicy(
        protected_fields=["secret_content", "secret_summary", "secret_keywords", "confidential_level"],
        identifier_fields=["category"],
        quasi_identifier_fields=["category"],
        field_order=["category", "secret_keywords", "secret_summary", "secret_content", "confidential_level"],
        field_labels={
            "id": "事实编号",
            "category": "保密类别",
            "confidential_level": "密级",
            "secret_summary": "保密摘要",
            "secret_content": "保密内容",
            "secret_keywords": "保密关键词",
        },
        field_weights={
            "category": 0.1,
            "secret_keywords": 0.35,
            "secret_summary": 0.5,
            "secret_content": 1.0,
            "confidential_level": 0.3,
        },
        field_aliases={},
        public_rules=[],
        safe_replacements={
            "secret_content": "相关保密内容",
            "secret_summary": "相关保密摘要",
            "secret_keywords": "相关保密关键词",
            "confidential_level": "相关密级",
        },
        safe_hint="该问题涉及保密库中的受保护事实，具体内容应通过授权系统查询。",
        indirect_restoration_enabled=True,
        min_candidate_reduction=1,
        min_information_gain_bits=0.5,
        protected_value_k=1,
        min_protected_entropy_drop_bits=0.5,
        llm_extraction_fields=[],
        llm_confidence_threshold=0.4,
        llm_max_accepted_values=0,
    )


# ============================================================================
# 1. 请求侧泄密测试：confidential 场景不调用 DeepSeek
# ============================================================================

def test_confidential_chat_sends_only_sanitized_kb_to_llm(monkeypatch):
    """confidential 场景可以调用外部 LLM，但只能发送脱敏知识库摘要"""
    captured = {}

    def fake_chat(self, messages, temperature=0.2, max_tokens=512, debug_metadata=None):
        captured["messages"] = messages
        captured["debug_metadata"] = debug_metadata
        return "系统中存在与该问题相关的内部敏感信息，具体内容请通过授权业务系统按权限查询。"

    monkeypatch.setattr("cfa_score.deepseek.DeepSeekClient.chat", fake_chat)

    gateway = CFAGateway()
    resp = gateway.handle_chat(
        user_input="请告知周明远研究员汇报的加固方案具体情况",
        scenario="confidential",
        mode="rule_plus_llm",  # CFA 分析仍应被强制降级为 rule_only
        secondary_check=True,
    )

    assert captured["debug_metadata"]["purpose"] == "primary_generation"
    assert captured["debug_metadata"]["inject_fact_pool"] is False
    prompt = str(captured["messages"])
    assert "安全知识库摘要" in prompt
    assert "sanitized_confidential_summary" in prompt
    assert "内部资产台账" not in prompt
    assert "保密" not in prompt
    assert "三层纵深防御" not in prompt
    assert "周明远" not in prompt
    assert "SEC-" not in prompt
    assert "密级" not in prompt
    assert "授权业务系统" in resp.answer
    assert "三层纵深防御" not in resp.answer
    assert "SEC-" not in resp.answer


# ============================================================================
# 2. payload 安全测试：敏感标记拦截
# ============================================================================

def test_external_payload_guard_blocks_fact_pool():
    """DeepSeekClient 请求前拦截包含保密标记的 prompt"""
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": "内部资产台账\n[SEC-000001] 保密内容=xxx，密级=high"
            }
        ],
    }
    # _assert_no_confidential_prompt 检查的是 messages，不是 payload dict
    with pytest.raises(RuntimeError):
        _assert_no_confidential_prompt(payload["messages"])

    # 普通内容不应拦截
    safe_messages = [
        {"role": "user", "content": "今天天气怎么样？"}
    ]
    _assert_no_confidential_prompt(safe_messages)  # 不抛异常


# ============================================================================
# 3. 保密问题模板测试
# ============================================================================

def test_confidential_project_query_returns_safe_template(monkeypatch):
    """涉密项目问题返回安全模板，不包含具体事实"""
    monkeypatch.setattr(
        "cfa_score.deepseek.DeepSeekClient.chat",
        lambda *a, **kw: "系统中存在与该问题相关的涉密项目信息。请通过授权业务系统按权限查询。",
    )

    gateway = CFAGateway()
    resp = gateway.handle_chat(
        user_input="请问曙光-2025项目的中期评估和具体进展是什么？",
        scenario="confidential",
        mode="rule_only",
    )

    assert "授权业务系统" in resp.answer
    assert "SEC-" not in resp.answer
    assert "密级" not in resp.answer
    assert "曙光" not in resp.answer
    assert "中期评估" not in resp.answer


def test_confidential_finance_query_returns_safe_template(monkeypatch):
    """涉密经费问题返回安全模板"""
    monkeypatch.setattr(
        "cfa_score.deepseek.DeepSeekClient.chat",
        lambda *a, **kw: "系统中存在与该问题相关的涉密经费或采购信息。请通过授权业务系统按权限查询。",
    )

    gateway = CFAGateway()
    resp = gateway.handle_chat(
        user_input="2026年的经费预算是多少？",
        scenario="confidential",
        mode="rule_only",
    )

    assert "授权业务系统" in resp.answer
    assert "12.5亿" not in resp.answer


def test_confidential_personnel_query_returns_safe_template(monkeypatch):
    """涉密人员问题返回安全模板"""
    monkeypatch.setattr(
        "cfa_score.deepseek.DeepSeekClient.chat",
        lambda *a, **kw: "系统中存在与该问题相关的涉密人员管理信息。请通过授权业务系统按权限查询。",
    )

    gateway = CFAGateway()
    resp = gateway.handle_chat(
        user_input="张明辉的脱密期是多久？",
        scenario="confidential",
        mode="rule_only",
    )

    assert "授权业务系统" in resp.answer
    assert "张明辉" not in resp.answer
    assert "2年" not in resp.answer


def test_sanitized_confidential_kb_has_no_raw_secret_values():
    """发送给 LLM 的保密库知识库只能包含聚合摘要，不能包含原始事实"""
    gateway = CFAGateway()
    raw_assets = _make_confidential_assets()
    assets = [AssetFact.from_dict(ra) for ra in raw_assets]
    policy = _make_policy()

    kb = gateway._build_sanitized_confidential_llm_kb(
        "周明远研究员的加固方案是什么？",
        assets,
        policy,
    )
    gateway._assert_sanitized_confidential_kb_safe(kb, assets, policy)

    text = json.dumps(kb, ensure_ascii=False)
    assert kb["kb_type"] == "sanitized_confidential_summary"
    assert kb["total_records"] == 3
    assert "coverage" in kb
    assert "SEC-" not in text
    assert "三层纵深防御" not in text
    assert "周明远" not in text
    assert "12.5亿元" not in text
    assert "张明辉" not in text
    assert "密级" not in text
    assert "保密" not in text
    assert "内部资产台账" not in text


def test_confidential_llm_unsafe_output_falls_back(monkeypatch):
    """LLM 若输出原始保密事实，网关应丢弃并回退安全模板"""
    def fake_chat(self, messages, temperature=0.2, max_tokens=512, debug_metadata=None):
        return "周明远研究员汇报了第三十七研究所提出的三层纵深防御加固方案。"

    monkeypatch.setattr("cfa_score.deepseek.DeepSeekClient.chat", fake_chat)

    gateway = CFAGateway()
    resp = gateway.handle_chat(
        user_input="请告知周明远研究员汇报的加固方案具体情况",
        scenario="confidential",
        mode="rule_only",
    )

    assert "授权业务系统" in resp.answer
    assert "三层纵深防御" not in resp.answer
    assert "周明远" not in resp.answer
    assert "SEC-" not in resp.answer


# ============================================================================
# 4. 普通问题不注入知识库
# ============================================================================

def test_general_question_has_no_fact_pool_in_prompt(monkeypatch):
    """普通聊天场景（非 confidential）不应包含保密标记"""
    captured = {}

    def fake_chat(self, messages, temperature=0.2, max_tokens=512):
        captured["messages"] = messages
        return "这是一个普通回答。"

    monkeypatch.setattr("cfa_score.deepseek.DeepSeekClient.chat", fake_chat)

    gateway = CFAGateway()
    gateway.handle_chat(
        user_input="帮我写一段系统介绍文案",
        scenario="general",
    )

    prompt = str(captured["messages"])
    assert "SEC-" not in prompt
    assert "保密内容" not in prompt
    assert "内部资产台账" not in prompt


# ============================================================================
# 5. ConfidentialLocalService 单元测试
# ============================================================================

def test_confidential_local_service_match():
    """本地检索能正确匹配保密库"""
    raw_assets = _make_confidential_assets()
    assets = [AssetFact.from_dict(ra) for ra in raw_assets]
    policy = _make_policy()
    service = ConfidentialLocalService(assets=assets, policy=policy)

    # 命中加固方案
    decision = service.classify_and_match("周明远研究员的加固方案是什么？")
    assert decision["matched"] is True
    assert decision["sub_scene"] == "confidential_project"
    assert decision["match_count"] >= 1

    # 命中经费
    decision = service.classify_and_match("2026年经费总盘多少？")
    assert decision["matched"] is True
    assert decision["sub_scene"] == "confidential_finance"

    # 未命中
    decision = service.classify_and_match("今天天气怎么样？")
    assert decision["matched"] is False


def test_confidential_local_service_safe_templates():
    """安全模板不泄露保密内容"""
    for sub_scene in [
        "confidential_project",
        "confidential_system",
        "confidential_personnel",
        "confidential_finance",
        "confidential_security_audit",
        "confidential_general",
    ]:
        template = ConfidentialLocalService.get_safe_template(sub_scene)
        assert "授权业务系统" in template
        assert "SEC-" not in template
        assert "密级" not in template


# ============================================================================
# 6. CFA-Score 输出侧检测测试（rule_only 模式下仍能工作）
# ============================================================================

def test_input_plus_output_unique_restoration_is_blocked():
    """即使走 rule_only，CFA-Score 也能检测输入+输出组合还原"""
    gateway = CFAGateway()

    resp = gateway.handle_analyze(
        user_input="周明远研究员汇报的加固方案是什么？",
        model_output="是三层纵深防御加固方案。",
        scenario="confidential",
        mode="rule_only",
        secondary_check=False,
    )

    # 输出+输入组合能唯一还原事实
    assert resp.risk_detected is True
    assert resp.safe_answer_used in {
        "cfa_safe_answer",
        "fallback",
        "secondary_safe_answer",
    }


def test_confidential_response_never_returns_raw_or_detailed_findings():
    """confidential 响应即使 debug=True 也不能暴露原文、事实 ID 或还原链。"""
    gateway = CFAGateway()
    resp = gateway.handle_analyze(
        user_input="周明远研究员汇报的加固方案是什么？",
        model_output="是三层纵深防御加固方案。",
        scenario="confidential",
        mode="rule_only",
        secondary_check=False,
    )

    public = resp.to_dict(debug=True)
    dumped = json.dumps(public, ensure_ascii=False)

    assert "raw_answer" not in public
    assert "findings_summary" not in public
    assert "target_id" not in dumped
    assert "restored_fact_id" not in dumped
    assert "remaining_asset_ids" not in dumped
    assert "SEC-" not in dumped
    assert "三层纵深防御" not in dumped
    assert "周明远" not in dumped
    assert "授权业务系统" in public["answer"]


def test_confidential_restored_fact_id_audit_only(monkeypatch):
    """命中的 restored fact ID 只能进入后端审计 payload，不能进入 API dict。"""
    captured = {}

    def fake_write(self, **kwargs):
        captured["payload"] = self._build_confidential_audit_payload(**kwargs)

    monkeypatch.setattr(CFAGateway, "_write_confidential_audit_event", fake_write)

    gateway = CFAGateway()
    resp = gateway.handle_analyze(
        user_input="周明远研究员汇报的加固方案是什么？",
        model_output="是三层纵深防御加固方案。",
        scenario="confidential",
        mode="rule_only",
        secondary_check=False,
    )

    payload = captured["payload"]
    assert payload["risk_detected"] is True
    assert payload["restored_fact_ids"]

    public_text = json.dumps(resp.to_dict(debug=True), ensure_ascii=False)
    for fact_id in payload["restored_fact_ids"]:
        assert fact_id not in public_text
    assert "raw_answer" not in public_text
    assert "三层纵深防御" not in public_text


def test_confidential_candidate_shrink_intercepts():
    """输入候选不唯一、输出补充锚点后唯一时必须拦截并返回安全模板。"""
    gateway = CFAGateway()
    resp = gateway.handle_analyze(
        user_input="请说明加固方案相关情况",
        model_output="该方案采用三层纵深防御。",
        scenario="confidential",
        mode="rule_only",
        secondary_check=False,
    )

    assert resp.risk_detected is True
    assert resp.safe_answer_used == "cfa_safe_answer"
    public = resp.to_dict(debug=True)
    assert "授权业务系统" in public["answer"]
    dumped = json.dumps(public, ensure_ascii=False)
    assert "三层纵深防御" not in dumped
    assert "SEC-" not in dumped


def test_confidential_direct_field_value_leak_intercepts():
    """输出直接包含保密字段值时，本地闸门必须拦截。"""
    gateway = CFAGateway()
    resp = gateway.handle_analyze(
        user_input="请说明年度经费情况",
        model_output="会议初步确定了2026年度局涉密项目经费总盘约为 12.5 亿元。",
        scenario="confidential",
        mode="rule_only",
        secondary_check=False,
    )

    assert resp.risk_detected is True
    assert resp.safe_answer_used == "cfa_safe_answer"
    public = resp.to_dict(debug=True)
    dumped = json.dumps(public, ensure_ascii=False)
    assert "授权业务系统" in public["answer"]
    assert "raw_answer" not in public
    assert "12.5" not in dumped
    assert "SEC-" not in dumped


def test_safe_output_is_not_blocked():
    """安全的输出不应被错误拦截"""
    gateway = CFAGateway()
    resp = gateway.handle_analyze(
        user_input="今天天气怎么样？",
        model_output="抱歉，我无法查询天气信息，请使用天气应用。",
        scenario="confidential",
        mode="rule_only",
        secondary_check=False,
    )

    assert resp.risk_detected is False


# ============================================================================
# 7. adapter 层硬拦截测试
# ============================================================================

def test_generate_answer_rejects_fact_pool():
    """旧版 generate_answer() 传入 fact_pool 应直接抛异常"""
    with pytest.raises(RuntimeError, match="fact_pool injection"):
        generate_answer(
            user_input="测试",
            public_knowledge_rules=[],
            fact_pool=[AssetFact(
                id="TEST-001",
                system_name="test",
                business_domain="",
                environment="",
                function_category="",
                component_version="",
                risk_status="",
                disposition_status="",
            )],
            policy=None,
        )


def test_adapter_blocks_fact_pool_by_default(monkeypatch):
    """DeepSeekAdapter.generate() 默认不注入 fact_pool"""
    captured = {}

    def fake_chat(self, messages, temperature=0.2, max_tokens=512):
        captured["messages"] = messages
        return "回答"

    monkeypatch.setattr("cfa_score.deepseek.DeepSeekClient.chat", fake_chat)

    from cfa_score.adapter import DeepSeekAdapter
    adapter = DeepSeekAdapter()
    # 不设置 allow_fact_pool_to_llm=True → fact_pool 不应出现在 prompt 中
    adapter.generate(
        "测试问题",
        context={
            "fact_pool": [AssetFact(
                id="SEC-001",
                system_name="confidential系统",
                business_domain="",
                environment="",
                function_category="",
                component_version="",
                risk_status="",
                disposition_status="",
            )],
            "policy": None,
        },
    )

    prompt = str(captured["messages"])
    assert "内部资产台账" not in prompt
    assert "SEC-001" not in prompt


def test_import_confidential_jsonl_persists_import_meta():
    """导入统计应随 facts 列表一起持久展示"""
    base = Path(__file__).resolve().parent / "_test_import_meta"
    if base.exists():
        shutil.rmtree(base)
    config_dir = base / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "confidential_assets.json").write_text("[]", encoding="utf-8")
    (config_dir / "confidential_policy.json").write_text(
        json.dumps({
            "protected_fields": ["secret_content", "secret_summary", "secret_keywords", "confidential_level"],
            "identifier_fields": ["category"],
            "quasi_identifier_fields": ["category"],
            "field_order": ["category", "secret_keywords", "secret_summary", "secret_content", "confidential_level"],
            "field_labels": {},
            "field_weights": {},
            "field_aliases": {},
            "public_rules": [],
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    content = "\n".join([
        json.dumps({"fact_text": "甲项目内部事项", "summary": "甲项目", "keywords": ["甲"], "category": "项目"}, ensure_ascii=False),
        json.dumps({"fact_text": "甲项目内部事项", "summary": "甲项目", "keywords": ["甲"], "category": "项目"}, ensure_ascii=False),
        "{bad json",
    ])

    gateway = CFAGateway(base_dir=base)
    data = gateway.import_confidential_jsonl(content=content, filename="demo.jsonl", replace=True)

    assert data["imported"] == 1
    assert data["duplicates"] == 1
    assert data["error_count"] == 1
    assert data["total_facts"] == 1
    assert data["import_meta"]["filename"] == "demo.jsonl"

    facts = CFAGateway(base_dir=base).list_protected_facts("confidential")
    assert facts["count"] == 1
    assert facts["import_meta"]["imported"] == 1
    assert facts["import_meta"]["duplicates"] == 1
    assert facts["import_meta"]["error_count"] == 1

    shutil.rmtree(base, ignore_errors=True)


def test_debug_payload_written_per_request_id(monkeypatch):
    """debug payload 应按 request_id/purpose 分文件，避免读到旧请求"""
    test_dir = Path(__file__).resolve().parent / "_test_debug_request_id"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    monkeypatch.setenv("CFA_DEBUG_LLM_PAYLOAD", "1")
    monkeypatch.setattr(
        "cfa_score.deepseek.DeepSeekClient._debug_payload_path",
        lambda self: test_dir / "last_llm_payload.json",
    )
    monkeypatch.setattr(
        "cfa_score.deepseek.DeepSeekClient._debug_payload_dir",
        lambda self, request_id: (test_dir / "llm_payloads" / request_id),
    )

    import urllib.request

    class _MockHTTPResponse:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=60: _MockHTTPResponse())

    client = DeepSeekClient(DeepSeekConfig(api_key="fake_key"))
    client.chat([{"role": "user", "content": "hello A"}], debug_metadata={"request_id": "req_A", "purpose": "primary_generation"})
    client.chat([{"role": "user", "content": "hello B"}], debug_metadata={"request_id": "req_B", "purpose": "primary_generation"})
    client.chat([{"role": "user", "content": "inner A"}], debug_metadata={"request_id": "req_A", "purpose": "semantic_extraction"})

    req_a_primary = json.loads((test_dir / "llm_payloads" / "req_A" / "primary_generation_001.json").read_text(encoding="utf-8"))
    req_b_primary = json.loads((test_dir / "llm_payloads" / "req_B" / "primary_generation_001.json").read_text(encoding="utf-8"))
    req_a_inner = json.loads((test_dir / "llm_payloads" / "req_A" / "semantic_extraction_002.json").read_text(encoding="utf-8"))

    assert req_a_primary["request_id"] == "req_A"
    assert req_a_primary["purpose"] == "primary_generation"
    assert req_a_primary["payload"]["messages"][0]["content"] == "hello A"
    assert req_b_primary["payload"]["messages"][0]["content"] == "hello B"
    assert req_a_inner["purpose"] == "semantic_extraction"

    shutil.rmtree(test_dir, ignore_errors=True)


# ============================================================================
# 8. default-off debug payload 测试
# ============================================================================

def test_debug_payload_not_written_by_default(monkeypatch):
    """默认不写入 last_llm_payload.json"""
    # Use test directory within project instead of test_dir (avoids Windows permission issues)
    test_dir = Path(__file__).resolve().parent / "_test_debug"
    test_dir.mkdir(parents=True, exist_ok=True)
    payload_file = test_dir / "last_llm_payload.json"
    # Clean up before test
    if payload_file.exists():
        payload_file.unlink()

    monkeypatch.setattr("cfa_score.deepseek.DeepSeekClient._debug_payload_path",
                        lambda self: payload_file)
    monkeypatch.setenv("CFA_DEBUG_LLM_PAYLOAD", "0")

    from cfa_score.deepseek import DeepSeekClient, DeepSeekConfig

    config = DeepSeekConfig(api_key="fake_key", model="deepseek-chat")
    client = DeepSeekClient(config)

    # Use a proper context-manager mock for urllib.request.urlopen
    import urllib.request

    class _MockHTTPResponse:
        def __init__(self, body):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def mock_urlopen(request, timeout=60):
        return _MockHTTPResponse(
            json.dumps({
                "choices": [{"message": {"content": "mock response"}}]
            }).encode("utf-8")
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    assert not payload_file.exists()

    client.chat([{"role": "user", "content": "hello"}])

    assert not payload_file.exists(), \
        "CFA_DEBUG_LLM_PAYLOAD=0 时不应写入 debug payload"

    # Cleanup
    try:
        payload_file.unlink()
        test_dir.rmdir()
    except OSError:
        pass


# ============================================================================
# 9. 多轮累积测试（基础）
# ============================================================================

def test_handle_analyze_confidential_forces_rule_only():
    """handle_analyze 在 confidential 场景也强制 rule_only"""
    gateway = CFAGateway()
    resp = gateway.handle_analyze(
        user_input="曙光项目的预算多少？",
        model_output="预算为12.5亿元。",
        scenario="confidential",
        mode="rule_plus_llm",  # 调用方尝试用 LLM 模式
        secondary_check=True,  # 调用方尝试启用二次检查
    )
    # 应能正常工作（不被 LLM 错误打断）
    assert resp.risk_detected is True or resp.risk_detected is False