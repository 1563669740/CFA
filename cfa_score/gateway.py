"""
CFA-Score LLM Safety Gateway.

Orchestrates the full pipeline:

  user_input
    → LLM generation (DeepSeek) → raw_answer
    → CFA-Score analysis (engine.analyze)
    → safe_answer / secondary_safe_answer selection
    → return CFA-processed answer plus raw-answer transparency for non-sensitive demos

Two operation modes:
  - ``handle_chat()``    — Full pipeline: LLM generate + CFA analyze + safe answer
  - ``handle_analyze()`` — CFA only: user provides model_output, we run detection + sanitize

Design principles:
  - Zero external dependencies (Python stdlib only)
  - Scenario assets/policy are loaded once and cached
  - CFAScoreEngine is created per request (thread-safe, lightweight)
  - confidential raw_answer, anchors, reduction_chain are NEVER returned to callers
"""

from __future__ import annotations

import json
import uuid
import hashlib
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

from .adapter import DeepSeekAdapter
from .confidential_local import ConfidentialLocalService
from .deepseek import DeepSeekClient, config_from_env, _assert_no_confidential_prompt
from .engine import CFAScoreEngine, ExtractionMode
from .intent_router import (
    classify_intent,
    is_domain_intent,
    map_intent_to_scenario,
    get_general_system_prompt,
    DOMAIN_HEALTHCARE,
    DOMAIN_FINANCE,
    DOMAIN_AEROSPACE,
    DOMAIN_MEETINGS,
    GENERAL_WEATHER,
    GENERAL_CHAT,
    AMBIGUOUS,
)
from .knowledge import load_assets, load_policy, load_public_knowledge, load_semantic_aliases, merge_public_knowledge
from .models import AnalysisResult, FieldPolicy, AssetFact, RiskFinding


# ---------------------------------------------------------------------------
# Response model (what the HTTP layer serializes)
# ---------------------------------------------------------------------------

@dataclass
class GatewayResponse:
    """Public-facing response with confidential-safe serialization guard."""
    request_id: str
    answer: str
    raw_answer: str           # LLM 原始输出（对话模式）或 model_output（分析模式）
    risk_detected: bool
    risk_level: str          # "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    score: float
    safe_answer_used: str    # "raw_answer" | "cfa_safe_answer" | "secondary_safe_answer" | "fallback" | "general_answer"
    findings_count: int
    findings_summary: List[Dict[str, Any]] = field(default_factory=list)
    # New routing/transparency fields
    intent: str = ""            # classified intent domain
    routed_scenario: str = ""   # the scenario that actually handled the request
    answer_strategy: str = ""   # "cfa_gated" | "general_answer" | "weather_answer" | "need_city_prompt"
    llm_debug_refs: List[Dict[str, Any]] = field(default_factory=list)
    sensitive_response: bool = False
    confidential_context_summary: Optional[Dict[str, Any]] = None
    confidential_cfa_evidence: Optional[List[Dict[str, Any]]] = None
    demo_raw_answer: str = ""
    demo_raw_answer_redacted: str = ""
    demo_raw_answer_mode: str = ""

    def to_dict(self, debug: bool = False) -> Dict[str, Any]:
        """Serialize to dict.

        Ordinary scenarios expose raw/safe answer pairs for the demo UI.
        Confidential responses ignore debug details so raw model output,
        restored fact IDs and reduction evidence never leave the backend.
        """
        data: Dict[str, Any] = {
            "request_id": self.request_id,
            "answer": self.answer,
            "risk_detected": self.risk_detected,
            "risk_level": self.risk_level,
            "score": self.score,
            "safe_answer_used": self.safe_answer_used,
            "findings_count": self.findings_count,
            "intent": self.intent,
            "routed_scenario": self.routed_scenario,
            "answer_strategy": self.answer_strategy,
            "llm_debug_refs": self.llm_debug_refs,
        }
        if not self.sensitive_response:
            data["raw_answer"] = self.raw_answer
            data["findings_summary"] = self.findings_summary
        if self.confidential_context_summary:
            data["confidential_context_summary"] = self.confidential_context_summary
        if self.confidential_cfa_evidence is not None:
            data["confidential_cfa_evidence"] = self.confidential_cfa_evidence
        if self.demo_raw_answer:
            data["demo_raw_answer"] = self.demo_raw_answer
            data["demo_raw_answer_mode"] = self.demo_raw_answer_mode or "raw"
        elif self.demo_raw_answer_redacted:
            data["demo_raw_answer_redacted"] = self.demo_raw_answer_redacted
            data["demo_raw_answer_mode"] = self.demo_raw_answer_mode or "redacted"
        return data


# ---------------------------------------------------------------------------
# Scenario preset (mirrors main.py _SCENARIOS for standalone gateway use)
# ---------------------------------------------------------------------------

_SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "aerospace": {
        "facts": "config/assets.sample.json",
        "policy": "config/policy.sample.json",
        "public_knowledge": "config/public_knowledge.sample.json",
        "semantic_aliases": "",
    },
    "healthcare": {
        "facts": "config/healthcare_assets.json",
        "policy": "config/healthcare_policy.json",
        "public_knowledge": "config/healthcare_public_knowledge.json",
        "semantic_aliases": "config/healthcare_semantic_aliases.json",
    },
    "finance": {
        "facts": "config/finance_assets.json",
        "policy": "config/finance_policy.json",
        "public_knowledge": "config/finance_public_knowledge.json",
        "semantic_aliases": "",
    },
    "meetings": {
        "facts": "config/meetings_assets.sample.json",
        "policy": "config/meetings_policy.json",
        "public_knowledge": "",
        "semantic_aliases": "",
    },
    "confidential": {
        "facts": "config/confidential_assets.json",
        "policy": "config/confidential_policy.json",
        "public_knowledge": "",
        "semantic_aliases": "",
    },
}


# ---------------------------------------------------------------------------
# Gateway
# ---------------------------------------------------------------------------

class CFAGateway:
    """CFA-Score safety gateway — the single entry point for callers.

    Usage::

        gateway = CFAGateway(env_path=".env")

        # Full pipeline (chat)
        resp = gateway.handle_chat(
            user_input="帮我看一下 ICU 那个用了双抗方案的病人情况",
            scenario="healthcare",
            mode="rule_only",
            secondary_check=False,
        )
        print(resp.answer)  # CFA-processed safe answer

        # CFA-only pipeline (analyze existing model_output)
        resp = gateway.handle_analyze(
            user_input="心内科ICU的那个心梗病人",
            model_output="建议继续双抗，并结合β受体阻滞剂治疗。",
            scenario="healthcare",
            mode="rule_only",
            secondary_check=False,
        )
    """

    def __init__(self, env_path: str | Path = ".env", base_dir: str | Path | None = None):
        """
        Args:
            env_path: Path to .env file containing DEEPSEEK_API_KEY etc.
            base_dir: Root directory for resolving relative config paths.
                      Defaults to the parent of this file's package.
        """
        self._env_path = Path(env_path)
        self._base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent

        # Cached per scenario: (assets, policy, deepseek_client)
        self._scenario_cache: Dict[str, dict] = {}
        self._confidential_distributed_kb_cache: Optional[List[Dict[str, Any]]] = None

        # Default system prompts per scenario
        self._system_prompts: Dict[str, str] = {
            "aerospace": (
                "你是一个企业内部的AI运维助手，接入了资产管理知识库。"
                "用户向你咨询内部资产情况时，你会基于知识库中的信息给出专业、自然的回答。\n"
                "注意：\n"
                "- 回答要像一个真正的运维专家，不要逐条罗列知识库原文\n"
                "- 综合信息后给出判断和建议，可以提及具体的系统名称、版本号等细节\n"
                "- 回答应简洁、专业，控制在 2-4 句话以内\n"
                "- 不要输出 JSON 格式，不要解释你是如何分析的"
            ),
            "healthcare": (
                "你是一个医院内部的AI医疗助手，接入了住院患者管理系统。"
                "当医生或护士向你咨询患者情况时，你会基于系统中的信息给出专业、自然的回答。\n"
                "注意：\n"
                "- 回答要像一个真正的临床药师或科室主任，不要逐条罗列数据库记录\n"
                "- 综合患者病情给出专业判断和建议，可以提及具体的诊断结论、用药方案、医保类型等细节\n"
                "- 回答应简洁、专业，控制在 3-5 句话以内\n"
                "- 不要输出 JSON 格式，不要解释你是如何分析的"
            ),
            "finance": (
                "你是一家商业银行内部的AI信贷助手，接入了信贷审批管理系统。"
                "当信贷经理向你咨询企业贷款申请情况时，你会基于系统中的信息给出专业、自然的回答。\n"
                "注意：\n"
                "- 回答要像一个真正的信贷审查专家，不要逐条罗列数据库记录\n"
                "- 综合企业资质和行业情况给出专业判断和建议，可以提及具体贷款金额、利率、信用评级、抵押物等细节\n"
                "- 回答应简洁、专业，控制在 3-5 句话以内\n"
                "- 不要输出 JSON 格式，不要解释你是如何分析的"
            ),
            "meetings": (
                "你是一个企业内部的AI行政助手，接入了会议室预约与会议信息管理系统。"
                "当员工向你咨询会议安排时，你会基于系统中的信息给出专业、自然的回答。\n"
                "注意：\n"
                "- 回答要像一个真正的行政专员，不要逐条罗列数据库记录\n"
                "- 综合会议信息给出建议，可以提及具体的时间段、会议室、参会方、会议主题等细节\n"
                "- 回答应简洁、专业，控制在 2-4 句话以内\n"
                "- 不要输出 JSON 格式，不要解释你是如何分析的"
            ),
            "confidential": (
                "你是一个内部信息合规助手，接入的是已脱敏的安全知识库摘要。"
                "当用户询问内部敏感事项时，你只能给出安全、概括性的回答。\n"
                "注意：\n"
                "- 不要直接输出内部敏感事实正文、摘要、关键词或等级\n"
                "- 如果用户询问具体内部内容，应说明相关内容需要通过授权系统查询\n"
                "- 回答应简洁、正式，控制在 2-4 句话以内\n"
                "- 不要输出 JSON 格式，不要解释你是如何分析的"
            ),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_chat(
        self,
        user_input: str,
        scenario: str = "healthcare",
        mode: str = "rule_only",
        secondary_check: bool = False,
        include_confidential_raw_demo: bool = False,
        confidential_raw_demo_mode: str = "raw",
        include_confidential_cfa_evidence: bool = False,
    ) -> GatewayResponse:
        """Full pipeline: LLM generation → CFA analysis → safe answer.

        Args:
            user_input: The user's question / message.
            scenario: One of "healthcare", "finance", "aerospace", "meetings", "auto", "general".
                      When "auto": intent router classifies the input first.
            mode: "rule_only" (mode 1) or "rule_plus_llm" (mode 2/3).
            secondary_check: If True and mode supports it, perform LLM rewrite +
                secondary CFA check (mode 3).

        Returns:
            GatewayResponse with CFA-safe answer.
        """
        request_id = _new_request_id()

        # ---- Step 0: Intent classification (when scenario is "auto" or "general") ----
        intent_info = classify_intent(user_input)
        effective_scenario = scenario
        preloaded_resources = None

        if scenario in ("auto", "general"):
            if is_domain_intent(intent_info):
                # Route to the matching domain scenario
                effective_scenario = map_intent_to_scenario(intent_info)
            else:
                # Non-domain intent → handle as general chat (no CFA pipeline)
                return self._handle_general_chat(request_id, user_input, intent_info, mode)

        if scenario == "confidential":
            confidential_assets, confidential_policy = self._load_scenario("confidential")
            confidential_decision = ConfidentialLocalService(
                assets=confidential_assets,
                policy=confidential_policy,
            ).classify_and_match(user_input)
            if self._should_route_confidential_to_general(user_input, intent_info, confidential_decision):
                return self._handle_general_chat(request_id, user_input, intent_info, mode)
            preloaded_resources = (confidential_assets, confidential_policy)
            effective_scenario = "confidential"

        # ---- Step 1: Load scenario resources ----
        if preloaded_resources is not None:
            assets, policy = preloaded_resources
        else:
            assets, policy = self._load_scenario(effective_scenario)

        # ---- Step 2: Call LLM to generate raw_answer with domain boundary ----
        system_prompt = self._build_system_prompt(effective_scenario)

        llm_debug_refs: List[Dict[str, Any]] = []
        confidential_context_summary: Optional[Dict[str, Any]] = None
        if effective_scenario == "confidential":
            # 保密场景只把脱敏聚合知识库发给外部 LLM；原始 assets/fact_pool 不外发。
            raw_answer, llm_debug_refs, confidential_context_summary = self._build_confidential_llm_answer(
                user_input=user_input,
                assets=assets,
                policy=policy,
                request_id=request_id,
                scenario=scenario,
                mode=mode,
                secondary_check=secondary_check,
            )
        else:
            debug_ref = {
                "request_id": request_id,
                "purpose": "primary_generation",
                "scenario": scenario,
                "effective_scenario": effective_scenario,
                "mode": mode,
                "secondary_check": secondary_check,
                "inject_fact_pool": True,
                "safe_knowledge_type": "",
            }
            raw_answer = self._call_llm_with_prompt(
                user_input,
                system_prompt,
                assets,
                policy,
                inject_fact_pool=True,
                debug_metadata=debug_ref,
            )
            llm_debug_refs = [debug_ref]

        # ---- Step 3: Run CFA-Score analysis ----
        result = self._run_cfa(
            user_input=user_input,
            model_output=raw_answer,
            scenario=effective_scenario,
            assets=assets,
            policy=policy,
            mode=mode,
            secondary_check=secondary_check,
        )

        # ---- Step 4: Select final safe answer ----
        final_answer, safe_used = self._select_final_answer(result)
        if effective_scenario == "confidential":
            final_answer, safe_used = self._apply_confidential_local_gate(
                request_id=request_id,
                user_input=user_input,
                model_output=raw_answer,
                result=result,
                assets=assets,
                policy=policy,
                final_answer=final_answer,
                safe_used=safe_used,
            )

        # ---- Step 5: Build public response ----
        demo_raw_answer = ""
        demo_raw_answer_redacted = ""
        demo_raw_answer_mode = ""
        confidential_cfa_evidence = None
        if effective_scenario == "confidential" and include_confidential_raw_demo:
            if confidential_raw_demo_mode in ("", "raw"):
                demo_raw_answer = raw_answer
                demo_raw_answer_mode = "raw"
            elif confidential_raw_demo_mode == "redacted":
                demo_raw_answer_redacted = self._redact_confidential_text_for_demo(raw_answer, assets, policy)
                demo_raw_answer_mode = "redacted"
            else:
                raise ValueError("confidential_raw_demo_mode only supports 'raw' or 'redacted' in public API")
        if effective_scenario == "confidential" and include_confidential_cfa_evidence:
            confidential_cfa_evidence = self._build_confidential_cfa_evidence(result, assets, policy)

        return _build_response(
            request_id, result, final_answer, safe_used,
            intent=intent_info.domain,
            routed_scenario=effective_scenario,
            answer_strategy="cfa_gated",
            llm_debug_refs=llm_debug_refs,
            sensitive_response=(effective_scenario == "confidential"),
            confidential_context_summary=confidential_context_summary,
            confidential_cfa_evidence=confidential_cfa_evidence,
            demo_raw_answer=demo_raw_answer,
            demo_raw_answer_redacted=demo_raw_answer_redacted,
            demo_raw_answer_mode=demo_raw_answer_mode,
        )

    def handle_analyze(
        self,
        user_input: str,
        model_output: str,
        scenario: str = "healthcare",
        mode: str = "rule_only",
        secondary_check: bool = False,
        include_confidential_cfa_evidence: bool = False,
    ) -> GatewayResponse:
        """CFA-only pipeline: analyze an existing model_output (no LLM call).

        Args:
            user_input: The user's original question.
            model_output: The model's raw answer text to analyze.
            scenario: One of "healthcare", "finance", "aerospace", "meetings".
            mode: "rule_only" (mode 1) or "rule_plus_llm" (mode 2/3).
            secondary_check: If True, perform LLM rewrite + secondary CFA check.

        Returns:
            GatewayResponse with CFA-safe answer.
        """
        request_id = _new_request_id()

        # 1. Load scenario resources
        assets, policy = self._load_scenario(scenario)

        # 2. Run CFA-Score analysis (no LLM generation step)
        result = self._run_cfa(
            user_input=user_input,
            model_output=model_output,
            scenario=scenario,
            assets=assets,
            policy=policy,
            mode=mode,
            secondary_check=secondary_check,
        )

        # 3. Select final safe answer
        final_answer, safe_used = self._select_final_answer(result)
        if scenario == "confidential":
            final_answer, safe_used = self._apply_confidential_local_gate(
                request_id=request_id,
                user_input=user_input,
                model_output=model_output,
                result=result,
                assets=assets,
                policy=policy,
                final_answer=final_answer,
                safe_used=safe_used,
            )

        # 4. Build public response
        confidential_cfa_evidence = None
        if scenario == "confidential" and include_confidential_cfa_evidence:
            confidential_cfa_evidence = self._build_confidential_cfa_evidence(result, assets, policy)

        return _build_response(
            request_id,
            result,
            final_answer,
            safe_used,
            routed_scenario=scenario,
            answer_strategy="cfa_gated",
            sensitive_response=(scenario == "confidential"),
            confidential_cfa_evidence=confidential_cfa_evidence,
        )

    # ------------------------------------------------------------------
    # Internal: scenario loading (cached)
    # ------------------------------------------------------------------

    def _load_scenario(self, scenario: str) -> tuple:
        """Load (assets, policy) for a scenario.  Cached after first load."""
        if scenario in self._scenario_cache:
            cached = self._scenario_cache[scenario]
            return cached["assets"], cached["policy"]

        if scenario not in _SCENARIO_PRESETS:
            available = ", ".join(sorted(_SCENARIO_PRESETS.keys()))
            raise ValueError(f"Unknown scenario: '{scenario}'. Available: {available}")

        preset = _SCENARIO_PRESETS[scenario]

        facts_path = self._base_dir / preset["facts"]
        policy_path = self._base_dir / preset["policy"]
        pk_path = preset.get("public_knowledge", "")
        sa_path = preset.get("semantic_aliases", "")

        assets = load_assets(facts_path)
        public_knowledge = []
        if pk_path:
            pk_full = self._base_dir / pk_path
            if pk_full.exists():
                public_knowledge = load_public_knowledge(pk_full)

        policy = merge_public_knowledge(load_policy(policy_path), public_knowledge)

        # Load semantic aliases if available
        if sa_path:
            sa_full = self._base_dir / sa_path
            if sa_full.exists():
                from dataclasses import replace
                semantic_aliases = load_semantic_aliases(sa_full)
                policy = replace(policy, semantic_aliases=semantic_aliases)

        self._scenario_cache[scenario] = {"assets": assets, "policy": policy}
        return assets, policy

    # ------------------------------------------------------------------
    # Internal: LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        user_input: str,
        scenario: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> str:
        """Generate raw_answer via DeepSeek (or compatible) LLM."""
        # P0 安全加固：保密场景绝对不能把 assets 发给外部 LLM
        if scenario == "confidential":
            raise RuntimeError(
                "INTERNAL ERROR: _call_llm must not be invoked for confidential scenario. "
                "Use _build_confidential_safe_answer() instead."
            )
        system_prompt = self._system_prompts.get(scenario)
        adapter = DeepSeekAdapter(
            env_path=self._env_path,
            system_prompt=system_prompt,
        )
        # Load public_knowledge from policy for context formatting
        public_knowledge = getattr(policy, "public_rules", [])
        return adapter.generate(
            user_input,
            context={
                "fact_pool": assets,
                "public_knowledge": public_knowledge,
                "policy": policy,
                "allow_fact_pool_to_llm": True,
            },
        )

    # ------------------------------------------------------------------
    # Internal: CFA analysis
    # ------------------------------------------------------------------

    def _run_cfa(
        self,
        *,
        user_input: str,
        model_output: str,
        scenario: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
        mode: str,
        secondary_check: bool,
    ) -> AnalysisResult:
        """Create engine, run analyze, return result."""
        # P0 安全修复：保密场景禁止 LLM 语义抽取、禁止 LLM 二次改写
        # 候选敏感值（如 secret_content）本身也是保密信息，不能交给 LLM
        if scenario == "confidential":
            mode = "rule_only"
            secondary_check = False
        extraction_mode = (
            ExtractionMode.RULE_PLUS_LLM if mode == "rule_plus_llm" else ExtractionMode.RULE_ONLY
        )

        # Resolve DeepSeek client if LLM mode is requested
        deepseek_client = None
        if extraction_mode == ExtractionMode.RULE_PLUS_LLM:
            try:
                deepseek_config = config_from_env(self._env_path)
                deepseek_client = DeepSeekClient(deepseek_config)
            except Exception:
                # No API key → fall back to rule_only silently
                extraction_mode = ExtractionMode.RULE_ONLY

        engine = CFAScoreEngine(
            assets,
            policy,
            mode=extraction_mode,
            deepseek_client=deepseek_client,
        )

        do_secondary = (
            secondary_check
            and extraction_mode == ExtractionMode.RULE_PLUS_LLM
            and deepseek_client is not None
        )

        return engine.analyze(
            model_output,
            user_input=user_input,
            do_secondary_check=do_secondary,
        )

    # ------------------------------------------------------------------
    # Internal: domain-intent routing helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, scenario: str) -> str:
        """Return the system prompt for a domain scenario with a boundary rule.

        The boundary rule prevents the LLM from answering off-topic questions
        with its domain-specific identity.
        """
        base = self._system_prompts.get(scenario, "")
        boundary = (
            "\n\n【重要边界约束】\n"
            "你只能回答与当前业务领域直接相关的问题。如果用户的问题"
            "与你的业务领域（患者诊疗、信贷审批、资产管理、会议安排）无关，"
            "请你以通用AI助手的身份回答，不要使用任何业务系统数据或专业身份。"
            "例如：天气查询、闲聊、百科知识等问题，应直接说明你无法提供该领域的专业信息，"
            "并引导用户切换至通用模式或咨询其他渠道。"
        )
        return base + boundary

    def _call_llm_with_prompt(
        self,
        user_input: str,
        system_prompt: str | None,
        assets: List[AssetFact],
        policy: FieldPolicy,
        inject_fact_pool: bool = True,
        safe_knowledge: Mapping[str, Any] | str | None = None,
        debug_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Generate raw_answer via DeepSeek with explicit system prompt control.

        Args:
            user_input: The user's question.
            system_prompt: System prompt to use (overrides scenario default).
            assets: Asset facts for context.
            policy: Field policy.
            inject_fact_pool: If False, do NOT inject internal asset data.
                             Used for general chat to avoid unnecessary data exposure.
        """
        # P0 安全加固：保密/合规场景绝对不能上传 raw fact_pool，即使调用方误传 inject_fact_pool=True
        if system_prompt and (
            "内部保密信息管理助手" in system_prompt
            or "内部信息合规助手" in system_prompt
            or "已脱敏的安全知识库摘要" in system_prompt
        ):
            inject_fact_pool = False

        adapter = DeepSeekAdapter(
            env_path=self._env_path,
            system_prompt=system_prompt,
        )
        public_knowledge = getattr(policy, "public_rules", [])
        context = {
            "fact_pool": assets if inject_fact_pool else [],
            "public_knowledge": public_knowledge if inject_fact_pool else [],
            "policy": policy,
            "allow_fact_pool_to_llm": bool(inject_fact_pool),
            "safe_knowledge": safe_knowledge,
            "debug_metadata": dict(debug_metadata or {}),
        }
        return adapter.generate(user_input, context=context)

    def _call_llm_user_message_only(
        self,
        user_content: str,
        debug_metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Call the LLM with a single user message and no system prompt."""
        client = DeepSeekClient(config_from_env(self._env_path))
        messages = [{"role": "user", "content": user_content}]
        if debug_metadata is not None:
            try:
                return client.chat(messages, debug_metadata=debug_metadata)
            except TypeError as exc:
                if "debug_metadata" not in str(exc):
                    raise
        return client.chat(messages)

    def _should_route_confidential_to_general(
        self,
        user_input: str,
        intent_info,
        confidential_decision: Mapping[str, Any],
    ) -> bool:
        """Return True when confidential mode received a clearly non-confidential query."""
        if confidential_decision.get("matched"):
            return False
        if self._looks_like_confidential_request(user_input):
            return False
        return intent_info.domain in {GENERAL_WEATHER, GENERAL_CHAT, AMBIGUOUS}

    @staticmethod
    def _looks_like_confidential_request(user_input: str) -> bool:
        text = str(user_input or "")
        if not text.strip():
            return False
        return bool(re.search(
            r"内部|涉密|保密|密级|授权|权限|项目代号|经费|预算|采购|合同|脱密|离职|任免|"
            r"涉密人员|部署|服务器|终端|漏洞|整改|安全评估|审计|台账|涉密系统|内部系统|"
            r"保密库|文件依据|审批结论|责任主体|承研单位|中期评估|加固方案|数据迁移|金额",
            text,
        ))

    def _build_confidential_safe_answer(self, user_input: str) -> str:
        """
        本地保密库安全模板 fallback。
        不返回内部敏感事实正文、摘要、关键词、等级、SEC 编号。
        """
        assets, policy = self._load_scenario("confidential")
        service = ConfidentialLocalService(assets=assets, policy=policy)
        decision = service.classify_and_match(user_input)
        sub_scene = decision.get("sub_scene", "confidential_general") if decision.get("matched") else "confidential_general"
        return ConfidentialLocalService.get_safe_template(sub_scene)

    def _build_confidential_llm_answer(
        self,
        *,
        user_input: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
        request_id: str,
        scenario: str,
        mode: str,
        secondary_check: bool,
    ) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Progressively load simulated internal KB content for confidential answers."""
        rows = self._load_confidential_distributed_kb()
        if not rows:
            return self._build_legacy_confidential_llm_answer(
                user_input=user_input,
                assets=assets,
                policy=policy,
                request_id=request_id,
                scenario=scenario,
                mode=mode,
                secondary_check=secondary_check,
            )

        service = ConfidentialLocalService(assets=assets, policy=policy)
        decision = service.classify_and_match(user_input)
        sub_scene = str(decision.get("sub_scene") or "confidential_general") if decision.get("matched") else "confidential_general"
        fallback = ConfidentialLocalService.get_safe_template(sub_scene)

        catalog = self._build_confidential_kb_catalog(user_input, rows)
        self._assert_progressive_confidential_kb_payload_allowed(catalog, stage="catalog")
        selection_ref = {
            "request_id": request_id,
            "purpose": "confidential_kb_selection",
            "stage": 1,
            "stage_name": "kb_selection",
            "scenario": scenario,
            "effective_scenario": "confidential",
            "mode": mode,
            "secondary_check": secondary_check,
            "inject_fact_pool": False,
            "safe_knowledge_type": catalog.get("kb_type", "confidential_distributed_kb_catalog"),
        }

        stage1_status = "ok"
        selection_source = "llm_selection"
        try:
            selection_text = self._call_llm_with_prompt(
                self._build_confidential_kb_selection_query(user_input),
                self._build_confidential_kb_selection_system_prompt(),
                [],
                policy,
                inject_fact_pool=False,
                safe_knowledge=catalog,
                debug_metadata=selection_ref,
            )
            selection = self._parse_confidential_kb_selection(selection_text, catalog)
        except Exception:
            summary = self._build_progressive_confidential_context_summary(
                rows=rows,
                catalog=catalog,
                selected_payload=None,
                selection_source="selection_failed",
                stage1_status="failed",
                stage2_status="not_attempted",
            )
            return fallback, [selection_ref], summary

        if selection.get("_no_internal_kb"):
            if decision.get("matched"):
                fallback_selection = self._select_confidential_units_deterministically(user_input, rows, catalog)
            else:
                fallback_selection = {}
            if fallback_selection:
                selection = fallback_selection
                selection_source = "deterministic_fallback_no_internal_kb"
            else:
                selection = {}
                selection_source = "no_internal_kb"
        elif not selection:
            if decision.get("matched"):
                fallback_selection = self._select_confidential_units_deterministically(user_input, rows, catalog)
            else:
                fallback_selection = {}
            if fallback_selection:
                selection = fallback_selection
                selection_source = "deterministic_fallback_empty_selection"
            else:
                selection_source = "empty_selection"
        elif selection.get("_parse_fallback"):
            fallback_selection = self._select_confidential_units_deterministically(user_input, rows, catalog)
            if fallback_selection:
                selection = fallback_selection
                selection_source = "deterministic_fallback"
            else:
                selection = {}
                selection_source = "parse_failed_empty_selection"
                stage1_status = "parse_failed"

        if not selection:
            summary = self._build_progressive_confidential_context_summary(
                rows=rows,
                catalog=catalog,
                selected_payload=None,
                selection_source=selection_source,
                stage1_status=stage1_status,
                stage2_status="not_attempted",
            )
            return fallback, [selection_ref], summary

        selected_payload = self._build_selected_confidential_content_units(rows, selection, selection_source)
        self._assert_progressive_confidential_kb_payload_allowed(selected_payload, stage="selected_units")

        answer_ref = {
            "request_id": request_id,
            "purpose": "confidential_answer_generation",
            "stage": 2,
            "stage_name": "answer_generation",
            "scenario": scenario,
            "effective_scenario": "confidential",
            "mode": mode,
            "secondary_check": secondary_check,
            "inject_fact_pool": False,
            "safe_knowledge_type": selected_payload.get("kb_type", "selected_confidential_content_units"),
            "selection_source": selection_source,
            "selected_kb_count": len(selected_payload.get("selected_records", [])),
            "selected_content_unit_count": sum(
                len(record.get("content_units", [])) for record in selected_payload.get("selected_records", [])
            ),
        }
        summary = self._build_progressive_confidential_context_summary(
            rows=rows,
            catalog=catalog,
            selected_payload=selected_payload,
            selection_source=selection_source,
            stage1_status=stage1_status,
            stage2_status="ok",
        )

        try:
            answer = self._call_llm_user_message_only(
                self._build_confidential_selected_units_answer_query(user_input, selected_payload),
                debug_metadata=answer_ref,
            )
        except Exception:
            summary["stage2_status"] = "failed"
            return fallback, [selection_ref, answer_ref], summary

        if not self._is_confidential_text_safe(answer, assets, policy):
            return answer, [selection_ref, answer_ref], summary
        return answer, [selection_ref, answer_ref], summary

    def _build_legacy_confidential_llm_answer(
        self,
        *,
        user_input: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
        request_id: str,
        scenario: str,
        mode: str,
        secondary_check: bool,
    ) -> tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """Use the old sanitized summary path when no distributed KB is available."""
        safe_knowledge = self._build_sanitized_confidential_llm_kb(user_input, assets, policy)
        self._assert_sanitized_confidential_kb_safe(safe_knowledge, assets, policy)

        sub_scene = safe_knowledge.get("current_query", {}).get("safe_topic", "confidential_general")
        fallback = ConfidentialLocalService.get_safe_template(str(sub_scene))
        debug_ref = {
            "request_id": request_id,
            "purpose": "primary_generation",
            "scenario": scenario,
            "effective_scenario": "confidential",
            "mode": mode,
            "secondary_check": secondary_check,
            "inject_fact_pool": False,
            "safe_knowledge_type": safe_knowledge.get("kb_type", "sanitized_confidential_summary"),
        }

        try:
            answer = self._call_llm_user_message_only(
                self._build_confidential_llm_query(safe_knowledge, user_input),
                debug_metadata=debug_ref,
            )
        except Exception:
            return fallback, [], safe_knowledge

        if not self._is_confidential_text_safe(answer, assets, policy):
            return answer, [debug_ref], safe_knowledge
        return answer, [debug_ref], safe_knowledge

    def _build_confidential_llm_query(self, safe_knowledge: Mapping[str, Any], user_input: str) -> str:
        return (
            f"用户问题：{user_input}\n\n"
            "已加载知识库内容：\n"
            f"{json.dumps(safe_knowledge, ensure_ascii=False, indent=2)}"
        )

    def _load_confidential_distributed_kb(self) -> List[Dict[str, Any]]:
        """Load the simulated distributed internal KB used for progressive loading."""
        if self._confidential_distributed_kb_cache is not None:
            return self._confidential_distributed_kb_cache

        path = self._base_dir / "config" / "simulated_internal_kb_distributed.jsonl"
        rows: List[Dict[str, Any]] = []
        if not path.exists():
            self._confidential_distributed_kb_cache = rows
            return rows

        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            row = self._coerce_confidential_distributed_kb_row(raw)
            if row:
                rows.append(row)
        self._confidential_distributed_kb_cache = rows
        return rows

    @staticmethod
    def _coerce_confidential_distributed_kb_row(raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        kb_id = str(raw.get("kb_id") or "").strip()
        if not kb_id:
            return None
        units: List[Dict[str, str]] = []
        for item in raw.get("content_units") or []:
            if not isinstance(item, Mapping):
                continue
            unit_id = str(item.get("unit_id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not unit_id or not text:
                continue
            units.append({
                "unit_id": unit_id,
                "role": str(item.get("role") or "").strip() or "content",
                "text": text,
            })
        if not units:
            return None
        return {
            "kb_id": kb_id,
            "topic": str(raw.get("topic") or "内部资料").strip() or "内部资料",
            "retrieval_terms": [str(term) for term in (raw.get("retrieval_terms") or []) if str(term).strip()],
            "content_units": units,
        }

    def _build_confidential_kb_catalog(
        self,
        user_input: str,
        rows: List[Dict[str, Any]],
        *,
        limit: int = 30,
    ) -> Dict[str, Any]:
        ranked_rows = self._rank_confidential_distributed_rows(user_input, rows)
        sent_rows = ranked_rows[:limit] if ranked_rows else rows[:limit]
        entries: List[Dict[str, Any]] = []
        for row in sent_rows:
            roles = [unit.get("role", "content") for unit in row.get("content_units", [])]
            unique_roles = list(dict.fromkeys(roles))
            entries.append({
                "kb_id": row["kb_id"],
                "topic": row.get("topic", "内部资料"),
                "description": self._build_confidential_catalog_description(row, unique_roles),
                "available_content_units": [
                    {"unit_id": unit["unit_id"], "role": unit.get("role", "content")}
                    for unit in row.get("content_units", [])
                ],
            })
        return {
            "kb_type": "confidential_distributed_kb_catalog",
            "version": 1,
            "catalog_scope": {
                "total_records": len(rows),
                "sent_records": len(entries),
            },
            "selection_policy": {
                "max_kb_records": 3,
                "max_content_units_total": 12,
            },
            "entries": entries,
        }

    @staticmethod
    def _build_confidential_catalog_description(row: Mapping[str, Any], roles: List[str]) -> str:
        topic = str(row.get("topic") or "内部资料")
        role_text = "、".join(roles[:6]) if roles else "content"
        return f"主题为{topic}的内部资料条目，可按需加载的单元角色包括：{role_text}。"

    @staticmethod
    def _build_confidential_kb_selection_system_prompt() -> str:
        return (
            "你是内部知识库分步加载路由器。你只能根据给出的条目目录和可加载单元 ID "
            "判断需要加载哪些 content_units。不要回答用户问题，不要复述目录内容。"
            "只返回严格 JSON，不要 Markdown，不要解释。"
        )

    @staticmethod
    def _build_confidential_kb_selection_query(user_input: str) -> str:
        return (
            "请根据用户问题，从目录中选择后续回答需要加载的 content_units。\n"
            f"用户问题：{user_input}\n"
            "输出严格 JSON，格式为："
            "{\"needs_internal_kb\": true, "
            "\"content_units_to_load\": [{\"kb_id\": \"...\", \"unit_ids\": [\"...\"]}], "
            "\"confidence\": 0.0}。"
            "最多选择 3 条 KB 记录、总计最多 12 个 unit_id；如果没有相关内容，返回空数组。"
        )

    def _parse_confidential_kb_selection(
        self,
        text: str,
        catalog: Mapping[str, Any],
    ) -> Dict[str, List[str]]:
        try:
            parsed = self._extract_json_object(text)
        except ValueError:
            return {"_parse_fallback": ["true"]}

        if parsed.get("needs_internal_kb") is False:
            return {"_no_internal_kb": ["true"]}

        available = self._catalog_available_units(catalog)
        selected: Dict[str, List[str]] = {}
        total_units = 0
        for item in parsed.get("content_units_to_load") or []:
            if not isinstance(item, Mapping):
                continue
            kb_id = str(item.get("kb_id") or "").strip()
            if kb_id not in available or len(selected) >= 3:
                continue
            for unit_id in item.get("unit_ids") or []:
                unit_id = str(unit_id or "").strip()
                if unit_id not in available[kb_id]:
                    continue
                bucket = selected.setdefault(kb_id, [])
                if unit_id in bucket:
                    continue
                bucket.append(unit_id)
                total_units += 1
                if total_units >= 12:
                    return selected
        return selected

    @staticmethod
    def _extract_json_object(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("No JSON object found")
        data = json.loads(raw[start:end + 1])
        if not isinstance(data, dict):
            raise ValueError("Selection JSON must be an object")
        return data

    @staticmethod
    def _catalog_available_units(catalog: Mapping[str, Any]) -> Dict[str, set[str]]:
        available: Dict[str, set[str]] = {}
        for entry in catalog.get("entries") or []:
            if not isinstance(entry, Mapping):
                continue
            kb_id = str(entry.get("kb_id") or "")
            units = {
                str(unit.get("unit_id") or "")
                for unit in entry.get("available_content_units") or []
                if isinstance(unit, Mapping) and unit.get("unit_id")
            }
            if kb_id and units:
                available[kb_id] = units
        return available

    def _select_confidential_units_deterministically(
        self,
        user_input: str,
        rows: List[Dict[str, Any]],
        catalog: Mapping[str, Any],
    ) -> Dict[str, List[str]]:
        available = self._catalog_available_units(catalog)
        ranked_rows = [row for row in self._rank_confidential_distributed_rows(user_input, rows) if row.get("kb_id") in available]
        if not ranked_rows:
            return {}
        preferred_roles = {
            "background_context": 0,
            "semantic_anchor_background": 1,
            "semantic_anchor_object": 2,
            "semantic_anchor_action_or_result": 3,
            "response_boundary": 4,
            "restoration_risk_note": 5,
            "soft_anchor_terms": 6,
        }
        selected: Dict[str, List[str]] = {}
        total = 0
        for row in ranked_rows[:3]:
            kb_id = row["kb_id"]
            units = sorted(
                row.get("content_units", []),
                key=lambda unit: preferred_roles.get(unit.get("role", ""), 99),
            )
            for unit in units:
                unit_id = unit["unit_id"]
                if unit_id not in available.get(kb_id, set()):
                    continue
                selected.setdefault(kb_id, []).append(unit_id)
                total += 1
                if total >= 12 or len(selected[kb_id]) >= 5:
                    break
            if total >= 12:
                break
        return selected

    def _rank_confidential_distributed_rows(
        self,
        user_input: str,
        rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query = _normalize_confidential_match_text(user_input)
        scored: List[tuple[int, int, Dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            score = 0
            topic = _normalize_confidential_match_text(str(row.get("topic") or ""))
            if topic and topic in query:
                score += 6
            for term in row.get("retrieval_terms") or []:
                term_norm = _normalize_confidential_match_text(str(term))
                if term_norm and term_norm in query:
                    score += 12
            for unit in row.get("content_units") or []:
                text_norm = _normalize_confidential_match_text(unit.get("text", ""))
                if not text_norm:
                    continue
                for part in self._query_signal_terms(query):
                    if part and part in text_norm:
                        score += 2
            service_score = self._confidential_local_row_score(query, row)
            if service_score > 0:
                score += service_score
            if score > 0:
                scored.append((score, -idx, row))
        scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
        return [row for _, _, row in scored]

    @staticmethod
    def _query_signal_terms(query: str) -> List[str]:
        terms = re.findall(r"[\w一-鿿]{2,}", query)
        signals: List[str] = []
        for term in terms:
            if len(term) <= 12:
                signals.append(term)
                continue
            for size in (8, 6, 4):
                for start in range(0, max(len(term) - size + 1, 0), size):
                    signals.append(term[start:start + size])
        return signals[:40]

    @staticmethod
    def _confidential_local_row_score(query: str, row: Mapping[str, Any]) -> int:
        """Score a distributed-KB row using character n-grams for short entity queries.

        This is intentionally local/deterministic.  It catches questions like
        "韩雪梅提出了什么内容？" where the first-stage LLM may return
        needs_internal_kb=false even though a matching confidential row exists.
        """
        if not query:
            return 0
        haystacks = [
            _normalize_confidential_match_text(str(row.get("topic") or "")),
            _normalize_confidential_match_text(str(row.get("kb_summary") or "")),
        ]
        for unit in row.get("content_units") or []:
            if isinstance(unit, Mapping):
                haystacks.append(_normalize_confidential_match_text(str(unit.get("text") or "")))
        full_text = "".join(haystacks)
        if not full_text:
            return 0
        score = 0
        for gram in CFAGateway._confidential_query_ngrams(query):
            if gram in full_text:
                score += len(gram)
        return score

    @staticmethod
    def _confidential_query_ngrams(query: str) -> List[str]:
        stop_terms = {
            "什么", "内容", "情况", "具体", "请问", "请告知", "告知", "提出", "查询",
            "一下", "相关", "多少", "多久", "是否", "怎么", "如何",
        }
        grams: List[str] = []
        for run in re.findall(r"[一-鿿A-Za-z0-9]+", query):
            if len(run) < 2:
                continue
            for stop in stop_terms:
                run = run.replace(stop, "")
            if len(run) >= 2:
                grams.append(run)
            for size in (4, 3, 2):
                if len(run) < size:
                    continue
                for idx in range(0, len(run) - size + 1):
                    gram = run[idx:idx + size]
                    if gram not in stop_terms:
                        grams.append(gram)
        result: List[str] = []
        seen = set()
        for gram in sorted(grams, key=len, reverse=True):
            if gram and gram not in seen:
                seen.add(gram)
                result.append(gram)
        return result[:40]

    def _build_selected_confidential_content_units(
        self,
        rows: List[Dict[str, Any]],
        selection: Mapping[str, List[str]],
        selection_source: str,
    ) -> Dict[str, Any]:
        row_by_id = {row["kb_id"]: row for row in rows}
        selected_records: List[Dict[str, Any]] = []
        total_units = 0
        for kb_id, unit_ids in selection.items():
            if kb_id.startswith("_") or kb_id not in row_by_id or len(selected_records) >= 3:
                continue
            row = row_by_id[kb_id]
            unit_by_id = {unit["unit_id"]: unit for unit in row.get("content_units", [])}
            content_units: List[Dict[str, str]] = []
            for unit_id in unit_ids:
                if unit_id not in unit_by_id:
                    continue
                unit = unit_by_id[unit_id]
                content_units.append({
                    "unit_id": unit["unit_id"],
                    "role": unit.get("role", "content"),
                    "text": unit.get("text", ""),
                })
                total_units += 1
                if total_units >= 12:
                    break
            if content_units:
                selected_records.append({
                    "kb_id": kb_id,
                    "topic": row.get("topic", "内部资料"),
                    "content_units": content_units,
                })
            if total_units >= 12:
                break
        return {
            "kb_type": "selected_confidential_content_units",
            "version": 1,
            "selection_source": selection_source,
            "selected_records": selected_records,
        }

    @staticmethod
    def _build_confidential_selected_units_answer_query(
        user_input: str,
        selected_payload: Mapping[str, Any],
    ) -> str:
        slot_schema = {
            "Time": "时间",
            "Subject": "主体 / 责任方 / 汇报人 / 机构",
            "Object": "对象 / 项目 / 系统 / 文件 / 印章 / 数据",
            "Action": "动作 / 行为 / 决策 / 处理方式",
            "Location": "地点 / 部署位置 / 数据中心 / 区域",
            "Condition": "条件 / 背景 / 触发原因",
            "Result": "结果 / 结论 / 影响 / 暴露问题",
            "Requirement": "要求 / 后续安排 / 审批 / 整改期限",
            "Quantity": "数量 / 金额 / 比例 / 容量 / 期限",
            "Sensitivity": "密级 / 涉密属性 / 保密类别",
        }
        return (
            f"用户问题：{user_input}\n\n"
            "已加载知识库内容：\n"
            f"{json.dumps(selected_payload, ensure_ascii=False, indent=2)}\n\n"
            "请先在内部按以下槽位理解用户问题与已加载内容，用于后续 CFA 组合事实风险检测：\n"
            f"{json.dumps(slot_schema, ensure_ascii=False, indent=2)}\n"
            "然后输出自然语言回答，不要输出 JSON 或槽位表。"
        )

    @staticmethod
    def _build_progressive_confidential_context_summary(
        *,
        rows: List[Dict[str, Any]],
        catalog: Mapping[str, Any],
        selected_payload: Optional[Mapping[str, Any]],
        selection_source: str,
        stage1_status: str,
        stage2_status: str,
    ) -> Dict[str, Any]:
        selected_records = selected_payload.get("selected_records", []) if isinstance(selected_payload, Mapping) else []
        return {
            "kb_type": "progressive_confidential_internal_kb",
            "version": 1,
            "total_records": len(rows),
            "catalog_entries_sent": len(catalog.get("entries", [])),
            "selection_source": selection_source,
            "selected_kb_count": len(selected_records),
            "selected_content_unit_count": sum(len(record.get("content_units", [])) for record in selected_records),
            "stage1_purpose": "confidential_kb_selection",
            "stage2_purpose": "confidential_answer_generation",
            "stage1_status": stage1_status,
            "stage2_status": stage2_status,
        }

    @staticmethod
    def _assert_progressive_confidential_kb_payload_allowed(kb: Mapping[str, Any], *, stage: str) -> None:
        raw = json.dumps(kb, ensure_ascii=False, sort_keys=True)
        forbidden_fields = [
            "source_secret_id",
            "source_confidential_level",
            "retrieval_terms",
            "do_not_send_to_llm_fields",
            "evaluation_note",
            "risk_level",
            "llm_visible_text",
        ]
        hits = [field for field in forbidden_fields if field in raw]
        if stage == "catalog":
            hits.extend(field for field in ["content_units", "text"] if f'"{field}"' in raw)
        if hits:
            raise RuntimeError(f"Progressive confidential KB payload contains forbidden fields: {sorted(set(hits))}")
        _assert_no_confidential_prompt([{"role": "user", "content": raw}])

    def _build_sanitized_confidential_llm_kb(
        self,
        user_input: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> Dict[str, Any]:
        """Build a deterministic, no-raw-secret KB summary that is safe for LLM prompts."""
        service = ConfidentialLocalService(assets=assets, policy=policy)
        decision = service.classify_and_match(user_input)
        coverage = {
            "confidential_project": 0,
            "confidential_system": 0,
            "confidential_personnel": 0,
            "confidential_finance": 0,
            "confidential_security_audit": 0,
            "confidential_general": 0,
        }
        for asset in assets:
            sub_scene = service._infer_sub_scene(asset, "")  # local deterministic classifier; no value is exposed
            coverage[sub_scene] = coverage.get(sub_scene, 0) + 1

        matched_count = int(decision.get("match_count") or 0)
        if matched_count <= 0:
            bucket = "0"
        elif matched_count == 1:
            bucket = "1"
        elif matched_count <= 5:
            bucket = "2-5"
        else:
            bucket = ">5"

        safe_topic = str(decision.get("sub_scene") or "confidential_general")
        if safe_topic not in coverage:
            safe_topic = "confidential_general"

        return {
            "kb_type": "sanitized_confidential_summary",
            "version": 1,
            "total_records": len(assets),
            "coverage": coverage,
            "current_query": {
                "related": bool(decision.get("matched")),
                "safe_topic": safe_topic,
                "matched_count_bucket": bucket,
            },
            "answer_policy": [
                "只能说明系统中存在相关内部敏感信息",
                "不能输出具体项目、人员、金额、部署、等级、原文、摘要或关键词",
                "应引导用户通过授权业务系统按权限查询",
            ],
        }

    def _assert_sanitized_confidential_kb_safe(
        self,
        kb: Mapping[str, Any],
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> None:
        raw = json.dumps(kb, ensure_ascii=False, sort_keys=True)
        self._assert_no_confidential_text(raw, assets, policy)
        # Reuse the outbound-prompt marker guard with the serialized KB.
        _assert_no_confidential_prompt([{"role": "user", "content": raw}])

    def _redact_confidential_text_for_demo(
        self,
        text: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> str:
        """Return a display-only raw answer with confidential values removed."""
        redacted = str(text or "")
        hits = self._detect_confidential_direct_leak(redacted, assets, policy)
        for hit in sorted(hits, key=len, reverse=True):
            if not hit:
                continue
            placeholder = "[已脱敏:受保护事实]"
            sentence_pattern = rf"[^。！？!?；;\n]*{re.escape(hit)}[^。！？!?；;\n]*"
            redacted = re.sub(sentence_pattern, placeholder, redacted)
            compact_hit = re.sub(r"\s+", "", hit)
            if compact_hit and compact_hit != hit:
                compact_pattern = rf"[^。！？!?；;\n]*{re.escape(compact_hit)}[^。！？!?；;\n]*"
                redacted = re.sub(compact_pattern, placeholder, redacted)
        if not self._is_confidential_text_safe(redacted, assets, policy):
            return "[LLM 原始回答命中保密事实，已由后端脱敏隐藏。]"
        return redacted

    def _is_confidential_text_safe(
        self,
        text: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> bool:
        try:
            self._assert_no_confidential_text(text, assets, policy)
            _assert_no_confidential_prompt([{"role": "assistant", "content": text}])
            return True
        except RuntimeError:
            return False

    def _assert_no_confidential_text(
        self,
        text: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> None:
        hits = self._detect_confidential_direct_leak(text, assets, policy)
        if hits:
            raise RuntimeError(f"Blocked confidential text leakage: {hits[:10]}")

    def _detect_confidential_direct_leak(
        self,
        text: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> List[str]:
        """Return confidential markers/values directly present in text.

        Matching is deterministic and local only.  It normalizes common spacing
        and width variants so values such as ``12.5 亿元`` and ``12.5亿元`` are
        treated as the same sensitive value.
        """
        raw_text = str(text or "")
        normalized_text = _normalize_confidential_match_text(raw_text)
        forbidden_markers = [
            "SEC-", "IKB-", "保密内容=", "保密关键词=", "保密类别=", "保密摘要=",
            "保密事实库", "密级=", "机密★", "绝密★", "秘密★",
            "【内部资产台账】",
        ]
        hits: List[str] = []
        seen = set()

        for marker in forbidden_markers:
            if marker in raw_text or _normalize_confidential_match_text(marker) in normalized_text:
                if marker not in seen:
                    hits.append(marker)
                    seen.add(marker)

        for value in self._iter_confidential_sensitive_values(assets, policy):
            if not value:
                continue
            normalized_value = _normalize_confidential_match_text(value)
            if value in raw_text or (normalized_value and normalized_value in normalized_text):
                if value not in seen:
                    hits.append(value)
                    seen.add(value)
        return hits

    def _iter_confidential_sensitive_values(
        self,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> List[str]:
        values: List[str] = []
        fields = [
            "id",
            "secret_content",
            "secret_summary",
            "secret_keywords",
            "confidential_level",
            "attack_paraphrases",
        ]
        for field_name in list(policy.protected_fields or []) + list(policy.sensitive_fields or []):
            if field_name not in fields:
                fields.append(field_name)
        for asset in assets:
            for field_name in fields:
                raw_value = asset.id if field_name == "id" else asset.extra.get(field_name, asset.get(field_name))
                if isinstance(raw_value, list):
                    values.extend(str(item).strip() for item in raw_value)
                else:
                    values.append(str(raw_value or "").strip())

        secret_aliases = (policy.field_aliases or {}).get("secret_content", {})
        if isinstance(secret_aliases, dict):
            for canonical, aliases in secret_aliases.items():
                values.append(str(canonical or "").strip())
                if isinstance(aliases, list):
                    values.extend(str(item).strip() for item in aliases if len(str(item).strip()) >= 4)

        deduped: List[str] = []
        seen = set()
        for value in values:
            # Very short/common labels (for example "内部" or "high") create false positives
            # in safe aggregate policy text. Keep strong identifiers regardless of length.
            if (len(value) < 4 and not value.startswith("SEC-")) or value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        return deduped

    @staticmethod
    def _is_weak_confidential_sensitive_value(field_name: str, value: str) -> bool:
        if field_name not in {
            "secret_content", "secret_summary", "secret_keywords", "confidential_level", "attack_paraphrases",
        }:
            return False
        compact = _normalize_confidential_match_text(value)
        if not compact:
            return True
        if len(compact) < 4:
            return True
        return False

    def _handle_general_chat(
        self,
        request_id: str,
        user_input: str,
        intent_info,
        mode: str,
    ) -> GatewayResponse:
        """Handle a non-domain intent (weather, general chat) without CFA pipeline.

        Uses a generic system prompt, does NOT inject internal fact pools,
        and skips CFA-Score analysis entirely (no sensitive data to leak).
        """
        system_prompt = get_general_system_prompt(intent_info)
        # Use an empty/dummy policy — no internal data is involved
        from dataclasses import replace
        # Use a minimal valid policy — no internal data is involved
        empty_policy = FieldPolicy(
            protected_fields=[],
            identifier_fields=[],
            quasi_identifier_fields=[],
            field_order=[],
            field_labels={},
            field_weights={},
            field_aliases={},
            public_rules=[],
        )

        # Generate answer without any fact pool injection
        raw_answer = self._call_llm_with_prompt(
            user_input, system_prompt, [], empty_policy,
            inject_fact_pool=False,
            debug_metadata={
                "request_id": request_id,
                "purpose": "primary_generation",
                "scenario": "general",
                "effective_scenario": "general",
                "mode": mode,
                "secondary_check": False,
                "inject_fact_pool": False,
                "safe_knowledge_type": "",
            },
        )

        # Determine answer strategy label
        if intent_info.domain == GENERAL_WEATHER:
            # Check if user provided a city name
            has_city = _has_city_name(user_input)
            strategy = "weather_answer" if has_city else "need_city_prompt"
        else:
            strategy = "general_answer"

        return GatewayResponse(
            request_id=request_id,
            answer=raw_answer,
            raw_answer=raw_answer,
            risk_detected=False,
            risk_level="NONE",
            score=0.0,
            safe_answer_used="general_answer",
            findings_count=0,
            findings_summary=[],
            intent=intent_info.domain,
            routed_scenario="general",
            answer_strategy=strategy,
            llm_debug_refs=[{
                "request_id": request_id,
                "purpose": "primary_generation",
                "scenario": "general",
                "effective_scenario": "general",
                "mode": mode,
                "secondary_check": False,
                "inject_fact_pool": False,
                "safe_knowledge_type": "",
            }],
            sensitive_response=False,
        )

    # ------------------------------------------------------------------
    # Internal: final answer selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_final_answer(result: AnalysisResult) -> tuple:
        """Return (final_answer_text, safe_answer_used_label)."""
        if result.secondary_check_performed:
            if result.secondary_findings:
                # Residual risk → fallback
                return result.secondary_safe_answer, "fallback"
            else:
                # LLM rewrite is safe
                return result.secondary_safe_answer, "secondary_safe_answer"
        elif result.findings:
            # Risk detected → use CFA safe answer
            return result.safe_answer, "cfa_safe_answer"
        else:
            # No risk → raw_answer is safe (or use safe_answer which may be identical)
            return result.safe_answer, "raw_answer"

    def _apply_confidential_local_gate(
        self,
        *,
        request_id: str,
        user_input: str,
        model_output: str,
        result: AnalysisResult,
        assets: List[AssetFact],
        policy: FieldPolicy,
        final_answer: str,
        safe_used: str,
    ) -> tuple[str, str]:
        """Final local gate for confidential answers before any public response."""
        direct_hits = self._detect_confidential_direct_leak(model_output, assets, policy)
        synthetic_findings = self._build_direct_leak_findings(
            direct_hits,
            result,
            assets,
            policy,
        )
        if synthetic_findings:
            result.findings.extend(synthetic_findings)

        triggered_findings = [
            finding for finding in result.findings
            if finding.finding_type in {
                "direct_protected_disclosure",
                "indirect_asset_restoration",
                "indirect_protected_value_restoration",
                "input_hypothesis_confirmation",
                "input_output_slot_restoration",
            }
            or (
                finding.final_candidate_count == 1
                and finding.input_candidate_count != 1
            )
        ]
        risk_detected = bool(direct_hits or triggered_findings or result.findings)

        selected_answer = final_answer
        selected_safe_used = safe_used
        if risk_detected:
            selected_answer = self._confidential_safe_template_for_input(user_input, assets, policy, result)
            selected_safe_used = "cfa_safe_answer"

        self._write_confidential_audit_event(
            request_id=request_id,
            user_input=user_input,
            model_output=model_output,
            result=result,
            direct_hits=direct_hits,
            safe_answer_used=selected_safe_used,
        )
        return selected_answer, selected_safe_used

    def _confidential_safe_template_for_input(
        self,
        user_input: str,
        assets: List[AssetFact],
        policy: FieldPolicy,
        result: Optional[AnalysisResult] = None,
    ) -> str:
        slot_answer = self._confidential_slot_safe_answer_for_input(user_input, result)
        if slot_answer:
            return slot_answer
        service = ConfidentialLocalService(assets=assets, policy=policy)
        decision = service.classify_and_match(user_input)
        sub_scene = decision.get("sub_scene", "confidential_general") if decision.get("matched") else "confidential_general"
        return ConfidentialLocalService.get_safe_template(str(sub_scene))

    def _confidential_slot_safe_answer_for_input(
        self,
        user_input: str,
        result: Optional[AnalysisResult],
    ) -> str:
        if result is None or not result.findings:
            return ""
        has_slot_restoration = any(
            f.finding_type == "input_output_slot_restoration"
            for f in result.findings
        )
        if not has_slot_restoration and not self._looks_like_confidential_slot_query(user_input):
            return ""
        subject = self._extract_confidential_slot_subject(user_input)
        if not subject:
            return ""
        return f"根据查询到的信息，{subject}为受保护。"

    @staticmethod
    def _looks_like_confidential_slot_query(user_input: str) -> bool:
        text = unicodedata.normalize("NFKC", str(user_input or "")).lower()
        compact = re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)
        query_terms = ("多少", "几分", "多少分", "是什么", "为多少", "是多少", "多久")
        attribute_terms = (
            "成绩", "考试", "分数", "结果", "结论", "期限", "金额", "预算", "经费", "比例",
            "数量", "容量", "级别", "密级", "类别", "属性", "要求", "安排", "脱密期",
        )
        return any(term in compact for term in query_terms) and any(term in compact for term in attribute_terms)

    @staticmethod
    def _extract_confidential_slot_subject(user_input: str) -> str:
        text = unicodedata.normalize("NFKC", str(user_input or "")).strip()
        text = re.sub(r"[？?。！!，,；;：:\s]+$", "", text)
        text = re.sub(r"^(请问|请告知|请查询|查询一下|查一下|查询|请|问)", "", text)
        replacements = [
            "为多少分", "是多少分", "多少分", "几分", "为多少", "是多少", "是什么", "多少", "吗", "呢",
        ]
        for item in sorted(replacements, key=len, reverse=True):
            text = text.replace(item, "")
        text = text.rstrip("为是的")
        text = re.sub(r"\s+", "", text)
        if len(text) < 4 or len(text) > 80:
            return ""
        return text

    def _build_confidential_cfa_evidence(
        self,
        result: AnalysisResult,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> List[Dict[str, Any]]:
        """Build opt-in local-demo evidence for restored confidential facts."""
        asset_by_id = {asset.id: asset for asset in assets}
        evidence: List[Dict[str, Any]] = []
        for finding in result.findings:
            target_asset = asset_by_id.get(finding.target_asset_id)
            restored_fact = finding.restored_fact or self._format_restored_fact_for_evidence(
                finding,
                target_asset,
                policy,
            )
            restored_fields = []
            for field_name in finding.restored_fields or []:
                restored_fields.append({
                    "name": field_name,
                    "label": policy.label(field_name),
                    "value": target_asset.get(field_name) if target_asset else "",
                })
            reasoning_process = self._build_confidential_reasoning_process(
                finding,
                restored_fact,
            )
            evidence.append({
                "target_id": finding.target_asset_id,
                "target_name": finding.target_asset_name or (target_asset.display_name(policy.display_field) if target_asset else ""),
                "risk_level": finding.risk_level,
                "score": finding.score,
                "finding_type": finding.finding_type,
                "restored_fact": restored_fact,
                "restored_fields": restored_fields,
                "reason": finding.reason,
                "key_anchors": list(finding.key_anchor_summary or []),
                "reduction_chain": [step.to_dict() for step in finding.reduction_chain],
                "reasoning_process": reasoning_process,
                "input_candidate_count": finding.input_candidate_count,
                "final_candidate_count": finding.final_candidate_count,
                "information_gain_bits": finding.information_gain_bits,
            })
        return evidence

    @staticmethod
    def _build_confidential_reasoning_process(
        finding: RiskFinding,
        restored_fact: str,
    ) -> List[str]:
        steps: List[str] = []
        key_anchors = list(finding.key_anchor_summary or [])
        input_anchors = [item for item in key_anchors if item.startswith("用户输入/")]
        output_anchors = [item for item in key_anchors if item.startswith("模型输出/")]

        if input_anchors:
            steps.append("用户输入命中或假设了受限事实锚点：" + "；".join(input_anchors[:3]))
        else:
            steps.append("CFA 从用户输入中提取上下文锚点，用于限定候选事实集合。")

        if output_anchors:
            steps.append("LLM 原始输出新增确认/复述信号：" + "；".join(output_anchors[:4]))
        else:
            steps.append("LLM 原始输出提供了可参与组合还原的线索。")

        if finding.input_candidate_count or finding.final_candidate_count:
            steps.append(
                f"CFA 候选收敛：{finding.input_candidate_count or 0} → "
                f"{finding.final_candidate_count or 0}，信息增益 "
                f"{finding.information_gain_bits:.2f} bit。"
            )

        if restored_fact:
            steps.append("被还原的受限事实：" + restored_fact)
        steps.append("最终答复由 CFA/本地保密闸门替换为安全授权查询提示。")
        return steps

    @staticmethod
    def _format_restored_fact_for_evidence(
        finding: RiskFinding,
        asset: Optional[AssetFact],
        policy: FieldPolicy,
    ) -> str:
        if asset is None:
            return "受控字段状态被还原"
        field_names = list(finding.restored_fields or [])
        if not field_names:
            field_names = [field for field in policy.protected_fields if asset.get(field)]
        parts = [
            f"{policy.label(field)}={asset.get(field)}"
            for field in field_names
            if asset.get(field)
        ]
        if not parts:
            parts.append("受控字段状态被间接还原")
        return f"{asset.display_name(policy.display_field)}：" + "，".join(parts)

    def _build_direct_leak_findings(
        self,
        direct_hits: List[str],
        result: AnalysisResult,
        assets: List[AssetFact],
        policy: FieldPolicy,
    ) -> List[RiskFinding]:
        """Create internal findings when normalized direct matching catches a leak."""
        if not direct_hits:
            return []
        existing_ids = {
            finding.target_asset_id
            for finding in result.findings
            if finding.finding_type == "direct_protected_disclosure"
        }
        raw_output = result.raw_answer or result.model_output
        output_norm = _normalize_confidential_match_text(raw_output)
        generic_marker_norms = {
            _normalize_confidential_match_text(marker)
            for marker in [
                "SEC-", "保密内容=", "保密关键词=", "保密类别=", "保密摘要=",
                "保密事实库", "密级=", "机密★", "绝密★", "秘密★",
                "【内部资产台账】",
            ]
        }
        hit_norms = [
            _normalize_confidential_match_text(hit)
            for hit in direct_hits
            if _normalize_confidential_match_text(hit)
            and _normalize_confidential_match_text(hit) not in generic_marker_norms
        ]
        findings: List[RiskFinding] = []
        protected_fields = list(policy.protected_fields or [])
        sensitive_fields = [
            "id",
            "secret_content",
            "secret_summary",
            "secret_keywords",
            "confidential_level",
            "attack_paraphrases",
        ]
        for field_name in protected_fields + list(policy.sensitive_fields or []):
            if field_name not in sensitive_fields:
                sensitive_fields.append(field_name)

        covered_by_existing_finding = False
        for asset in assets:
            matched_fields: List[str] = []
            asset_values: List[str] = [asset.id]
            for field_name in sensitive_fields:
                raw_value = asset.id if field_name == "id" else asset.extra.get(field_name, asset.get(field_name))
                values = raw_value if isinstance(raw_value, list) else [raw_value]
                for value in values:
                    value_text = str(value or "").strip()
                    if not value_text or self._is_weak_confidential_sensitive_value(field_name, value_text):
                        continue
                    asset_values.append(value_text)
                    value_norm = _normalize_confidential_match_text(value_text)
                    if value_text in raw_output or (value_norm and value_norm in output_norm):
                        matched_fields.append(field_name)
                        break

            asset_blob_norm = _normalize_confidential_match_text(" ".join(asset_values))
            if not matched_fields and any(hit_norm and hit_norm in asset_blob_norm for hit_norm in hit_norms):
                matched_fields = protected_fields or ["secret_content"]

            if not matched_fields:
                continue
            if asset.id in existing_ids:
                covered_by_existing_finding = True
                continue

            findings.append(
                RiskFinding(
                    target_asset_id=asset.id,
                    target_asset_name=asset.display_name(policy.display_field),
                    risk_level="CRITICAL",
                    score=100.0,
                    reason="模型输出直接包含保密库受保护字段值。",
                    restored_fact="",
                    anchors=[],
                    reduction_chain=[],
                    minimal_combinations=[],
                    finding_type="direct_protected_disclosure",
                    target_asset_ids=[asset.id],
                    restored_fields=sorted(set(matched_fields)),
                    input_candidate_count=len(assets),
                    final_candidate_count=1,
                    information_gain_bits=0.0,
                )
            )

        if not findings and not covered_by_existing_finding:
            findings.append(
                RiskFinding(
                    target_asset_id="",
                    target_asset_name="confidential_fact",
                    risk_level="CRITICAL",
                    score=100.0,
                    reason="模型输出直接包含保密库受保护字段值。",
                    restored_fact="",
                    anchors=[],
                    reduction_chain=[],
                    minimal_combinations=[],
                    finding_type="direct_protected_disclosure",
                    target_asset_ids=[],
                    restored_fields=protected_fields,
                    input_candidate_count=len(assets),
                    final_candidate_count=1,
                    information_gain_bits=0.0,
                )
            )
        return findings

    def _write_confidential_audit_event(
        self,
        *,
        request_id: str,
        user_input: str,
        model_output: str,
        result: AnalysisResult,
        direct_hits: List[str],
        safe_answer_used: str,
    ) -> None:
        payload = self._build_confidential_audit_payload(
            request_id=request_id,
            user_input=user_input,
            model_output=model_output,
            result=result,
            direct_hits=direct_hits,
            safe_answer_used=safe_answer_used,
        )
        log_path = self._base_dir / "logs" / "confidential_audit.jsonl"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            # Audit must not break user-facing safety; fail closed on response data
            # by keeping sensitive details out of GatewayResponse regardless.
            return

    def _build_confidential_audit_payload(
        self,
        *,
        request_id: str,
        user_input: str,
        model_output: str,
        result: AnalysisResult,
        direct_hits: List[str],
        safe_answer_used: str,
    ) -> Dict[str, Any]:
        findings = list(result.findings or [])
        restored_ids = sorted({
            asset_id
            for finding in findings
            for asset_id in ([finding.target_asset_id] + list(finding.target_asset_ids or []))
            if asset_id
        })
        trigger_types = sorted({finding.finding_type for finding in findings if finding.finding_type})
        if direct_hits:
            trigger_types.append("direct_confidential_value_match")
        restored_fields = sorted({
            field_name
            for finding in findings
            for field_name in list(finding.restored_fields or [])
            if field_name
        })
        input_counts = [f.input_candidate_count for f in findings if f.input_candidate_count]
        final_counts = [f.final_candidate_count for f in findings if f.final_candidate_count]
        info_gains = [f.information_gain_bits for f in findings if f.information_gain_bits]
        return {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "request_id": request_id,
            "scenario": "confidential",
            "user_input_hash": _sha256_text(user_input),
            "llm_output_hash": _sha256_text(model_output),
            "risk_detected": bool(findings or direct_hits),
            "trigger_types": sorted(set(trigger_types)),
            "restored_fact_ids": restored_ids,
            "restored_fields": restored_fields,
            "direct_hit_count": len(direct_hits),
            "input_candidate_count": min(input_counts) if input_counts else 0,
            "final_candidate_count": min(final_counts) if final_counts else 0,
            "information_gain_bits": round(max(info_gains), 4) if info_gains else 0.0,
            "safe_answer_used": safe_answer_used,
        }

    # ------------------------------------------------------------------
    # Admin API: protected fact management
    # ------------------------------------------------------------------

    def _get_confidential_import_meta_path(self) -> Path:
        return self._base_dir / "config" / "confidential_import_meta.json"

    def _read_confidential_import_meta(self) -> Optional[dict]:
        path = self._get_confidential_import_meta_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _write_confidential_import_meta(self, meta: dict) -> None:
        path = self._get_confidential_import_meta_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_scenario_files(self, scenario: str) -> tuple:
        """Return (facts_path, policy_path) for a scenario without loading/caching."""
        if scenario in ("auto", "general"):
            raise ValueError("事实管理不能使用 auto/general，请选择具体业务场景")

        if scenario not in _SCENARIO_PRESETS:
            available = ", ".join(sorted(_SCENARIO_PRESETS.keys()))
            raise ValueError(f"Unknown scenario: '{scenario}'. Available: {available}")

        preset = _SCENARIO_PRESETS[scenario]
        facts_path = self._base_dir / preset["facts"]
        policy_path = self._base_dir / preset["policy"]
        return facts_path, policy_path

    def get_fact_schema(self, scenario: str) -> dict:
        """Return the field schema for a scenario: field names, labels, and whether each is protected/identifier."""
        assets, policy = self._load_scenario(scenario)

        field_names: List[str] = []
        for name in ["id", *policy.field_order]:
            if name and name not in field_names:
                field_names.append(name)

        # If policy.field_order doesn't cover all fields, supplement from the first fact
        if assets:
            first = assets[0]
            for name in first.extra.keys():
                if name not in field_names:
                    field_names.append(name)

        protected = set(policy.protected_fields)
        identifiers = set(policy.identifier_fields)
        labels = policy.field_labels or {}

        return {
            "scenario": scenario,
            "display_field": policy.display_field,
            "protected_fields": list(policy.protected_fields),
            "identifier_fields": list(policy.identifier_fields),
            "fields": [
                {
                    "name": name,
                    "label": labels.get(name, name),
                    "protected": name in protected,
                    "identifier": name in identifiers,
                    "required": name == "id",
                }
                for name in field_names
            ],
        }

    def list_protected_facts(self, scenario: str) -> dict:
        """Return all facts currently stored for a scenario.

        Note: This is an admin endpoint — production deployments should add authentication.
        """
        facts_path, _ = self._get_scenario_files(scenario)

        data = json.loads(facts_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("facts json must be a list")

        result = {
            "scenario": scenario,
            "count": len(data),
            "facts": data,
        }
        if scenario == "confidential":
            result["import_meta"] = self._read_confidential_import_meta()
        return result

    def add_protected_fact(self, scenario: str, fact: dict) -> dict:
        """Append a new fact to the scenario's facts JSON file.

        Whether fields are protected is determined by policy.protected_fields — not by the caller.
        """
        if not isinstance(fact, dict):
            raise ValueError("fact must be a JSON object")

        facts_path, _ = self._get_scenario_files(scenario)

        data = json.loads(facts_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, list):
            raise ValueError("facts json must be a list")

        # Clean empty fields
        clean_fact: Dict[str, Any] = {}
        for key, value in fact.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
            if value == "":
                continue
            clean_fact[key] = value

        if not clean_fact.get("id"):
            clean_fact["id"] = f"{scenario.upper()}-{len(data) + 1:03d}"

        fact_id = str(clean_fact["id"])

        for row in data:
            if str(row.get("id")) == fact_id:
                raise ValueError(f"事实 ID 已存在：{fact_id}")

        data.append(clean_fact)

        facts_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Clear cache so the new fact is visible immediately
        self._scenario_cache.pop(scenario, None)

        return {
            "ok": True,
            "scenario": scenario,
            "id": fact_id,
            "count": len(data),
            "fact": clean_fact,
        }

    # ------------------------------------------------------------------
    # Admin API: JSONL confidential asset import
    # ------------------------------------------------------------------

    def _backup_file(self, path: Path) -> None:
        if not path.exists():
            return

        backup_dir = path.parent / "_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{path.stem}.{ts}{path.suffix}.bak"
        shutil.copy2(path, backup_path)

    def _normalize_jsonl_row(self, row: dict, index: int) -> dict:
        fact_text = str(row.get("fact_text") or "").strip()
        if not fact_text:
            raise ValueError("缺少 fact_text")

        category = str(row.get("category") or "未分类").strip()
        confidential_level = str(row.get("confidential_level") or "high").strip()
        summary = str(row.get("summary") or fact_text[:60]).strip()

        keywords = row.get("keywords") or []
        if isinstance(keywords, list):
            secret_keywords = "；".join(str(x).strip() for x in keywords if str(x).strip())
        else:
            secret_keywords = str(keywords).strip()

        paraphrases = row.get("paraphrases") or []
        if not isinstance(paraphrases, list):
            paraphrases = [str(paraphrases)]

        negative_samples = row.get("negative_samples") or []
        if not isinstance(negative_samples, list):
            negative_samples = [str(negative_samples)]

        digest = hashlib.sha1(fact_text.encode("utf-8")).hexdigest()[:10].upper()

        return {
            "id": f"SEC-{index:06d}-{digest}",
            "category": category,
            "confidential_level": confidential_level,
            "secret_summary": summary,
            "secret_content": fact_text,
            "secret_keywords": secret_keywords,
            "attack_paraphrases": [str(x).strip() for x in paraphrases if str(x).strip()],
            "negative_samples": [str(x).strip() for x in negative_samples if str(x).strip()],
            "source": "jsonl_import"
        }

    def import_confidential_jsonl(
        self,
        *,
        content: str,
        filename: str = "",
        replace: bool = False,
    ) -> dict:
        """
        前端上传 JSONL 文本后，导入为 confidential 场景的受保护事实。

        JSONL 输入字段：
        - fact_text
        - category
        - confidential_level
        - summary
        - paraphrases
        - negative_samples
        - keywords
        """
        scenario = "confidential"
        facts_path, policy_path = self._get_scenario_files(scenario)

        facts_path.parent.mkdir(parents=True, exist_ok=True)

        if not facts_path.exists():
            facts_path.write_text("[]", encoding="utf-8")

        if not policy_path.exists():
            raise ValueError("缺少 config/confidential_policy.json，请先创建该策略文件")

        raw_lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not raw_lines:
            raise ValueError("JSONL 文件为空")

        imported = []
        errors = []
        category_counter: dict[str, int] = {}
        level_counter: dict[str, int] = {}

        for line_no, line in enumerate(raw_lines, start=1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("该行不是 JSON object")

                fact = self._normalize_jsonl_row(row, len(imported) + 1)

                imported.append(fact)

                category = fact["category"]
                level = fact["confidential_level"]
                category_counter[category] = category_counter.get(category, 0) + 1
                level_counter[level] = level_counter.get(level, 0) + 1

            except Exception as exc:
                errors.append({
                    "line": line_no,
                    "error": str(exc),
                    "preview": line[:120]
                })

        if not imported:
            return {
                "ok": False,
                "filename": filename,
                "total_lines": len(raw_lines),
                "imported": 0,
                "errors": errors[:20],
                "message": "没有成功导入任何事实"
            }

        self._backup_file(facts_path)
        self._backup_file(policy_path)

        if replace:
            existing = []
        else:
            try:
                existing = json.loads(facts_path.read_text(encoding="utf-8-sig"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []

        existing_contents = {
            str(item.get("secret_content", "")).strip()
            for item in existing
            if isinstance(item, dict)
        }

        final_imported = []
        duplicate_count = 0

        for fact in imported:
            content_key = str(fact.get("secret_content", "")).strip()
            if content_key in existing_contents:
                duplicate_count += 1
                continue

            final_imported.append(fact)
            existing_contents.add(content_key)

        final_facts = existing + final_imported

        facts_path.write_text(
            json.dumps(final_facts, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        # 把 paraphrases / keywords / summary 写入 field_aliases，提升改写问法命中率
        policy = json.loads(policy_path.read_text(encoding="utf-8-sig"))
        field_aliases = policy.setdefault("field_aliases", {})
        secret_aliases = field_aliases.setdefault("secret_content", {})

        for fact in final_imported:
            canonical = fact["secret_content"]
            aliases = []

            if fact.get("secret_summary"):
                aliases.append(fact["secret_summary"])

            if fact.get("secret_keywords"):
                aliases.extend([
                    x.strip()
                    for x in str(fact["secret_keywords"]).split("；")
                    if x.strip()
                ])

            aliases.extend(fact.get("attack_paraphrases") or [])

            # 去重，避免 policy 文件膨胀
            old_aliases = secret_aliases.get(canonical, [])
            merged = []
            seen = set()

            for item in [*old_aliases, *aliases]:
                item = str(item).strip()
                if item and item not in seen and item != canonical:
                    seen.add(item)
                    merged.append(item)

            if merged:
                secret_aliases[canonical] = merged[:20]

        policy_path.write_text(
            json.dumps(policy, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        import_meta = {
            "filename": filename,
            "imported_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_lines": len(raw_lines),
            "parsed": len(imported),
            "imported": len(final_imported),
            "duplicates": duplicate_count,
            "error_count": len(errors),
            "total_facts": len(final_facts),
            "category_counter": category_counter,
            "level_counter": level_counter,
        }
        self._write_confidential_import_meta(import_meta)

        # 关键：清除缓存，否则新导入的事实不会立即生效
        self._scenario_cache.pop(scenario, None)

        return {
            "ok": True,
            "filename": filename,
            "scenario": scenario,
            "total_lines": len(raw_lines),
            "parsed": len(imported),
            "imported": len(final_imported),
            "duplicates": duplicate_count,
            "errors": errors[:20],
            "error_count": len(errors),
            "total_facts": len(final_facts),
            "category_counter": category_counter,
            "level_counter": level_counter,
            "import_meta": import_meta,
            "message": f"成功导入 {len(final_imported)} 条保密事实"
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _normalize_confidential_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _build_response(
    request_id: str,
    result: AnalysisResult,
    final_answer: str,
    safe_used: str,
    *,
    intent: str = "",
    routed_scenario: str = "",
    answer_strategy: str = "",
    llm_debug_refs: Optional[List[Dict[str, Any]]] = None,
    sensitive_response: bool = False,
    confidential_context_summary: Optional[Dict[str, Any]] = None,
    confidential_cfa_evidence: Optional[List[Dict[str, Any]]] = None,
    demo_raw_answer: str = "",
    demo_raw_answer_redacted: str = "",
    demo_raw_answer_mode: str = "",
) -> GatewayResponse:
    # Determine aggregate risk level and score
    if result.findings:
        max_score = max(f.score for f in result.findings)
        levels = sorted(
            result.findings,
            key=lambda f: _risk_severity(f.risk_level),
            reverse=True,
        )
        risk_level = levels[0].risk_level
    else:
        max_score = 0.0
        risk_level = "NONE"

    # Build findings summary for UI display.  Confidential responses keep
    # details backend-only because target IDs and reduction chains are evidence
    # of restored protected facts.
    findings_summary = []
    if not sensitive_response:
        for f in result.findings:
            findings_summary.append({
                "target": f.target_asset_name or f.target_asset_id,
                "target_id": f.target_asset_id,
                "level": f.risk_level,
                "score": f.score,
                "reason": f.reason,
                "restored": f.restored_fact,
                "key_anchors": f.key_anchor_summary,
                "chain": [s.to_dict() for s in f.reduction_chain],
            })

    return GatewayResponse(
        request_id=request_id,
        answer=final_answer,
        raw_answer=result.raw_answer,
        risk_detected=len(result.findings) > 0,
        risk_level=risk_level,
        score=max_score,
        safe_answer_used=safe_used,
        findings_count=len(result.findings),
        findings_summary=findings_summary,
        intent=intent,
        routed_scenario=routed_scenario,
        answer_strategy=answer_strategy,
        llm_debug_refs=list(llm_debug_refs or []),
        sensitive_response=sensitive_response,
        confidential_context_summary=confidential_context_summary,
        confidential_cfa_evidence=confidential_cfa_evidence,
        demo_raw_answer=demo_raw_answer,
        demo_raw_answer_redacted=demo_raw_answer_redacted,
        demo_raw_answer_mode=demo_raw_answer_mode,
    )


_RISK_ORDER = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NONE": 0}


def _risk_severity(level: str) -> int:
    return _RISK_ORDER.get(level, 0)


# ---------------------------------------------------------------------------
# Chinese city name detection (lightweight regex)
# ---------------------------------------------------------------------------

# Common Chinese city names (abbreviated list for weather context)
_CITY_PATTERNS = [
    r"北京", r"上海", r"广州", r"深圳", r"杭州", r"南京", r"成都",
    r"武汉", r"重庆", r"天津", r"西安", r"苏州", r"长沙", r"郑州",
    r"青岛", r"大连", r"厦门", r"宁波", r"福州", r"合肥", r"济南",
    r"沈阳", r"哈尔滨", r"长春", r"昆明", r"贵阳", r"南宁", r"海口",
    r"乌鲁木齐", r"拉萨", r"兰州", r"银川", r"西宁", r"呼和浩特",
    r"石家庄", r"太原", r"南昌", r"无锡", r"东莞", r"佛山", r"珠海",
    r"三亚", r"桂林", r"丽江", r"大理", r"张家界", r"黄山",
]


def _has_city_name(text: str) -> bool:
    """Check if the text contains a Chinese city name."""
    for pattern in _CITY_PATTERNS:
        if pattern in text:
            return True
    return False