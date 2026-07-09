"""
CFA-Score LLM Safety Gateway.

Orchestrates the full pipeline:

  user_input
    → LLM generation (DeepSeek) → raw_answer
    → CFA-Score analysis (engine.analyze)
    → safe_answer / secondary_safe_answer selection
    → return CFA-processed answer only (raw_answer is never exposed)

Two operation modes:
  - ``handle_chat()``    — Full pipeline: LLM generate + CFA analyze + safe answer
  - ``handle_analyze()`` — CFA only: user provides model_output, we run detection + sanitize

Design principles:
  - Zero external dependencies (Python stdlib only)
  - Scenario assets/policy are loaded once and cached
  - CFAScoreEngine is created per request (thread-safe, lightweight)
  - raw_answer, anchors, reduction_chain are NEVER returned to callers
"""

from __future__ import annotations

import json
import uuid
import hashlib
import shutil
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
from .models import AnalysisResult, FieldPolicy, AssetFact


# ---------------------------------------------------------------------------
# Response model (what the HTTP layer serializes)
# ---------------------------------------------------------------------------

@dataclass
class GatewayResponse:
    """Public-facing response.  Includes risk detail and routing info for UI transparency."""
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

    def to_dict(self, debug: bool = False) -> Dict[str, Any]:
        """Serialize to dict.  ``debug=True`` includes raw_answer and findings_summary."""
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
        if debug:
            data["raw_answer"] = self.raw_answer
            data["findings_summary"] = self.findings_summary
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

        if scenario in ("auto", "general"):
            if is_domain_intent(intent_info):
                # Route to the matching domain scenario
                effective_scenario = map_intent_to_scenario(intent_info)
            else:
                # Non-domain intent → handle as general chat (no CFA pipeline)
                return self._handle_general_chat(request_id, user_input, intent_info, mode)

        # ---- Step 1: Load scenario resources ----
        assets, policy = self._load_scenario(effective_scenario)

        # ---- Step 2: Call LLM to generate raw_answer with domain boundary ----
        system_prompt = self._build_system_prompt(effective_scenario)

        llm_debug_refs: List[Dict[str, Any]] = []
        if effective_scenario == "confidential":
            # 保密场景只把脱敏聚合知识库发给外部 LLM；原始 assets/fact_pool 不外发。
            raw_answer, llm_debug_refs = self._build_confidential_llm_answer(
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

        # ---- Step 5: Build public response ----
        return _build_response(
            request_id, result, final_answer, safe_used,
            intent=intent_info.domain,
            routed_scenario=effective_scenario,
            answer_strategy="cfa_gated",
            llm_debug_refs=llm_debug_refs,
        )

    def handle_analyze(
        self,
        user_input: str,
        model_output: str,
        scenario: str = "healthcare",
        mode: str = "rule_only",
        secondary_check: bool = False,
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

        # 4. Build public response
        return _build_response(request_id, result, final_answer, safe_used)

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
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Use a sanitized confidential KB summary for primary LLM generation."""
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
            answer = self._call_llm_with_prompt(
                self._build_confidential_llm_query(safe_knowledge),
                self._build_system_prompt("confidential"),
                [],
                policy,
                inject_fact_pool=False,
                safe_knowledge=safe_knowledge,
                debug_metadata=debug_ref,
            )
        except Exception:
            return fallback, []

        if not self._is_confidential_text_safe(answer, assets, policy):
            return fallback, [debug_ref]
        return answer, [debug_ref]

    def _build_confidential_llm_query(self, safe_knowledge: Mapping[str, Any]) -> str:
        current_query = safe_knowledge.get("current_query", {}) if isinstance(safe_knowledge, Mapping) else {}
        related = bool(current_query.get("related"))
        safe_topic = str(current_query.get("safe_topic") or "confidential_general")
        return (
            "用户正在询问内部敏感信息。"
            f"本地安全检索结果：related={related}，safe_topic={safe_topic}。"
            "请只根据安全知识库摘要和回答策略生成概括性答复，不要猜测或补充任何具体内部事实。"
        )

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
        forbidden_markers = [
            "SEC-", "保密内容=", "保密关键词=", "保密类别=", "保密摘要=",
            "保密事实库", "密级=", "机密★", "绝密★", "秘密★",
            "【内部资产台账】",
        ]
        hits = [marker for marker in forbidden_markers if marker in text]
        for value in self._iter_confidential_sensitive_values(assets, policy):
            if value and value in text:
                hits.append(value)
        if hits:
            raise RuntimeError(f"Blocked confidential text leakage: {hits[:10]}")

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

    # Build findings summary for UI display
    findings_summary = []
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