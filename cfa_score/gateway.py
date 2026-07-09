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

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter import DeepSeekAdapter
from .deepseek import DeepSeekClient, config_from_env
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "risk_detected": self.risk_detected,
            "risk_level": self.risk_level,
            "score": self.score,
            "safe_answer_used": self.safe_answer_used,
            "findings_count": self.findings_count,
            "findings_summary": self.findings_summary,
            "intent": self.intent,
            "routed_scenario": self.routed_scenario,
            "answer_strategy": self.answer_strategy,
        }


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
        raw_answer = self._call_llm_with_prompt(
            user_input, system_prompt, assets, policy,
            inject_fact_pool=True,
        )

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
        adapter = DeepSeekAdapter(
            env_path=self._env_path,
            system_prompt=system_prompt,
        )
        public_knowledge = getattr(policy, "public_rules", [])
        context = {
            "fact_pool": assets if inject_fact_pool else [],
            "public_knowledge": public_knowledge if inject_fact_pool else [],
            "policy": policy,
        }
        return adapter.generate(user_input, context=context)

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
        empty_policy = FieldPolicy(
            protected_fields=[],
            identifier_fields=[],
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
