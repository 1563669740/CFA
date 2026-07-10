"""
Intent Router — lightweight keyword/pattern-based classifier.

Routes user_input to one of the predefined domains or a general category
before the question reaches the scenario pipeline.  This prevents
off-topic questions (weather, chit-chat, encyclopedia) from being
answered with a domain-specific identity (e.g. hospital AI assistant).

Usage::

    from cfa_score.intent_router import classify_intent, Intent

    intent = classify_intent("介绍一下明天的天气")
    # Intent(domain="general_weather", confidence=0.9, reason="...")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Intent domain constants
# ---------------------------------------------------------------------------

DOMAIN_HEALTHCARE  = "domain_healthcare"
DOMAIN_FINANCE     = "domain_finance"
DOMAIN_AEROSPACE   = "domain_aerospace"
DOMAIN_MEETINGS    = "domain_meetings"
GENERAL_WEATHER    = "general_weather"
GENERAL_CHAT       = "general_chat"
AMBIGUOUS          = "ambiguous"


@dataclass
class Intent:
    """Classified intent with confidence and rationale."""
    domain: str
    confidence: float          # 0.0 – 1.0
    reason: str = ""
    matched_keywords: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keyword → domain mapping (Chinese + English patterns)
# Each tuple: (keyword_pattern, weight)
# ---------------------------------------------------------------------------

# ---------- Healthcare ----------
_HEALTHCARE_PATTERNS: List[Tuple[str, float]] = [
    # Strong signals — explicit medical context
    (r"住院|出院|转院|入院", 0.95),
    (r"患者|病人|病患|病号", 0.95),
    (r"ICU|CCU|NICU|重症监护|监护室", 0.95),
    (r"诊断|确诊|疑似|查体|体格检查", 0.90),
    (r"用药|处方|医嘱|给药|剂量|服用|静脉|口服|注射", 0.90),
    (r"医保|社保|自费|报销|结算|缴费|住院费|门诊费", 0.85),
    (r"病区|科室|病房|床位|护理|护士站", 0.85),
    (r"病史|既往史|过敏史|家族史|个人史", 0.90),
    (r"化验|体检|医学检查|身体检查|影像检查|"
     r"CT|MRI|X光|B超|心电图|血常规|尿常规", 0.85),
    (r"手术|术后|术前|麻醉|切口|拆线|换药", 0.85),
    (r"抗凝|降压|降糖|降脂|抗生素|激素|化疗|放疗", 0.85),
    (r"急性|慢性|心肌梗死|心梗|脑梗|脑出血|肺炎|糖尿病|高血压|冠心病", 0.90),
    (r"体温|血压|心率|呼吸|血氧|脉搏", 0.70),
    # Moderate signals — could be ambiguous in isolation
    (r"替格瑞洛|阿司匹林|氯吡格雷|华法林|肝素", 0.90),
    (r"主治医生|主任医师|住院医生|管床医生|护士长", 0.80),
]

# ---------- Finance ----------
_FINANCE_PATTERNS: List[Tuple[str, float]] = [
    (r"贷款|信贷|授信|额度|放款|还款|展期|续贷", 0.95),
    (r"利率|利息|LPR|基准利率|浮动利率|固定利率", 0.90),
    (r"抵押|质押|担保|保证人|反担保", 0.90),
    (r"企业信用|征信|信用评级|信用报告|黑名单|失信", 0.90),
    (r"流动资金|项目贷款|固定资产贷款|并购贷款|银团贷款", 0.85),
    (r"审批|风控|尽调|贷前|贷后|贷中", 0.80),
    (r"银行|支行|分行|信贷部|风险部", 0.70),
    (r"资产负债表|利润表|现金流量表|财报|审计", 0.75),
    (r"不良贷款|逾期|坏账|拨备|核销", 0.85),
    (r"半导体|芯片|集成电路|光伏|新能源|制造业|房地产", 0.40),  # weak — could be general
]

# ---------- Aerospace ----------
_AEROSPACE_PATTERNS: List[Tuple[str, float]] = [
    (r"航天|测控|卫星|火箭|发射|轨道|遥测", 0.95),
    (r"CVE-\d{4}-\d+|漏洞|补丁|版本号|组件版本|依赖", 0.85),
    (r"XZ|\.xz|liblzma|供应链|后门|植入", 0.85),
    (r"生产区|测试区|研发区|办公区|网段|VLAN", 0.80),
    (r"资产|台账|受控|涉密|密级|定密", 0.50),  # could be generic
    (r"远程入口|SSH|RDP|堡垒机|跳板机|VPN|专线", 0.75),
    (r"处置|止血|隔离|下线|封禁|应急响应", 0.60),
    (r"运维|监控|告警|日志|审计|态势感知", 0.50),
]

# ---------- Meetings ----------
_MEETINGS_PATTERNS: List[Tuple[str, float]] = [
    (r"会议室|会议|例会|周会|月会|汇报|评审|研讨会", 0.90),
    (r"预约|预订|预定|占用|空闲|时间段", 0.80),
    (r"参会|与会|主持人|记录人|纪要|议题|议程", 0.85),
    (r"视频会议|电话会议|线上会议|线下会议|远程会议", 0.85),
    (r"投屏|投影|白板|麦克风|音响|设备", 0.60),
]

# ---------- Weather ----------
_WEATHER_PATTERNS: List[Tuple[str, float]] = [
    (r"天气|气温|温度|降雨|降雪|刮风|风力|雾霾|空气质量|AQI|湿度|紫外线", 0.95),
    (r"明天|今天|后天|未来.*天|下周|本周|周末", 0.70),  # weak alone, but combined with weather = strong
    (r"晴|阴|多云|雨|雪|暴|台风|寒潮|高温|低温|霜冻|冰雹", 0.60),
    (r"天气预报|气象|天气查询|带伞|穿衣|出行|防晒", 0.85),
]

# ---------- General Chat (weak catch-alls) ----------
_GENERAL_CHAT_PATTERNS: List[Tuple[str, float]] = [
    (r"你好|您好|嗨|哈喽|hello|hi", 0.80),
    (r"谢谢|感谢|多谢|thank", 0.80),
    (r"再见|拜拜|bye|晚安|早安|下午好|晚上好|中午好", 0.80),
    (r"你是谁|你叫什么|你的名字|你的功能|你能做什么|介绍一下自己", 0.80),
    (r"讲个笑话|说个笑话|冷笑话|段子", 0.80),
    (r"什么是|介绍一下|解释一下|科普|百科|知识", 0.40),  # very weak — could be domain-specific too
]


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> Intent:
    """Classify user input into a domain category.

    Returns an Intent with the highest-confidence matching domain.
    If no domain pattern matches above threshold, falls back to ``general_chat``.
    """
    if not text or not text.strip():
        return Intent(domain=GENERAL_CHAT, confidence=0.5, reason="empty input")

    text_lower = text.lower()

    # ---- 上下文判断：避免“联合检查组”“安全检查”等非医学“检查”触发医疗领域 ----
    _MEDICAL_CONTEXT = re.compile(
        r"医院|医生|患者|病人|门诊|住院|病房|科室|诊断|治疗|体检"
    )
    _has_medical_context_for_check = (
        "检查" in text_lower and _MEDICAL_CONTEXT.search(text_lower)
    )

    # Score each domain
    scores: List[Tuple[str, float, List[str]]] = []

    for domain, patterns in [
        (DOMAIN_HEALTHCARE, _HEALTHCARE_PATTERNS),
        (DOMAIN_FINANCE, _FINANCE_PATTERNS),
        (DOMAIN_AEROSPACE, _AEROSPACE_PATTERNS),
        (DOMAIN_MEETINGS, _MEETINGS_PATTERNS),
        (GENERAL_WEATHER, _WEATHER_PATTERNS),
        (GENERAL_CHAT, _GENERAL_CHAT_PATTERNS),
    ]:
        matched_kws: List[str] = []
        domain_score = 0.0
        for pattern, weight in patterns:
            if re.search(pattern, text_lower):
                matched_kws.append(pattern)
                # Use a simple multiplicative combiner: score = 1 - prod(1 - weight_i)
                # This penalizes single weak matches and rewards multiple strong matches
                domain_score = 1.0 - (1.0 - domain_score) * (1.0 - weight)
        if domain_score > 0:
            scores.append((domain, domain_score, matched_kws))

    # Context-aware "检查" check: only boost healthcare if medical context exists
    if _has_medical_context_for_check:
        healthcare_entry = next((s for s in scores if s[0] == DOMAIN_HEALTHCARE), None)
        if healthcare_entry:
            idx = scores.index(healthcare_entry)
            boosted = 1.0 - (1.0 - healthcare_entry[1]) * (1.0 - 0.75)
            scores[idx] = (DOMAIN_HEALTHCARE, min(1.0, boosted), healthcare_entry[2] + ["检查(医学上下文)"])
        else:
            scores.append((DOMAIN_HEALTHCARE, 0.75, ["检查(医学上下文)"]))

    # Special case: if weather keywords exist AND a time keyword exists,
    # boost weather confidence significantly
    has_weather_kw = any(
        re.search(p, text_lower) for p, _ in _WEATHER_PATTERNS
    )
    has_time_kw = bool(re.search(r"明天|今天|后天|未来|下周|本周|周末|预报|查询", text_lower))
    if has_weather_kw and has_time_kw:
        # Boost or add weather entry
        weather_entry = next((s for s in scores if s[0] == GENERAL_WEATHER), None)
        if weather_entry:
            # Boost existing
            idx = scores.index(weather_entry)
            boosted = min(1.0, weather_entry[1] * 1.5)
            scores[idx] = (GENERAL_WEATHER, boosted, weather_entry[2])
        else:
            scores.append((GENERAL_WEATHER, 0.85, ["天气+时间组合"]))

    # Sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    if scores and scores[0][1] >= 0.55:
        best = scores[0]
        return Intent(
            domain=best[0],
            confidence=round(best[1], 3),
            reason=f"matched {len(best[2])} pattern(s): {', '.join(best[2][:5])}",
            matched_keywords=best[2],
        )

    # No strong match → general chat
    return Intent(
        domain=GENERAL_CHAT,
        confidence=0.3,
        reason="no domain pattern matched above threshold",
        matched_keywords=[],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_domain_intent(intent: Intent) -> bool:
    """Return True if the intent maps to a specific business domain (not general)."""
    return intent.domain in (DOMAIN_HEALTHCARE, DOMAIN_FINANCE, DOMAIN_AEROSPACE, DOMAIN_MEETINGS)


def map_intent_to_scenario(intent: Intent) -> str:
    """Map an Intent to a scenario ID understood by CFAGateway."""
    mapping = {
        DOMAIN_HEALTHCARE: "healthcare",
        DOMAIN_FINANCE:    "finance",
        DOMAIN_AEROSPACE:  "aerospace",
        DOMAIN_MEETINGS:   "meetings",
    }
    return mapping.get(intent.domain, "general")


def get_general_system_prompt(intent: Intent) -> str:
    """Return an appropriate system prompt for a general (non-domain) intent."""
    if intent.domain == GENERAL_WEATHER:
        return (
            "你是一个通用的AI助手。当用户询问天气时，请遵循以下规则：\n"
            "- 如果用户没有指定城市，礼貌地追问「请问您想查询哪个城市的天气？」\n"
            "- 如果用户指定了城市，请说明「当前未接入实时天气数据源，无法提供准确的天气预报。"
            "建议您使用天气类 App 或访问中国气象局官网查询。」\n"
            "- 不要编造天气数据，不要以任何专业领域（医疗、金融、航天等）助手的身份回答。\n"
            "- 回答应友好、简洁，控制在 2-3 句话以内。"
        )
    # general_chat / ambiguous
    return (
        "你是一个通用的AI助手，可以帮助用户解答各种问题。\n"
        "注意：\n"
        "- 回答应友好、专业、自然\n"
        "- 不要以任何特定领域专家（医生、信贷经理、运维工程师等）的身份回答\n"
        "- 回答应简洁，控制在 2-4 句话以内\n"
        "- 如果问题超出你的知识范围，请诚实说明"
    )