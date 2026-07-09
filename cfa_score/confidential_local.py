"""
保密知识库本地检索与场景判断。

原则：
1. 不调用外部 LLM
2. 不返回事实原文给用户
3. 不返回关键词、密级、SEC 编号给用户
4. 只返回是否命中、命中类型、内部审计信息
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .models import AssetFact, FieldPolicy


@dataclass
class LocalMatch:
    asset_id: str
    score: float
    matched_fields: List[str]
    sub_scene: str


class ConfidentialLocalService:
    """保密知识库本地检索与场景判断。

    注意：
    1. 不调用外部 LLM。
    2. 不返回事实原文。
    3. 不返回关键词、密级、SEC 编号给用户。
    4. 只返回是否命中、命中类型、内部审计信息。
    """

    # 子场景分类术语表
    _FINANCE_TERMS = [
        "预算", "经费", "采购", "合同", "金额", "万元", "亿元", "执行率",
        "首付款", "质保金", "中标", "概算", "列支",
    ]
    _PERSONNEL_TERMS = [
        "人员", "任免", "资格", "复审", "脱密", "离职", "考试", "成绩",
        "上岗审查", "涉密人员", "定密管理处", "处长",
    ]
    _AUDIT_TERMS = [
        "保密检查", "定密", "弱口令", "整改复查", "安全预警", "通报批评",
        "风险评估", "定密不规范", "整改", "复查",
    ]
    _SYSTEM_TERMS = [
        "系统", "安全评估", "服务器", "终端", "部署", "数据中心",
        "漏洞", "日志", "防火墙", "整改", "安全确认测试",
        "光纤传感", "监测", "情报",
    ]
    _PROJECT_TERMS = [
        "项目", "立项", "中期评估", "承研", "方案", "项目代号",
        "曙光", "猎鹰", "天穹", "瀚海", "潜渊", "壁垒", "长城", "密钥",
        "纵深防御", "加固方案", "涉密网络",
    ]

    def __init__(self, assets: Sequence[AssetFact], policy: FieldPolicy):
        self.assets = list(assets)
        self.policy = policy

    def classify_and_match(self, user_input: str) -> Dict[str, Any]:
        """对用户输入做本地检索与子场景分类。"""
        text = self._normalize(user_input)
        matches = self.retrieve(text, top_k=5)

        if not matches:
            return {
                "matched": False,
                "sub_scene": "confidential_general",
                "match_count": 0,
            }

        best = matches[0]

        return {
            "matched": True,
            "sub_scene": best.sub_scene,
            "match_count": len(matches),
            # 只给后端审计使用，不给用户展示
            "audit_asset_ids": [m.asset_id for m in matches],
        }

    def retrieve(self, text: str, top_k: int = 5) -> List[LocalMatch]:
        """本地关键词/内容匹配检索，返回 top_k 结果。"""
        results: List[LocalMatch] = []

        for asset in self.assets:
            score = 0.0
            matched_fields: List[str] = []

            # 字段权重匹配
            for field_name, weight in self._field_weights().items():
                value = self._normalize(asset.get(field_name))
                if not value:
                    continue
                s = self._match_score(text, value)
                if s > 0:
                    score += s * weight
                    matched_fields.append(field_name)

            # keywords 匹配 (extra 中的字段)
            kw_score = self._list_field_score(
                text, asset.extra.get("secret_keywords", []), 3.0
            )
            if kw_score > 0:
                matched_fields.append("secret_keywords")
            score += kw_score

            # attack_paraphrases 匹配
            para_score = self._list_field_score(
                text, asset.extra.get("attack_paraphrases", []), 1.5
            )
            if para_score > 0:
                matched_fields.append("attack_paraphrases")
            score += para_score

            # secret_summary 匹配
            summary_score = (
                self._match_score(text, self._normalize(asset.extra.get("secret_summary", "")))
                * 2.0
            )
            if summary_score > 0:
                matched_fields.append("secret_summary")
            score += summary_score

            if score > 0:
                results.append(
                    LocalMatch(
                        asset_id=asset.id,
                        score=score,
                        matched_fields=sorted(set(matched_fields)),
                        sub_scene=self._infer_sub_scene(asset, text),
                    )
                )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # 安全模板
    # ------------------------------------------------------------------

    @staticmethod
    def get_safe_template(sub_scene: str) -> str:
        """根据子场景返回安全回答模板。"""
        templates: Dict[str, str] = {
            "confidential_project": (
                "系统中存在与该问题相关的涉密项目信息。"
                "根据保密管理要求，项目具体内容、代号、进展、承研单位、经费和人员信息"
                "不能通过对话方式直接提供。请通过授权业务系统按权限查询。"
            ),
            "confidential_system": (
                "系统中存在与该问题相关的涉密系统信息。"
                "根据保密管理要求，系统部署、资产规模、安全评估、整改要求等内容"
                "不能通过对话方式直接提供。请通过授权业务系统按权限查询。"
            ),
            "confidential_personnel": (
                "系统中存在与该问题相关的涉密人员管理信息。"
                "根据保密管理要求，人员资格、任免、脱密期、考试成绩等内容"
                "不能通过对话方式直接提供。请通过授权业务系统按权限查询。"
            ),
            "confidential_finance": (
                "系统中存在与该问题相关的涉密经费或采购信息。"
                "根据保密管理要求，预算、合同金额、采购方式、经费执行情况等内容"
                "不能通过对话方式直接提供。请通过授权业务系统按权限查询。"
            ),
            "confidential_security_audit": (
                "系统中存在与该问题相关的保密检查或安全整改信息。"
                "根据保密管理要求，检查问题、整改要求、系统漏洞和处置结果"
                "不能通过对话方式直接提供。请通过授权业务系统按权限查询。"
            ),
            "confidential_general": (
                "系统中存在与该问题相关的内部保密信息。"
                "根据保密管理要求，具体内容不能通过对话方式直接提供。"
                "请通过授权业务系统按权限查询。"
            ),
        }
        return templates.get(sub_scene, templates["confidential_general"])

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _field_weights(self) -> Dict[str, float]:
        """字段匹配权重。"""
        return {
            "secret_content": 3.0,
            "secret_summary": 2.0,
            "secret_keywords": 1.5,
            "category": 0.5,
            "confidential_level": 0.3,
        }

    def _list_field_score(self, text: str, values: Any, weight: float) -> float:
        if not isinstance(values, (str, list)):
            return 0.0
        if isinstance(values, str):
            return self._match_score(text, self._normalize(values)) * weight
        total = 0.0
        for v in values:
            total += self._match_score(text, self._normalize(str(v))) * weight
        return total

    def _match_score(self, query: str, value: str) -> float:
        """计算 query 与 value 的匹配分数。"""
        if not query or not value:
            return 0.0

        # 精确子串匹配
        if value in query:
            return 3.0
        if query in value and len(query) >= 4:
            return 2.0

        # Token overlap
        q_tokens = self._tokens(query)
        v_tokens = self._tokens(value)
        if not q_tokens or not v_tokens:
            return 0.0
        overlap = q_tokens & v_tokens
        if not overlap:
            return 0.0
        return len(overlap) / max(len(q_tokens), 1)

    def _tokens(self, text: str) -> set[str]:
        """提取文本中的 token。"""
        tokens = set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff★\-]+", text))
        # 中文 ngram 补充
        compact = re.sub(r"\s+", "", text)
        for n in (2, 3, 4):
            for i in range(0, max(0, len(compact) - n + 1)):
                tokens.add(compact[i : i + n])
        return tokens

    def _normalize(self, text: Any) -> str:
        return str(text or "").strip().lower()

    def _infer_sub_scene(self, asset: AssetFact, query: str) -> str:
        """根据资产内容和查询文本推断子场景。"""
        blob = " ".join([
            query,
            asset.get("category"),
            asset.extra.get("secret_content", ""),
            asset.extra.get("secret_summary", ""),
            asset.extra.get("secret_keywords", ""),
        ])

        if any(t in blob for t in self._FINANCE_TERMS):
            return "confidential_finance"
        if any(t in blob for t in self._PERSONNEL_TERMS):
            return "confidential_personnel"
        if any(t in blob for t in self._AUDIT_TERMS):
            return "confidential_security_audit"
        if any(t in blob for t in self._SYSTEM_TERMS):
            return "confidential_system"
        if any(t in blob for t in self._PROJECT_TERMS):
            return "confidential_project"
        return "confidential_general"