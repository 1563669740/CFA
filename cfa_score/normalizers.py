from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from .models import AssetFact, FieldPolicy, NormalizedValue

# Amount normalization patterns
# Matches Chinese RMB amounts like "人民币2.8亿元", "2.8亿元", "2.80亿", "28000万元", "28,000万元"
# Must include unit (亿/万/千) OR "元" suffix OR "人民币" prefix to avoid matching bare numbers
_AMOUNT_PATTERN = re.compile(
    r"(?:人民币|RMB|CNY)?\s*"
    r"(?P<num>\d[\d,，]*\.?\d*)\s*"
    r"(?P<unit>亿|万)\s*"
    r"(?:元|块|CNY|RMB)?"
)
_AMOUNT_PATTERN_YUAN = re.compile(
    r"(?:人民币|RMB|CNY)?\s*"
    r"(?P<num>\d[\d,，]*\.?\d*)\s*"
    r"元"
)
_AMOUNT_PATTERN_WAN = re.compile(
    r"(?P<num>\d[\d,，]+)\s*(?P<unit>万)\s*(?:元)?"
)
_AMOUNT_CLEAN_PATTERN = re.compile(r"[，,]")

# Rate normalization: LPR+120bp, LPR + 120 BP, LPR加120个基点, LPR+120bp（年化4.65%）
_RATE_LPR_PATTERNS = [
    re.compile(
        r"LPR\s*[\+\-加]\s*(?P<bp>\d{2,4})\s*(?:个)?\s*(?:BP|bp|基点|个基点)?"
    ),
]
_RATE_PERCENT_PATTERN = re.compile(
    r"年化\s*(?P<pct>\d+\.?\d*)\s*%"
)

# Credit rating normalization: AA+, AA ＋, 信用评级AA+, 主体评级AA+
_CREDIT_RATING_PATTERN = re.compile(
    r"(?:信用|主体)?评级(?P<rating>[A-C][A-C][+\-－＋﹣]?)"
)
_CREDIT_RATING_STANDALONE = re.compile(
    r"(?<![A-Za-z])(?P<rating>[AB][AB][+\-－＋﹣]?)(?![A-Za-z])"
)

# Date normalization: 2024-10-08, 2024年10月8日
_DATE_PATTERN = re.compile(
    r"(?P<y>\d{4})[年/\-](?P<m>\d{1,2})[月/\-](?P<d>\d{1,2})[日]?"
)

# Collateral component extraction: split by "、" "和" "及" "以及"
_COLLATERAL_SPLITTER = re.compile(r"[、，,和及以及;；]+")


class FieldNormalizer:
    """Normalize field values from natural language into canonical forms.
    
    Supports amount, rate, rating, date, version normalization out of the box.
    Extensible per policy via field_normalizer_rules.
    """

    _UNIT_TO_YUAN: Dict[str, int] = {
        "": 1,
        "万": 10_000,
        "千": 1_000,
        "百": 100,
        "十": 10,
        "亿": 100_000_000,
    }

    def __init__(
        self,
        policy: FieldPolicy,
        assets: Sequence[AssetFact],
    ):
        self._policy = policy
        self._assets = list(assets)
        # Pre-build canonical value index per protected field
        self._canonical_index: Dict[str, set] = {}
        for asset in self._assets:
            for field_name in policy.protected_fields:
                val = _clean_amount(str(asset.get(field_name)))
                if val:
                    self._canonical_index.setdefault(field_name, set()).add(val)

    # ------------------------------------------------------------------
    # Public: extract normalized values from text
    # ------------------------------------------------------------------

    def extract_normalized_anchors(
        self,
        text: str,
        source: str = "output",
    ) -> List[NormalizedValue]:
        """Extract all normalized anchors from text."""
        results: List[NormalizedValue] = []
        seen = set()

        for field_name in self._policy.field_order:
            # Amount normalization
            if self._is_amount_field(field_name):
                for nv in self._normalize_amounts(text, field_name, source):
                    key = (source, field_name, nv.canonical_value, nv.raw_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(nv)

            # Rate normalization  
            if self._is_rate_field(field_name):
                for nv in self._normalize_rates(text, field_name, source):
                    key = (source, field_name, nv.canonical_value, nv.raw_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(nv)

            # Rating normalization
            if self._is_rating_field(field_name):
                for nv in self._normalize_ratings(text, field_name, source):
                    key = (source, field_name, nv.canonical_value, nv.raw_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(nv)

            # Date normalization
            if self._is_date_field(field_name):
                for nv in self._normalize_dates(text, field_name, source):
                    key = (source, field_name, nv.canonical_value, nv.raw_text)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append(nv)

        return results

    # ------------------------------------------------------------------
    # Amount normalization
    # ------------------------------------------------------------------

    def _is_amount_field(self, field_name: str) -> bool:
        """Detect amount-like field by name."""
        lower = field_name.lower()
        amount_keywords = ("amount", "amount", "金额", "金额", "费用", "fee", "salary", "wage")
        return any(kw in lower for kw in amount_keywords)

    def _normalize_amounts(
        self,
        text: str,
        field_name: str,
        source: str,
    ) -> List[NormalizedValue]:
        results: List[NormalizedValue] = []
        # Only match amounts with clear unit suffixes (亿, 万, 元)
        patterns = [
            (_AMOUNT_PATTERN, True),
            (_AMOUNT_PATTERN_YUAN, True),
            (_AMOUNT_PATTERN_WAN, True),
        ]
        seen_spans: set = set()

        for pattern, _ in patterns:
            for m in pattern.finditer(text):
                num_str = _AMOUNT_CLEAN_PATTERN.sub("", m.group("num") or "")
                if not num_str:
                    continue
                try:
                    num = float(num_str)
                except ValueError:
                    continue
                unit = m.group("unit") or ""
                multiplier = self._UNIT_TO_YUAN.get(unit, 1)
                amount_yuan = int(num * multiplier)
                # Require at least 10000 yuan (= 1万) to filter noise
                if amount_yuan < 10000:
                    continue
                canonical = f"{amount_yuan} CNY"
                raw = m.group(0)
                span_key = (m.start(), m.end())
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)
                protected = field_name in self._policy.protected_fields
                results.append(
                    NormalizedValue(
                        field_name=field_name,
                        raw_text=raw,
                        canonical_value=canonical,
                        match_type="normalized",
                        protected=protected,
                        evidence=f"金额归一化: {raw} → {canonical}",
                    )
                )
        return results

    # ------------------------------------------------------------------
    # Rate normalization
    # ------------------------------------------------------------------

    def _is_rate_field(self, field_name: str) -> bool:
        lower = field_name.lower()
        rate_keywords = ("rate", "利率", "利息", "lpr", "bp")
        return any(kw in lower for kw in rate_keywords)

    def _normalize_rates(
        self,
        text: str,
        field_name: str,
        source: str,
    ) -> List[NormalizedValue]:
        results: List[NormalizedValue] = []
        for pattern in _RATE_LPR_PATTERNS:
            for m in pattern.finditer(text):
                bp_str = m.group("bp")
                if not bp_str:
                    continue
                try:
                    bp = int(bp_str)
                except ValueError:
                    continue
                canonical = f"LPR_PLUS_{bp}BP"
                raw = m.group(0)
                protected = field_name in self._policy.protected_fields
                results.append(
                    NormalizedValue(
                        field_name=field_name,
                        raw_text=raw,
                        canonical_value=canonical,
                        match_type="normalized",
                        protected=protected,
                        evidence=f"利率归一化: {raw} → {canonical}",
                    )
                )
        # Also normalize percentage rates like 年化4.65%
        for m in _RATE_PERCENT_PATTERN.finditer(text):
            pct = m.group("pct")
            if not pct:
                continue
            canonical = f"ANNUALIZED_{pct}PCT"
            raw = m.group(0)
            protected = field_name in self._policy.protected_fields
            results.append(
                NormalizedValue(
                    field_name=field_name,
                    raw_text=raw,
                    canonical_value=canonical,
                    match_type="normalized",
                    protected=protected,
                    evidence=f"年化利率归一化: {raw} → {canonical}",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Rating normalization
    # ------------------------------------------------------------------

    def _is_rating_field(self, field_name: str) -> bool:
        lower = field_name.lower()
        rating_keywords = ("rating", "评级", "信用", "等级", "score")
        return any(kw in lower for kw in rating_keywords)

    def _normalize_ratings(
        self,
        text: str,
        field_name: str,
        source: str,
    ) -> List[NormalizedValue]:
        results: List[NormalizedValue] = []
        # Pattern 1: explicit rating context
        for m in _CREDIT_RATING_PATTERN.finditer(text):
            rating = m.group("rating")
            if not rating:
                continue
            rating = _normalize_rating_symbols(rating)
            canonical = rating
            raw = m.group(0)
            protected = field_name in self._policy.protected_fields
            results.append(
                NormalizedValue(
                    field_name=field_name,
                    raw_text=raw,
                    canonical_value=canonical,
                    match_type="normalized",
                    protected=protected,
                    evidence=f"评级归一化: {raw} → {canonical}",
                )
            )
        # Pattern 2: standalone rating
        for m in _CREDIT_RATING_STANDALONE.finditer(text):
            rating = m.group("rating")
            if not rating:
                continue
            rating = _normalize_rating_symbols(rating)
            canonical = rating
            raw = m.group(0)
            protected = field_name in self._policy.protected_fields
            results.append(
                NormalizedValue(
                    field_name=field_name,
                    raw_text=raw,
                    canonical_value=canonical,
                    match_type="normalized",
                    protected=protected,
                    evidence=f"评级归一化(standalone): {raw} → {canonical}",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Date normalization
    # ------------------------------------------------------------------

    def _is_date_field(self, field_name: str) -> bool:
        lower = field_name.lower()
        date_keywords = ("date", "日期", "时间", "申请", "到期", "创建", "更新")
        return any(kw in lower for kw in date_keywords)

    def _normalize_dates(
        self,
        text: str,
        field_name: str,
        source: str,
    ) -> List[NormalizedValue]:
        results: List[NormalizedValue] = []
        for m in _DATE_PATTERN.finditer(text):
            y = m.group("y")
            mo = m.group("m")
            d = m.group("d")
            if not y or not mo or not d:
                continue
            canonical = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
            raw = m.group(0)
            protected = field_name in self._policy.protected_fields
            results.append(
                NormalizedValue(
                    field_name=field_name,
                    raw_text=raw,
                    canonical_value=canonical,
                    match_type="normalized",
                    protected=protected,
                    evidence=f"日期归一化: {raw} → {canonical}",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Collateral component matching
    # ------------------------------------------------------------------

    def match_collateral_components(
        self,
        text: str,
        field_name: str,
    ) -> List[NormalizedValue]:
        """Decompose collateral description into components for partial matching."""
        results: List[NormalizedValue] = []
        # Collect all known collateral values from assets
        known_components: Dict[str, str] = {}
        for asset in self._assets:
            val = str(asset.get(field_name))
            if val:
                for component in _COLLATERAL_SPLITTER.split(val):
                    component = component.strip()
                    if len(component) >= 3:
                        known_components.setdefault(component, val)

        # Match text against known components
        text_lower = text.lower()
        for component, canonical_full in known_components.items():
            if len(component) < 3:
                continue
            # Check both substrings and component pieces
            if component.lower() in text_lower:
                results.append(
                    NormalizedValue(
                        field_name=field_name,
                        raw_text=component,
                        canonical_value=canonical_full,
                        match_type="normalized",
                        protected=field_name in self._policy.protected_fields,
                        evidence=f"抵押物组件命中: '{component}' → '{canonical_full}'",
                    )
                )
                continue
            # Also check sub-components (e.g., "12英寸晶圆产线" matches "产线")
            sub_parts = _COLLATERAL_SPLITTER.split(component)
            for sub in sub_parts:
                sub = sub.strip()
                if len(sub) >= 3 and sub.lower() in text_lower:
                    results.append(
                        NormalizedValue(
                            field_name=field_name,
                            raw_text=sub,
                            canonical_value=canonical_full,
                            match_type="normalized",
                            protected=field_name in self._policy.protected_fields,
                            evidence=f"抵押物子组件命中: '{sub}' → '{canonical_full}'",
                        )
                    )
                    break

        return results


def _normalize_rating_symbols(rating: str) -> str:
    """Normalize rating symbols: ＋ → +, － → -, ﹣ → -"""
    return rating.replace("＋", "+").replace("－", "-").replace("﹣", "-").strip()


def _clean_amount(s: str) -> str:
    """Remove commas and Chinese punctuation from amount strings."""
    return _AMOUNT_CLEAN_PATTERN.sub("", s).strip()