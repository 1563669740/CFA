"""
Hybrid sparse retriever for CFA-Score semantic indexing.

Replaces the old alias-only candidate recall with a fused BM25 + Chinese
char n-gram + alias + field hint scoring pipeline.

All dependencies are Python stdlib only.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .models import AssetFact, CandidateValue, FieldPolicy, SemanticFieldAlias

# ============================================================================
# Text normalisation and tokenisation
# ============================================================================

_RE_WHITESPACE = re.compile(r"\s+")
_RE_CHINESE_CHAR = re.compile(r"[\u4e00-\u9fff]+")
_RE_ENGLISH_WORD = re.compile(r"[a-zA-Z]+")
_RE_NUMBER_LIKE = re.compile(r"\d+(?:\.\d+)*")
_RE_TECH_TOKEN = re.compile(r"[a-zA-Z0-9_.-]+")


def normalize_text(text: str) -> str:
    """Normalise text for matching.

    1. NFKC normalisation (fullwidth → halfwidth etc.)
    2. Lowercase English characters.
    3. Greek beta → ASCII beta.
    4. Fullwidth plus → ASCII plus.
    5. Collapse multiple whitespace into a single space.
    6. Version strings like 5.6.1 and CVE-2017-0144 are preserved.
    7. Chinese characters are NOT removed.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = text.replace("\u03b2", "beta")      # β
    text = text.replace("\uff0b", "+")          # ＋
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def char_ngrams(text: str, ns: Tuple[int, ...] = (2, 3)) -> List[str]:
    """Generate Chinese character n-grams from text.

    Only continuous Chinese character runs are n-gram'd.
    """
    result: List[str] = []
    for match in _RE_CHINESE_CHAR.finditer(text):
        run = match.group()
        for n in ns:
            if len(run) >= n:
                for i in range(len(run) - n + 1):
                    result.append(run[i:i + n])
    return result


def word_tokens(text: str) -> List[str]:
    """Tokenise text into English/technical/number tokens + Chinese char n-grams.

    Returns a list of lowercase, normalised tokens.
    """
    if not text:
        return []
    text = normalize_text(text)

    tokens: List[str] = []

    # Extract Chinese runs separately for n-gram generation
    chinese_runs: List[str] = []
    for match in _RE_CHINESE_CHAR.finditer(text):
        chinese_runs.append(match.group())

    # English words
    for match in _RE_ENGLISH_WORD.finditer(text):
        tokens.append(match.group())

    # Number-like sequences
    for match in _RE_NUMBER_LIKE.finditer(text):
        val = match.group()
        # Only add if not already covered by tech-token extraction
        if val not in tokens:
            tokens.append(val)

    # Tech tokens (includes hyphenated like xz-utils)
    for match in _RE_TECH_TOKEN.finditer(text):
        val = match.group()
        # Skip pure-number tokens already covered
        if _RE_NUMBER_LIKE.fullmatch(val):
            continue
        # Skip single-char tokens that are just digits or letters
        if len(val) == 1:
            continue
        if val not in tokens:
            tokens.append(val)

    # Chinese char n-grams
    for run in chinese_runs:
        tokens.extend(_char_ngrams_for_run(run, (2, 3)))

    return tokens


def _char_ngrams_for_run(run: str, ns: Tuple[int, ...] = (2, 3)) -> List[str]:
    result: List[str] = []
    for n in ns:
        if len(run) >= n:
            for i in range(len(run) - n + 1):
                result.append(run[i:i + n])
    return result


# ============================================================================
# Candidate document
# ============================================================================

@dataclass
class CandidateDocument:
    """A retrievable document for one (field_name, canonical_value)."""
    field_name: str
    canonical_value: str
    text: str
    tokens: List[str]
    token_counts: Counter                                    # Counter of token frequencies
    length: int                                              # Number of tokens
    source_terms: Dict[str, List[str]] = field(default_factory=dict)  # debug info


# ============================================================================
# Field hint boost table
# ============================================================================

CONFIDENTIAL_TEXT_FIELDS = {
    "secret_content",
    "secret_summary",
    "secret_keywords",
    "confidential_level",
    "attack_paraphrases",
}


def is_weak_confidential_value(field_name: str, value: str) -> bool:
    if field_name not in CONFIDENTIAL_TEXT_FIELDS:
        return False
    compact = normalize_text(str(value or ""))
    compact = re.sub(r"[\s\W_]+", "", compact, flags=re.UNICODE)
    if not compact:
        return True
    if len(compact) < 4:
        return True
    return False


FIELD_HINTS: Dict[str, List[str]] = {
    "medication":       ["用药", "治疗", "处方", "药物", "双抗", "抗血小板",
                         "beta受体阻滞剂", "替格瑞洛", "阿司匹林", "华法林"],
    "diagnosis":        ["诊断", "病情", "疾病", "心梗", "脑卒中", "冠心病",
                         "心房颤动", "心衰", "肿瘤", "骨折"],
    "insurance_level":  ["医保", "报销", "自费", "城镇职工", "城镇居民", "商业保险"],
    "ward_type":        ["病房", "icu", "ccu", "重症", "特需", "普通病房"],
    "department":       ["科室", "心内科", "神经内科", "外科", "肿瘤科", "骨科"],
    "condition_summary":["心梗", "胸痛", "心悸", "脑卒中", "化疗", "骨折", "心衰", "肿瘤"],
    "patient_name":     ["患者", "病人", "病例"],
    "loan_amount":      ["金额", "贷款", "万元", "亿元", "人民币"],
    "interest_rate":    ["利率", "lpr", "bp", "年化"],
    "credit_rating":    ["评级", "信用", "aa", "bbb", "a级", "b级"],
    "collateral":       ["抵押", "担保", "质押", "研发大楼", "晶圆产线"],
    "company_name":     ["公司", "企业", "有限", "科技", "新能源", "数据服务"],
    "industry":         ["行业", "半导体", "光伏", "医疗", "数据中心", "房地产", "装备制造"],
    "loan_type":        ["贷款", "流动资金", "项目", "固定资产", "并购", "技改"],
    "branch":           ["分行", "上海", "杭州", "深圳", "南京"],
    "component_version":["版本", "组件", "cve", "漏洞", "xz", "utils", "5.6"],
    "risk_status":      ["风险", "高危", "暴露", "远程代码执行", "恶意代码"],
    "disposition_status":["处置", "修复", "回滚", "未完成", "已完成", "已隔离"],
    "remote_entry":     ["入口", "ssh", "运维", "远程"],
    "system_name":      ["系统", "平台", "服务", "链路"],
    "function_category":["监测", "链路", "健康"],
    "meeting_topic":    ["会议", "议题", "主题"],
    "meeting_type":     ["闭门", "评审", "谈判", "验收", "例会", "复查"],
    "room":             ["会议室", "b-307", "a-306", "c-201", "d-101"],
    "time_slot":        ["周四", "周五", "周三", "上午", "下午", "15:30", "16:00"],
    "participants":     ["供应商", "乙方", "合作方", "内部", "员工"],
    "material_count":   ["材料", "份"],
    "restricted_content":["纪要", "内容", "涉密"],
    "confidential":     ["涉密", "机密", "保密", "true"],
}


def _field_hint_score(text_norm: str, field_name: str) -> float:
    """Lightweight boost: scan text for field-specific hint tokens.

    Returns 0.0-1.0.
    """
    hints = FIELD_HINTS.get(field_name, [])
    if not hints:
        return 0.0
    matched = sum(1 for h in hints if h in text_norm)
    if matched >= 2:
        return 1.0
    elif matched == 1:
        return 0.5
    return 0.0


# ============================================================================
# Smoothing function
# ============================================================================

def _squash(score: float) -> float:
    """Map [0, +inf) → [0, 1] smoothly."""
    if score <= 0:
        return 0.0
    return score / (score + 1.0)


# ============================================================================
# Hybrid Sparse Retriever
# ============================================================================

DEFAULT_MIN_SCORE = 0.08
DEFAULT_TOP_K = 40
DEFAULT_MAX_PER_FIELD = 8
BM25_K1 = 1.2
BM25_B = 0.75


class HybridSparseRetriever:
    """BM25 + Chinese char n-gram + alias + field-hint hybrid retrieval.

    Builds a sparse index from (field_name, canonical_value) pairs derived from
    the fact pool and semantic aliases, then scores candidates against a query
    using a weighted fusion of four signals.
    """

    def __init__(
        self,
        policy: FieldPolicy,
        assets: Sequence[AssetFact],
        alias_lookup: Dict[Tuple[str, str], SemanticFieldAlias],
    ):
        self._policy = policy
        self._assets = list(assets)
        self._alias_lookup = alias_lookup

        # Build candidate documents
        self._docs: Dict[Tuple[str, str], CandidateDocument] = {}
        self._doc_list: List[CandidateDocument] = []
        self._build_documents()

        # Pre-compute BM25 statistics
        self._doc_freqs: Counter = Counter()
        self._avgdl: float = 0.0
        self._idf_cache: Dict[str, float] = {}
        self._compute_statistics()

        # Recent stats
        self.last_stats: Dict[str, Any] = {
            "query_length": 0,
            "candidate_count": 0,
            "empty_candidate": True,
            "top_fields": [],
            "max_score": 0.0,
            "avg_score": 0.0,
            "source": "hybrid_sparse",
        }

    # ------------------------------------------------------------------
    # Document construction
    # ------------------------------------------------------------------

    def _build_documents(self) -> None:
        """Construct one CandidateDocument per unique (field_name, canonical_value)."""
        seen: set = set()
        # Only use fields that are in the policy field_order
        for field_name in self._policy.field_order:
            field_label = self._policy.label(field_name)
            # Collect all distinct values from the fact pool
            for asset in self._assets:
                value = asset.get(field_name)
                if not value:
                    continue
                if is_weak_confidential_value(field_name, value):
                    continue
                key = (field_name, value)
                if key in seen:
                    continue
                seen.add(key)

                # Semantic alias info for this value
                alias_info = self._alias_lookup.get(key, SemanticFieldAlias())
                aliases = list(alias_info.aliases)
                components = list(alias_info.components)
                partial_clues = list(alias_info.partial_clues)

                # Build document text
                parts = [field_name, field_label, value]
                parts.extend(aliases)
                parts.extend(components)
                parts.extend(partial_clues)
                doc_text = " | ".join(str(p) for p in parts if p)

                # Tokenise
                tokens = word_tokens(doc_text)
                token_counts = Counter(tokens)

                doc = CandidateDocument(
                    field_name=field_name,
                    canonical_value=value,
                    text=doc_text,
                    tokens=tokens,
                    token_counts=token_counts,
                    length=len(tokens),
                    source_terms={
                        "field_name": [field_name],
                        "field_label": [field_label],
                        "canonical_value": [value],
                        "aliases": aliases,
                        "components": components,
                        "partial_clues": partial_clues,
                    },
                )
                self._docs[key] = doc
                self._doc_list.append(doc)

    def _compute_statistics(self) -> None:
        """Compute document frequencies, average document length, IDF cache."""
        N = len(self._doc_list)
        if N == 0:
            self._avgdl = 0.0
            return

        total_length = 0
        for doc in self._doc_list:
            total_length += doc.length
            for token in doc.token_counts:
                self._doc_freqs[token] += 1

        self._avgdl = total_length / N

    def _idf(self, token: str) -> float:
        """Compute IDF for a token. Cached after first computation."""
        if token in self._idf_cache:
            return self._idf_cache[token]
        N = len(self._doc_list)
        df = self._doc_freqs.get(token, 0)
        if df == 0 or N == 0:
            val = 0.1  # default for unseen tokens
        else:
            val = math.log(1 + (N - df + 0.5) / (df + 0.5))
        self._idf_cache[token] = val
        return val

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _bm25_score(self, query_tokens: List[str], doc: CandidateDocument) -> float:
        """Compute BM25 score for a single document."""
        if self._avgdl == 0:
            return 0.0
        score = 0.0
        unique_tokens = set(query_tokens)
        for token in unique_tokens:
            tf = doc.token_counts.get(token, 0)
            if tf == 0:
                continue
            idf_val = self._idf(token)
            numerator = tf * (BM25_K1 + 1)
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc.length / self._avgdl)
            score += idf_val * numerator / denominator
        return score

    # ------------------------------------------------------------------
    # Chinese n-gram scoring
    # ------------------------------------------------------------------

    def _ngram_score(self, query_tokens: List[str], doc: CandidateDocument) -> float:
        """Compute overlap-based n-gram score (0-1)."""
        q_set = set(query_tokens)
        if not q_set:
            return 0.0
        d_set = set(doc.tokens)
        overlap = q_set & d_set
        if not overlap:
            return 0.0
        idf_sum_overlap = sum(self._idf(t) for t in overlap)
        idf_sum_query = sum(self._idf(t) for t in q_set)
        if idf_sum_query == 0:
            return 0.0
        return idf_sum_overlap / idf_sum_query

    # ------------------------------------------------------------------
    # Alias / component / partial_clue matching
    # ------------------------------------------------------------------

    def _alias_score(
        self, text_norm: str, doc: CandidateDocument
    ) -> Tuple[float, List[str]]:
        """Compute alias score and collect matched terms.

        Weights (additive):
            canonical_value in text  → +3.5
            alias in text            → +3.0
            component in text        → +2.0
            partial_clue in text     → +1.0
        """
        raw_score = 0.0
        matched: List[str] = []

        source = doc.source_terms

        # canonical_value match
        for cv in source.get("canonical_value", []):
            if is_weak_confidential_value(doc.field_name, cv):
                continue
            if cv and cv in text_norm:
                raw_score += 3.5
                matched.append(f"cv:{cv}")

        # alias match
        for alias in source.get("aliases", []):
            if is_weak_confidential_value(doc.field_name, alias):
                continue
            if alias and alias in text_norm:
                raw_score += 3.0
                matched.append(f"alias:{alias}")

        # component match
        for comp in source.get("components", []):
            if comp and comp in text_norm:
                raw_score += 2.0
                matched.append(f"comp:{comp}")

        # partial clue match
        for clue in source.get("partial_clues", []):
            if clue and clue in text_norm:
                raw_score += 1.0
                matched.append(f"clue:{clue}")

        return raw_score, matched

    # ------------------------------------------------------------------
    # Score fusion
    # ------------------------------------------------------------------

    def _fused_score(
        self,
        text_norm: str,
        query_tokens: List[str],
        doc: CandidateDocument,
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """Compute the fused retrieval score for one document."""
        alias_raw, matched_terms = self._alias_score(text_norm, doc)
        bm25_raw = self._bm25_score(query_tokens, doc)
        ngram_val = self._ngram_score(query_tokens, doc)
        hint_val = _field_hint_score(text_norm, doc.field_name)

        # Normalise
        alias_norm = _squash(alias_raw)
        bm25_norm = _squash(bm25_raw)

        final = (
            0.45 * alias_norm
            + 0.30 * bm25_norm
            + 0.20 * ngram_val
            + 0.05 * hint_val
        )

        breakdown = {
            "alias_raw": alias_raw,
            "alias_norm": alias_norm,
            "bm25_raw": bm25_raw,
            "bm25_norm": bm25_norm,
            "ngram": ngram_val,
            "field_hint": hint_val,
        }
        return final, breakdown, matched_terms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        max_per_field: int = DEFAULT_MAX_PER_FIELD,
        min_score: float = DEFAULT_MIN_SCORE,
    ) -> List[CandidateValue]:
        """Retrieve candidate (field_name, canonical_value) pairs for `text`.

        Returns candidates sorted by fused score, subject to per-field and
        global cut-offs.
        """
        if not text or not self._doc_list:
            self.last_stats = {
                "query_length": len(text) if text else 0,
                "candidate_count": 0,
                "empty_candidate": True,
                "top_fields": [],
                "max_score": 0.0,
                "avg_score": 0.0,
                "source": "hybrid_sparse",
            }
            return []

        text_norm = normalize_text(text)
        query_tokens = word_tokens(text)

        # Score every document
        scored: List[Tuple[CandidateValue, float]] = []
        for doc in self._doc_list:
            final_score, breakdown, matched_terms = self._fused_score(
                text_norm, query_tokens, doc
            )
            if final_score < min_score:
                continue
            candidate = CandidateValue(
                field_name=doc.field_name,
                canonical_value=doc.canonical_value,
                score=round(final_score, 4),
                source="hybrid_sparse",
                matched_terms=matched_terms,
                score_breakdown=breakdown,
            )
            scored.append((candidate, final_score))

        # Sort: highest score first
        scored.sort(key=lambda x: x[1], reverse=True)

        # Per-field truncation
        field_counts: Dict[str, int] = {}
        truncated: List[CandidateValue] = []
        for candidate, _ in scored:
            fn = candidate.field_name
            if field_counts.get(fn, 0) >= max_per_field:
                continue
            field_counts[fn] = field_counts.get(fn, 0) + 1
            truncated.append(candidate)

        # Global top-k
        result = truncated[:top_k]

        # Update stats
        scores = [c.score for c in result]
        field_list = sorted(
            set(c.field_name for c in result),
            key=lambda fn: sum(1 for c in result if c.field_name == fn),
            reverse=True,
        )[:10]

        self.last_stats = {
            "query_length": len(text),
            "candidate_count": len(result),
            "empty_candidate": len(result) == 0,
            "top_fields": field_list,
            "max_score": max(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "source": "hybrid_sparse",
        }
        return result