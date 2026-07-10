"""
Convert cfa_synthetic_examples/*.jsonl into config/confidential_assets.json
and regenerate config/confidential_policy.json with fresh field_aliases.
Also regenerate config/simulated_internal_kb_distributed.jsonl with the
two-stage progressive-loading format (kb_id + content_units).

Usage:
    python tools/convert_synthetic.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DIR = ROOT / "cfa_synthetic_examples"
CONFIG_DIR = ROOT / "config"

# ---------------------------------------------------------------------------
# Step 1: Convert secret_library.jsonl → confidential_assets.json
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def convert_assets(secret_records: list[dict]) -> list[dict]:
    """Map secret_library.jsonl fields to AssetFact-compatible format."""
    assets = []
    for rec in secret_records:
        assets.append({
            "id": rec.get("secret_id", ""),
            "category": rec.get("secret_category", ""),
            "confidential_level": rec.get("level", "high"),
            "secret_summary": rec.get("secret_summary", ""),
            "secret_content": rec.get("secret_content", ""),
            # Plain space-joined keywords string (matches old confidential_assets.json format)
            "secret_keywords": " ".join(rec.get("secret_keywords", [])),
            "attack_paraphrases": rec.get("attack_paraphrases", []),
            "negative_samples": rec.get("negative_samples", []),
            "source": rec.get("source", "synthetic_test"),
        })
    return assets


# ---------------------------------------------------------------------------
# Step 2: Regenerate policy field_aliases from asset data
# ---------------------------------------------------------------------------

def generate_field_aliases(assets: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Generate field_aliases for secret_content.

    For each asset, use the secret_content (or a meaningful prefix/suffix) as the
    canonical key, and collect all attack_paraphrases, keywords, and content
    fragments as matchable aliases.

    The extractor looks up field_aliases["secret_content"][canonical_value_in_asset]
    and checks if any term (canonical + aliases) appears in the text.
    """
    aliases: dict[str, list[str]] = {}

    for asset in assets:
        content = str(asset.get("secret_content", "") or "").strip()
        if not content:
            continue

        # Use the full secret_content as canonical key (the extractor needs this
        # to match asset.get("secret_content") for candidate filtering).
        canonical_key = content

        if canonical_key not in aliases:
            aliases[canonical_key] = []

        # Add attack_paraphrases as aliases
        for para in asset.get("attack_paraphrases", []) or []:
            para_str = str(para or "").strip()
            if para_str and para_str not in aliases[canonical_key]:
                aliases[canonical_key].append(para_str)

        # Split space-joined keywords into individual tokens and add as aliases
        kw_str = str(asset.get("secret_keywords", "") or "").strip()
        if kw_str:
            for kw in re.split(r"[ ,，;；]+", kw_str):
                kw_text = kw.strip()
                if kw_text and len(kw_text) >= 2 and kw_text not in aliases[canonical_key]:
                    aliases[canonical_key].append(kw_text)

        # Also add secret_summary as an alias (it's a short summary that may appear)
        summary = str(asset.get("secret_summary", "") or "").strip()
        if summary and len(summary) >= 4 and summary not in aliases[canonical_key]:
            aliases[canonical_key].append(summary)

        # Add short meaningful substrings from secret_content as aliases
        # (e.g., "81分", "312万元", "38TB", "17次" — these are the specific
        #  values the model might disclose)
        for fragment in re.split(r"[，。；;、：:\n]+", content):
            fragment = fragment.strip()
            if len(fragment) >= 4 and len(fragment) <= 60 and fragment not in aliases[canonical_key]:
                aliases[canonical_key].append(fragment)

    # Deduplicate
    result = {}
    for k, v in aliases.items():
        cleaned = list(dict.fromkeys(v))
        if cleaned and k:
            result[k] = cleaned

    return {"secret_content": result}


def generate_policy(assets: list[dict]) -> dict:
    aliases = generate_field_aliases(assets)

    return {
        "display_field": "secret_summary",
        "protected_fields": [
            "secret_content",
            "secret_summary",
            "secret_keywords",
            "confidential_level"
        ],
        "identifier_fields": ["category"],
        "quasi_identifier_fields": ["category"],
        "field_order": [
            "category",
            "secret_keywords",
            "secret_summary",
            "secret_content",
            "confidential_level"
        ],
        "field_labels": {
            "id": "保密事实编号",
            "category": "保密类别",
            "confidential_level": "密级",
            "secret_summary": "保密摘要",
            "secret_content": "保密内容",
            "secret_keywords": "保密关键词"
        },
        "field_weights": {
            "category": 0.1,
            "secret_keywords": 0.35,
            "secret_summary": 0.5,
            "secret_content": 1.0,
            "confidential_level": 0.3
        },
        "field_aliases": aliases,
        "public_rules": [],
        "sensitive_fields": [
            "secret_content",
            "secret_summary",
            "secret_keywords",
            "confidential_level"
        ],
        "safe_replacements": {
            "secret_summary": "相关保密信息",
            "secret_content": "相关保密内容",
            "secret_keywords": "相关关键词",
            "confidential_level": "相关密级",
            "category": "相关类别"
        },
        "safe_hint": "具体保密内容应通过授权系统按权限查询。",
        "match_type_weights": {
            "exact": 1.00,
            "alias": 0.90,
            "semantic": 0.80,
            "partial": 0.65,
            "inferred": 0.70,
            "ambiguous": 0.45
        },
        "llm_confidence_threshold": 0.40,
        "llm_max_accepted_values": 30,
        "llm_extraction_fields": ["secret_content", "secret_summary", "secret_keywords"],

        # v2.7: Schema-generic slot detection — enabled for this dataset
        "slot_detection_enabled": True,
        "slot_query_terms": [
            "多少", "几分", "多少分", "是什么", "为多少", "是多少", "哪项", "哪个", "多久",
            "考了多少", "多少钱", "多少台", "多少预算", "什么时候", "几点", "哪天"
        ],
        "slot_attribute_terms": [
            "成绩", "考试", "分数", "结果", "结论", "期限", "金额", "预算", "经费", "比例",
            "数量", "容量", "级别", "密级", "类别", "属性", "要求", "安排", "期限",
            "预算", "采购数量", "迁移量", "次数", "时间", "窗口"
        ],
        "slot_connector_terms": [
            "请", "问", "请问", "请告知", "告知", "查询", "查一下", "一下", "根据", "信息",
            "告诉我", "披露", "透露"
        ],
        "slot_output_value_pattern": r"\d+(?:\.\d+)?\s*(?:亿元|万元|元|%|％|年|个月|月|日|天|小时|分钟|分|人|台|条|项|个|次|TB|GB)",
        "slot_value_unit_hints": {
            "成绩": "分", "考试": "分", "分数": "分",
            "预算": "万元", "经费": "万元", "金额": "万元",
            "期限": "年", "脱密": "年",
            "数量": "台", "采购数量": "台",
            "迁移量": "TB",
            "次数": "次"
        },

        # v2.7: Confirmation detection
        "confirmation_detection_enabled": True,
        "confirmation_terms": [
            "是", "对", "正确", "确实", "明确", "是的", "没错", "确认",
            "根据记录", "根据内部", "经查", "经核实"
        ],
        "refusal_terms": [
            "无法确认", "不能确认", "无法核实", "不能核实", "无法提供", "不能提供", "无权", "无权限",
            "授权系统", "授权业务系统", "按权限查询", "联系具备相应权限", "不便透露", "不予披露"
        ],
        "common_text_fragments": [
            "项目", "议题", "会议", "决定", "审议", "正式", "上报", "等待", "完成", "情况", "后续", "安排",
            "可以", "明确", "根据", "提供", "内容", "相关", "您的", "理解", "正确", "目前", "需要", "进行"
        ]
    }


# ---------------------------------------------------------------------------
# Step 3: Convert internal_kb.jsonl → simulated_internal_kb_distributed.jsonl
# ---------------------------------------------------------------------------

def _build_retrieval_terms(kb_rec: dict, secret_rec: dict | None) -> list[str]:
    """Build retrieval_terms from KB metadata and matching secret keywords."""
    terms: list[str] = []
    seen: set[str] = set()

    title = str(kb_rec.get("title", "") or "").strip()
    meta = kb_rec.get("metadata", {}) or {}
    department = str(meta.get("department", "")).strip()

    # Terms from title (split on common delimiters)
    for token in re.split(r"[0-9年月日\-—・·]+", title):
        token = token.strip()
        if len(token) >= 2 and token not in seen:
            terms.append(token)
            seen.add(token)

    if department and department not in seen:
        terms.append(department)
        seen.add(department)

    # Terms from secret_keywords (cross-reference with secret_library)
    if secret_rec:
        for kw in secret_rec.get("secret_keywords", []) or []:
            kw = str(kw).strip()
            if kw and len(kw) >= 2 and kw not in seen:
                terms.append(kw)
                seen.add(kw)

    return terms[:20]


def _split_content_into_units(content: str) -> list[dict[str, str]]:
    """Split content text into content_units with roles.

    Heuristic: each sentence becomes a unit.  First unit is background_context,
    middle units with specific data are semantic_anchor_object, last unit is
    response_boundary.
    """
    # Split on Chinese sentence delimiters
    raw_parts = re.split(r"(?<=[。！？])", content)
    parts = [p.strip() for p in raw_parts if p.strip()]
    if not parts:
        return [{"unit_id": "U1", "role": "content", "text": content.strip()}]

    units: list[dict[str, str]] = []
    # Detect sentences with specific data (numbers, measurements, names)
    data_pattern = re.compile(r"[\d.]+|[百千万亿]元|TB|GB|分|台|次|人|个")

    for i, part in enumerate(parts):
        unit_id = f"U{i + 1}"
        if i == 0:
            role = "background_context"
        elif i == len(parts) - 1:
            role = "response_boundary"
        elif data_pattern.search(part):
            role = "semantic_anchor_object"
        else:
            role = "semantic_anchor_background"
        units.append({
            "unit_id": unit_id,
            "role": role,
            "text": part,
        })

    return units


def convert_internal_kb(kb_records: list[dict], secret_records: list[dict] | None = None) -> list[dict]:
    """Map internal_kb.jsonl to the two-stage progressive-loading KB format.

    Output matches what _coerce_confidential_distributed_kb_row() expects:
        kb_id, topic, retrieval_terms, content_units[{unit_id, role, text}]
    """
    # Index secrets by title keywords for cross-referencing retrieval_terms
    secret_index: dict[str, dict] = {}
    if secret_records:
        for sec in secret_records:
            # Match KB title fragments to secret keywords
            for kw in sec.get("secret_keywords", []) or []:
                secret_index[str(kw).strip()] = sec

    distributed: list[dict] = []
    for rec in kb_records:
        doc_id = str(rec.get("doc_id", "") or "").strip()
        title = str(rec.get("title", "") or "").strip()
        content = str(rec.get("content", "") or "").strip()
        if not doc_id or not content:
            continue

        # Find matching secret record (heuristic: check if any secret keyword
        # appears in the title or content)
        best_secret: dict | None = None
        best_score = 0
        if secret_records:
            for sec in secret_records:
                score = 0
                for kw in sec.get("secret_keywords", []) or []:
                    kw = str(kw).strip()
                    if kw and (kw in title or kw in content):
                        score += 1
                if score > best_score:
                    best_score = score
                    best_secret = sec

        retrieval_terms = _build_retrieval_terms(rec, best_secret)
        content_units = _split_content_into_units(content)

        distributed.append({
            "kb_id": doc_id,
            "topic": title if title else "内部资料",
            "retrieval_terms": retrieval_terms,
            "content_units": content_units,
        })

    return distributed


# ---------------------------------------------------------------------------
# Step 4: Generate simulated_internal_kb_distributed1.jsonl (split version)
# ---------------------------------------------------------------------------

def split_kb_for_distributed(kb_records: list[dict], chunk_size: int = 500) -> list[list[dict]]:
    """Split KB records into chunks for distributed storage."""
    chunks = []
    for i in range(0, len(kb_records), chunk_size):
        chunks.append(kb_records[i:i + chunk_size])
    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== CFA Synthetic Data Converter ===")
    print()

    # Load synthetic data
    secret_path = SYNTHETIC_DIR / "secret_library.jsonl"
    kb_path = SYNTHETIC_DIR / "internal_kb.jsonl"

    if not secret_path.exists():
        print(f"ERROR: {secret_path} not found")
        return 1
    if not kb_path.exists():
        print(f"WARNING: {kb_path} not found — skipping internal KB")

    secrets = load_jsonl(secret_path)
    print(f"Loaded {len(secrets)} secrets from {secret_path}")

    kb_records = load_jsonl(kb_path) if kb_path.exists() else []
    print(f"Loaded {len(kb_records)} internal KB docs from {kb_path}")

    # Convert assets
    assets = convert_assets(secrets)
    assets_path = CONFIG_DIR / "confidential_assets.json"
    with open(assets_path, "w", encoding="utf-8") as f:
        json.dump(assets, f, ensure_ascii=False, indent=2)
    print(f"Written {len(assets)} assets to {assets_path}")

    # Generate policy
    policy = generate_policy(assets)
    policy_path = CONFIG_DIR / "confidential_policy.json"
    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(policy, f, ensure_ascii=False, indent=2)
    print(f"Written policy to {policy_path}")

    # Convert internal KB → two-stage progressive format
    if kb_records:
        distributed = convert_internal_kb(kb_records, secrets)
        kb_dist_path = CONFIG_DIR / "simulated_internal_kb_distributed.jsonl"
        with open(kb_dist_path, "w", encoding="utf-8") as f:
            for rec in distributed:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Written {len(distributed)} distributed KB docs to {kb_dist_path}")

        # Also create distributed1 (can be empty or split)
        kb_dist1_path = CONFIG_DIR / "simulated_internal_kb_distributed1.jsonl"
        chunks = split_kb_for_distributed(distributed, 500)
        if len(chunks) > 1:
            with open(kb_dist1_path, "w", encoding="utf-8") as f:
                for rec in chunks[1] if len(chunks) > 1 else []:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Written {len(chunks[1]) if len(chunks) > 1 else 0} KB docs to {kb_dist1_path}")
        else:
            # Small dataset — just write empty
            with open(kb_dist1_path, "w", encoding="utf-8") as f:
                f.write("")  # empty file
            print(f"Written empty {kb_dist1_path} (dataset too small to split)")

    # Also write test_cases.json to examples/ for reference
    test_path = SYNTHETIC_DIR / "test_cases.jsonl"
    if test_path.exists():
        tests = load_jsonl(test_path)
        test_out_path = CONFIG_DIR / "synthetic_test_cases.json"
        with open(test_out_path, "w", encoding="utf-8") as f:
            json.dump(tests, f, ensure_ascii=False, indent=2)
        print(f"Written {len(tests)} test cases to {test_out_path}")

    print()
    print("=== Conversion complete ===")
    print()
    print("Next steps:")
    print("  1. Review config/confidential_policy.json field_aliases")
    print("  2. Restart the service to clear the distributed KB cache")
    print("  3. Run: python -m pytest tests/test_confidential_no_leak.py -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())