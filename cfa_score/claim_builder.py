from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .models import Anchor, AssetFact, DisclosureClaim, FieldPolicy, NormalizedValue

# Sentence splitting pattern for claim boundary detection
# Splits on: periods, semicolons, newlines, Chinese commas in list-like contexts
_CLAIM_SPLIT_PATTERN = re.compile(r"[。；;！!？?\n]+")
# Pattern for detecting multi-record indicators (list numbering, bullet points)
_LIST_ITEM_PATTERN = re.compile(r"(?:^|\n)\s*(?:\d+[.、．]|[（(]\d+[)）]|[-•●◆■▪▸►])")
# Pattern for sentence-level splitting within a claim body
_SENTENCE_PATTERN = re.compile(r"[，,、]+")
# Fields that indicate different records when multiple values appear in one output
# These are detected from policy.record_discriminator_fields + heuristics
_DISCRIMINATOR_HEURISTICS = {
    "loan_type", "loan_amount", "interest_rate", "collateral",
    "diagnosis", "medication", "treatment",
    "meeting_time", "meeting_title", "participant",
    "asset_type", "asset_value", "vulnerability",
}


class ClaimBuilder:
    """Split model output into independent disclosure claims.

    Design principles:
    1. Each claim represents one local fact disclosure
    2. Context anchors are shared across claims, not used to merge them
    3. Record discriminator fields with multiple values trigger claim splitting
    4. Claims are UNION-ed for final risk assessment (no global AND)
    """

    def __init__(
        self,
        policy: FieldPolicy,
        assets: Sequence[AssetFact],
    ):
        self._policy = policy
        self._assets = list(assets)

    # ------------------------------------------------------------------
    # Main entry: build claims from anchors + text
    # ------------------------------------------------------------------

    def build_claims(
        self,
        model_output: str,
        anchors: Sequence[Anchor],
        user_input: str = "",
    ) -> List[DisclosureClaim]:
        """Build DisclosureClaims from model output and extracted anchors."""
        if not model_output or not model_output.strip():
            return []

        all_anchors = list(anchors)
        output_anchors = [a for a in all_anchors if a.source == "output" and not a.inferred]
        input_anchors = [a for a in all_anchors if a.source == "input" and not a.inferred]

        # Identify context anchors (entity_fields + context_fields heuristic)
        context_anchors = self._identify_context_anchors(input_anchors)

        # Step 1: Try claim splitting by sentence boundaries first
        segments = self._split_into_segments(model_output)

        # Step 2: If multi-record conflict detected, refine splitting
        if self._has_multi_record_conflict(output_anchors):
            segments = self._refine_split_by_discriminator(model_output, output_anchors)

        # Step 3: Build claims from segments
        claims: List[DisclosureClaim] = []
        for i, (start, end, text) in enumerate(segments):
            local_anchors = self._filter_anchors_in_range(output_anchors, start, end, text)
            if not local_anchors:
                # Segments without any output anchors can be skipped
                # unless they contain direct protected disclosures
                if not self._contains_protected_text(text):
                    continue

            claim_id = self._make_claim_id(f"claim_{i}", text, start, end)
            claim = DisclosureClaim(
                claim_id=claim_id,
                text=text.strip(),
                start=start,
                end=end,
                context_anchors=list(context_anchors),
                local_output_anchors=local_anchors,
            )
            claims.append(claim)

        # Step 4: Deduplicate claims that share the same discriminator values
        claims = self._deduplicate_claims(claims)

        # Step 5: Bind candidate assets to each claim
        for claim in claims:
            candidates = self._bind_candidates(claim)
            claim.candidate_assets = candidates
            claim.candidate_asset_ids = [a.id for a in candidates]

        return claims

    # ------------------------------------------------------------------
    # Claim splitting
    # ------------------------------------------------------------------

    def _split_into_segments(
        self,
        text: str,
    ) -> List[Tuple[int, int, str]]:
        """Split text into claim segments by sentence boundaries."""
        segments: List[Tuple[int, int, str]] = []

        # First try list item splitting
        list_positions = [m.start() for m in _LIST_ITEM_PATTERN.finditer(text)]
        if len(list_positions) >= 2:
            # Split by list items
            for i, pos in enumerate(list_positions):
                next_pos = list_positions[i + 1] if i + 1 < len(list_positions) else len(text)
                seg_text = text[pos:next_pos].strip()
                if seg_text:
                    segments.append((pos, next_pos, seg_text))
            return segments

        # Split by major boundaries
        split_positions = [m.start() for m in _CLAIM_SPLIT_PATTERN.finditer(text)]
        if not split_positions:
            return [(0, len(text), text)]

        # Build segments between split positions
        prev_end = 0
        for pos in split_positions:
            seg_text = text[prev_end:pos].strip()
            if seg_text:
                segments.append((prev_end, pos, seg_text))
            prev_end = pos + 1  # skip the delimiter

        # Last segment after final delimiter
        if prev_end < len(text):
            seg_text = text[prev_end:].strip()
            if seg_text:
                segments.append((prev_end, len(text), seg_text))

        # If no segments produced (empty text between delimiters), return whole
        if not segments:
            return [(0, len(text), text)]

        return segments

    def _refine_split_by_discriminator(
        self,
        text: str,
        output_anchors: List[Anchor],
    ) -> List[Tuple[int, int, str]]:
        """Refine claim splitting when multi-record conflict detected.

        Only uses anchors with valid span info (start>=0) for split point
        calculation. Normalized anchors (start=-1) are skipped.
        """
        discriminator_anchors = [
            a for a in output_anchors
            if self._is_discriminator_field(a.field_name) and a.start >= 0
        ]
        if len(discriminator_anchors) < 2:
            return self._split_into_segments(text)

        discriminator_anchors.sort(key=lambda a: a.start)

        segments: List[Tuple[int, int, str]] = []
        prev_end = 0
        for i, anchor in enumerate(discriminator_anchors):
            near_text = text[max(0, anchor.start - 20):min(len(text), anchor.end + 100)]
            split_match = re.search(r"[；;]", near_text)
            if split_match:
                split_pos = max(0, anchor.start - 20) + split_match.start()
                if split_pos > prev_end:
                    seg_text = text[prev_end:split_pos].strip()
                    if seg_text:
                        segments.append((prev_end, split_pos, seg_text))
                    prev_end = split_pos + 1

        if prev_end < len(text):
            seg_text = text[prev_end:].strip()
            if seg_text:
                segments.append((prev_end, len(text), seg_text))

        if len(segments) < 2:
            return self._split_into_segments(text)

        return segments

    # ------------------------------------------------------------------
    # Context anchor identification
    # ------------------------------------------------------------------

    def _identify_context_anchors(
        self,
        input_anchors: List[Anchor],
    ) -> List[Anchor]:
        """Identify context anchors that apply to all claims.

        Context anchors are:
        1. From user input (shared across all claims)
        2. Entity identifiers (company_name, patient_name, etc.)
        3. Environmental/contextual fields (branch, department, industry, etc.)
        """
        context: List[Anchor] = []
        seen = set()
        context_field_names = self._context_field_names()

        for anchor in input_anchors:
            if anchor.inferred:
                continue
            field = anchor.field_name
            if field in context_field_names:
                key = (field, anchor.canonical_value)
                if key not in seen:
                    seen.add(key)
                    context.append(anchor)

        return context

    def _context_field_names(self) -> Set[str]:
        """Return field names that serve as context (not record discriminators)."""
        context: Set[str] = set(self._policy.identifier_fields)
        for field in self._policy.quasi_identifier_fields:
            if not self._is_discriminator_field(field):
                context.add(field)
        context.discard("loan_type")
        context.discard("loan_amount")
        context.discard("interest_rate")
        context.discard("collateral")
        return context

    # ------------------------------------------------------------------
    # Discriminator field detection
    # ------------------------------------------------------------------

    def _is_discriminator_field(self, field_name: str) -> bool:
        """Check if a field likely discriminates between multiple records."""
        if field_name in _DISCRIMINATOR_HEURISTICS:
            return True
        if any(kw in field_name.lower() for kw in ("amount", "rate", "rating")):
            return True
        return False

    def _has_multi_record_conflict(
        self,
        output_anchors: List[Anchor],
    ) -> bool:
        """Detect whether output contains multiple distinct records."""
        field_values: Dict[str, Set[str]] = {}
        for anchor in output_anchors:
            if not self._is_discriminator_field(anchor.field_name):
                continue
            val = anchor.canonical_value or anchor.text
            if not val:
                continue
            field_values.setdefault(anchor.field_name, set()).add(val)

        for field_name, values in field_values.items():
            if len(values) >= 2:
                return True
        return False

    # ------------------------------------------------------------------
    # Anchor filtering
    # ------------------------------------------------------------------

    def _filter_anchors_in_range(
        self,
        anchors: List[Anchor],
        start: int,
        end: int,
        seg_text: str = "",
    ) -> List[Anchor]:
        """Return anchors whose span falls within [start, end).

        For anchors without span info (start=-1, normalized anchors),
        include them if their text appears in the segment text.
        """
        result: List[Anchor] = []
        for anchor in anchors:
            if anchor.start < 0 or anchor.end < 0:
                # Anchors without span info: include if text appears in seg_text
                if seg_text and anchor.text and anchor.text in seg_text:
                    result.append(anchor)
                elif not seg_text:
                    result.append(anchor)
                continue
            # Anchor span overlaps with segment
            if anchor.end > start and anchor.start < end:
                result.append(anchor)
        return result

    def _contains_protected_text(self, text: str) -> bool:
        """Check if text contains any protected field values."""
        for field_name in self._policy.protected_fields:
            for asset in self._assets:
                val = asset.get(field_name)
                if val and val in text:
                    return True
        return False

    # ------------------------------------------------------------------
    # Claim deduplication
    # ------------------------------------------------------------------

    def _deduplicate_claims(
        self,
        claims: List[DisclosureClaim],
    ) -> List[DisclosureClaim]:
        """Deduplicate claims that represent the same record bindings."""
        if len(claims) <= 1:
            return claims

        seen_signatures: Set[str] = set()
        result: List[DisclosureClaim] = []

        for claim in claims:
            sig = self._claim_signature(claim)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            result.append(claim)

        return result

    def _claim_signature(self, claim: DisclosureClaim) -> str:
        """Build a stable signature for a claim."""
        parts: List[str] = []
        for anchor in sorted(
            claim.local_output_anchors,
            key=lambda a: (a.field_name, a.canonical_value),
        ):
            parts.append(f"{anchor.field_name}={anchor.canonical_value}")
        return "|".join(parts)

    # ------------------------------------------------------------------
    # Candidate binding per claim
    # ------------------------------------------------------------------

    def _bind_candidates(
        self,
        claim: DisclosureClaim,
    ) -> List[AssetFact]:
        """Find candidate assets that match a claim's anchors.

        Within a single claim: AND across different fields, OR within same field.
        Context anchors are applied first to narrow the pool.
        """
        candidates = list(self._assets)

        # Phase 1: Apply context anchors (AND)
        for ctx_anchor in claim.context_anchors:
            candidates = self._filter_by_anchor(candidates, ctx_anchor)
            if not candidates:
                break

        if not candidates:
            return []

        # Phase 2: Apply local output anchors (AND across fields, OR within field)
        field_values: Dict[str, Set[str]] = {}
        for anchor in claim.local_output_anchors:
            vals = self._anchor_values(anchor)
            if vals:
                field_values.setdefault(anchor.field_name, set()).update(vals)

        # Apply AND across fields with OR within field
        for field_name, accepted in field_values.items():
            candidates = [
                asset
                for asset in candidates
                if asset.get(field_name) in accepted
            ]
            if not candidates:
                break

        return candidates

    def _filter_by_anchor(
        self,
        candidates: List[AssetFact],
        anchor: Anchor,
    ) -> List[AssetFact]:
        """Filter candidates by a single anchor."""
        accepted = self._anchor_values(anchor)
        if not accepted:
            return candidates
        return [
            asset
            for asset in candidates
            if asset.get(anchor.field_name) in accepted
        ]

    @staticmethod
    def _anchor_values(anchor: Anchor) -> Set[str]:
        """Get all accepted values for an anchor."""
        values: Set[str] = set()
        if anchor.canonical_value:
            values.add(anchor.canonical_value)
        values.update(v for v in anchor.accepted_values if v)
        return values

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_claim_id(prefix: str, text: str, start: int, end: int) -> str:
        raw = f"{prefix}:{text[:40]}:{start}:{end}"
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8].upper()
        return f"CL{digest}"