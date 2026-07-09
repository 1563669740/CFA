from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .models import AssetFact, CandidateValue, FieldPolicy, SemanticFieldAlias
from .retriever import HybridSparseRetriever


class SemanticIndex:
    """Index that maps semantic aliases to candidate restricted-fact values.

    When LLM generates semantic aliases offline, this index is loaded and used
    for fast candidate retrieval *before* sending text to the LLM extractor.
    This way sensitive full fact pools are never sent to the LLM directly.

    As of v2, candidate retrieval uses a hybrid sparse retriever that fuses:
        - Alias / component / partial_clue matching (original logic)
        - BM25 scoring on pre-built document tokens
        - Chinese char 2-gram / 3-gram overlap scoring
        - Lightweight field-hint boosting
    """

    def __init__(
        self,
        policy: FieldPolicy,
        assets: Sequence[AssetFact],
        *,
        override_aliases: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    ):
        self._policy = policy
        self._assets = list(assets)
        self._semantic_aliases = override_aliases or policy.semantic_aliases

        # Pre-build a lookup: (field_name, canonical_value) -> SemanticFieldAlias
        self._alias_lookup: Dict[Tuple[str, str], SemanticFieldAlias] = {}
        for field_name, value_map in self._semantic_aliases.items():
            for canonical_value, info in value_map.items():
                key = (field_name, str(canonical_value))
                components = [str(c) for c in info.get("components", [])]
                aliases = [str(a) for a in info.get("aliases", [])]
                partial_clues = [str(p) for p in info.get("partial_clues", [])]
                self._alias_lookup[key] = SemanticFieldAlias(
                    components=components,
                    aliases=aliases,
                    partial_clues=partial_clues,
                    possible_inferences=[
                        dict(i) for i in info.get("possible_inferences", [])
                    ],
                    partial_match_policy=str(
                        info.get("partial_match_policy", "any_component_or_alias")
                    ),
                )

        # Pre-build a set of valid field values for verification
        self._valid_values: Dict[str, set] = {}
        for asset in self._assets:
            for field_name in self._policy.field_order:
                value = asset.get(field_name)
                if value:
                    self._valid_values.setdefault(field_name, set()).add(value)

        # Hybrid sparse retriever (v2)
        self._hybrid_retriever = HybridSparseRetriever(
            policy=self._policy,
            assets=self._assets,
            alias_lookup=self._alias_lookup,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve_candidates(self, text: str, top_k: int = 30) -> List[CandidateValue]:
        """Fast candidate retrieval using hybrid sparse scoring.

        Fuses alias/components/partial_clues, BM25, Chinese n-gram, and
        field-hint boosts.  Candidate objects now carry score_breakdown and
        matched_terms metadata (backwards-compatible).
        """
        return self._hybrid_retriever.retrieve(
            text=text,
            top_k=top_k,
            max_per_field=8,
            min_score=0.08,
        )

    def retrieve_candidates_for_field(
        self, text: str, field_name: str, top_k: int = 15
    ) -> List[CandidateValue]:
        """Retrieve candidates scoped to a single field."""
        all_candidates = self.retrieve_candidates(text, top_k=200)
        return [
            c for c in all_candidates if c.field_name == field_name
        ][:top_k]

    def build_candidate_text(
        self, candidates: List[CandidateValue], max_per_field: int = 10
    ) -> Dict[str, List[str]]:
        """Build a compact dict of candidate values grouped by field for LLM prompt.

        Only the top *max_per_field* candidates per field are included.
        """
        grouped: Dict[str, List[str]] = {}
        seen: Dict[str, set] = {}

        for candidate in candidates:
            if candidate.field_name not in seen:
                seen[candidate.field_name] = set()
            if candidate.canonical_value in seen[candidate.field_name]:
                continue
            if len(seen[candidate.field_name]) >= max_per_field:
                continue
            seen[candidate.field_name].add(candidate.canonical_value)
            grouped.setdefault(candidate.field_name, []).append(
                candidate.canonical_value
            )

        return grouped

    def get_valid_values(self, field_name: str) -> set:
        """Return all valid values for a field from the fact pool."""
        return self._valid_values.get(field_name, set())

    def get_aliases_for_value(
        self, field_name: str, canonical_value: str
    ) -> SemanticFieldAlias:
        """Return the semantic alias info for a specific canonical value."""
        return self._alias_lookup.get(
            (field_name, canonical_value),
            SemanticFieldAlias(),
        )

    def contains_value(self, field_name: str, value: str) -> bool:
        """Check whether a value exists in the fact pool."""
        return value in self._valid_values.get(field_name, set())