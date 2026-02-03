"""
Real-Time Document Intelligence & Compliance Engine
Entity Resolution System
Version: 1.0
Date: 2026-02-03

This module provides sophisticated entity resolution capabilities including
fuzzy matching, graph-based entity linking, and cross-document deduplication.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid
from collections import defaultdict, Counter
import json
import re
import math
from difflib import SequenceMatcher
from itertools import combinations

try:
    import numpy as np
    from scipy.spatial.distance import cosine

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

    # Mock implementations for basic functionality
    def cosine(a, b):
        return 0.5  # Default similarity


logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be resolved"""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    FINANCIAL_INSTRUMENT = "financial_instrument"
    CONTRACT = "contract"
    LEGAL_ENTITY = "legal_entity"
    PRODUCT = "product"
    DATE = "date"
    AMOUNT = "amount"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"


class MatchType(Enum):
    """Types of entity matches"""

    EXACT = "exact"
    FUZZY = "fuzzy"
    PHONETIC = "phonetic"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    CONTEXTUAL = "contextual"


class ResolutionStatus(Enum):
    """Status of entity resolution"""

    RESOLVED = "resolved"
    CANDIDATE = "candidate"
    UNRESOLVED = "unresolved"
    CONFLICTED = "conflicted"
    MERGED = "merged"


@dataclass
class EntityMention:
    """Individual mention of an entity in a document"""

    mention_id: str
    entity_text: str
    entity_type: EntityType
    document_id: str
    document_section: str
    start_position: int
    end_position: int
    confidence: float
    context: str
    extracted_attributes: Dict[str, Any]
    normalization_hints: List[str]
    created_at: datetime


@dataclass
class EntityCandidate:
    """Candidate entity for resolution"""

    candidate_id: str
    canonical_id: Optional[str]
    entity_type: EntityType
    canonical_name: str
    aliases: Set[str]
    attributes: Dict[str, Any]
    mentions: List[EntityMention]
    confidence_score: float
    resolution_status: ResolutionStatus
    last_updated: datetime

    def __post_init__(self):
        """Ensure aliases is a set"""
        if not isinstance(self.aliases, set):
            self.aliases = set(self.aliases) if self.aliases else set()


@dataclass
class EntityMatch:
    """Match between two entities"""

    match_id: str
    source_entity: EntityCandidate
    target_entity: EntityCandidate
    match_type: MatchType
    similarity_score: float
    confidence: float
    matching_features: Dict[str, float]
    explanation: str
    matched_at: datetime


@dataclass
class EntityCluster:
    """Cluster of resolved entities"""

    cluster_id: str
    canonical_entity: EntityCandidate
    merged_entities: List[EntityCandidate]
    cluster_confidence: float
    resolution_method: str
    created_at: datetime
    updated_at: datetime


class FuzzyMatcher:
    """Advanced fuzzy string matching with multiple algorithms"""

    def __init__(self):
        self.phonetic_cache = {}

        # Common business entity suffixes for normalization
        self.business_suffixes = {
            "inc",
            "corp",
            "ltd",
            "llc",
            "lp",
            "company",
            "corporation",
            "limited",
            "incorporated",
            "co",
            "llp",
            "pllc",
            "pc",
        }

        # Common title prefixes for person names
        self.name_prefixes = {"mr", "mrs", "ms", "dr", "prof", "sir", "lady", "lord"}

        # Common name suffixes
        self.name_suffixes = {"jr", "sr", "ii", "iii", "iv", "phd", "md", "esq"}

    def calculate_similarity(
        self, text1: str, text2: str, entity_type: EntityType = None
    ) -> float:
        """
        Calculate comprehensive similarity score between two text strings

        Args:
            text1: First text string
            text2: Second text string
            entity_type: Type of entity for specialized matching

        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not text1 or not text2:
                return 0.0

            # Normalize texts
            norm1 = self._normalize_text(text1, entity_type)
            norm2 = self._normalize_text(text2, entity_type)

            if norm1 == norm2:
                return 1.0

            # Calculate multiple similarity metrics
            similarities = {}

            # Exact match after normalization
            similarities["exact"] = 1.0 if norm1 == norm2 else 0.0

            # Levenshtein-based similarity
            similarities["levenshtein"] = SequenceMatcher(None, norm1, norm2).ratio()

            # Jaccard similarity (word-level)
            similarities["jaccard"] = self._jaccard_similarity(norm1, norm2)

            # Jaro-Winkler similarity (good for names)
            similarities["jaro_winkler"] = self._jaro_winkler_similarity(norm1, norm2)

            # Phonetic similarity
            similarities["phonetic"] = self._phonetic_similarity(norm1, norm2)

            # Token similarity (for multi-word entities)
            similarities["token"] = self._token_similarity(norm1, norm2)

            # Entity-type specific similarity
            if entity_type:
                similarities["entity_specific"] = self._entity_specific_similarity(
                    norm1, norm2, entity_type
                )

            # Weighted combination based on entity type
            weights = self._get_similarity_weights(entity_type)

            final_similarity = sum(
                similarities[metric] * weight
                for metric, weight in weights.items()
                if metric in similarities
            )

            return min(1.0, max(0.0, final_similarity))

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0

    def _normalize_text(self, text: str, entity_type: EntityType = None) -> str:
        """Normalize text for comparison"""
        try:
            # Convert to lowercase and strip
            normalized = text.lower().strip()

            # Remove extra whitespace
            normalized = re.sub(r"\s+", " ", normalized)

            # Remove punctuation except useful ones
            normalized = re.sub(r"[^\w\s\-\.]", " ", normalized)

            # Entity-specific normalization
            if entity_type == EntityType.ORGANIZATION:
                normalized = self._normalize_organization_name(normalized)
            elif entity_type == EntityType.PERSON:
                normalized = self._normalize_person_name(normalized)
            elif entity_type == EntityType.ADDRESS:
                normalized = self._normalize_address(normalized)

            return normalized.strip()

        except Exception:
            return text.lower().strip()

    def _normalize_organization_name(self, text: str) -> str:
        """Normalize organization names"""
        # Remove common suffixes
        words = text.split()
        filtered_words = []

        for word in words:
            # Remove punctuation from word
            clean_word = re.sub(r"[^\w]", "", word)
            if clean_word not in self.business_suffixes:
                filtered_words.append(word)

        # Handle "the" prefix
        if filtered_words and filtered_words[0] == "the":
            filtered_words = filtered_words[1:]

        return " ".join(filtered_words)

    def _normalize_person_name(self, text: str) -> str:
        """Normalize person names"""
        words = text.split()
        filtered_words = []

        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word)

            # Skip prefixes at beginning
            if i == 0 and clean_word in self.name_prefixes:
                continue

            # Skip suffixes at end
            if i == len(words) - 1 and clean_word in self.name_suffixes:
                continue

            filtered_words.append(word)

        return " ".join(filtered_words)

    def _normalize_address(self, text: str) -> str:
        """Normalize addresses"""
        # Common address abbreviations
        abbreviations = {
            "street": "st",
            "avenue": "ave",
            "boulevard": "blvd",
            "drive": "dr",
            "road": "rd",
            "lane": "ln",
            "court": "ct",
            "apartment": "apt",
            "suite": "ste",
            "floor": "fl",
        }

        normalized = text
        for full, abbrev in abbreviations.items():
            normalized = re.sub(rf"\b{full}\b", abbrev, normalized)
            normalized = re.sub(rf"\b{abbrev}\b", abbrev, normalized)

        return normalized

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        try:
            words1 = set(text1.split())
            words2 = set(text2.split())

            if not words1 and not words2:
                return 1.0

            intersection = words1 & words2
            union = words1 | words2

            return len(intersection) / len(union) if union else 0.0

        except Exception:
            return 0.0

    def _jaro_winkler_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaro-Winkler similarity"""
        try:
            # Simplified Jaro-Winkler implementation
            if text1 == text2:
                return 1.0

            len1, len2 = len(text1), len(text2)

            if len1 == 0 or len2 == 0:
                return 0.0

            # Calculate Jaro similarity
            match_window = max(len1, len2) // 2 - 1
            if match_window < 1:
                match_window = 1

            text1_matches = [False] * len1
            text2_matches = [False] * len2

            matches = 0
            transpositions = 0

            # Find matches
            for i in range(len1):
                start = max(0, i - match_window)
                end = min(i + match_window + 1, len2)

                for j in range(start, end):
                    if text2_matches[j] or text1[i] != text2[j]:
                        continue

                    text1_matches[i] = True
                    text2_matches[j] = True
                    matches += 1
                    break

            if matches == 0:
                return 0.0

            # Count transpositions
            k = 0
            for i in range(len1):
                if not text1_matches[i]:
                    continue

                while not text2_matches[k]:
                    k += 1

                if text1[i] != text2[k]:
                    transpositions += 1

                k += 1

            jaro = (
                matches / len1
                + matches / len2
                + (matches - transpositions / 2) / matches
            ) / 3

            # Calculate Jaro-Winkler with prefix scaling
            prefix = 0
            for i in range(min(len1, len2, 4)):
                if text1[i] == text2[i]:
                    prefix += 1
                else:
                    break

            return jaro + (0.1 * prefix * (1 - jaro))

        except Exception:
            return 0.0

    def _phonetic_similarity(self, text1: str, text2: str) -> float:
        """Calculate phonetic similarity using simplified Soundex"""
        try:
            soundex1 = self._simple_soundex(text1)
            soundex2 = self._simple_soundex(text2)

            if soundex1 == soundex2:
                return 1.0

            # Partial match for multi-word entities
            words1 = text1.split()
            words2 = text2.split()

            if len(words1) > 1 or len(words2) > 1:
                soundex_matches = 0
                total_words = max(len(words1), len(words2))

                for word1 in words1:
                    for word2 in words2:
                        if self._simple_soundex(word1) == self._simple_soundex(word2):
                            soundex_matches += 1
                            break

                return soundex_matches / total_words if total_words > 0 else 0.0

            return 0.0

        except Exception:
            return 0.0

    def _simple_soundex(self, text: str) -> str:
        """Simple Soundex implementation"""
        if not text:
            return "0000"

        # Cache for performance
        if text in self.phonetic_cache:
            return self.phonetic_cache[text]

        text = text.upper()

        # Keep first letter
        soundex = text[0]

        # Mapping for consonants
        mapping = {
            "B": "1",
            "F": "1",
            "P": "1",
            "V": "1",
            "C": "2",
            "G": "2",
            "J": "2",
            "K": "2",
            "Q": "2",
            "S": "2",
            "X": "2",
            "Z": "2",
            "D": "3",
            "T": "3",
            "L": "4",
            "M": "5",
            "N": "5",
            "R": "6",
        }

        # Convert to soundex
        for char in text[1:]:
            if char in mapping:
                code = mapping[char]
                if soundex[-1] != code:  # Avoid consecutive duplicates
                    soundex += code

        # Remove vowels and pad/truncate to 4 characters
        soundex = (
            soundex.replace("A", "")
            .replace("E", "")
            .replace("I", "")
            .replace("O", "")
            .replace("U", "")
            .replace("Y", "")
            .replace("H", "")
            .replace("W", "")
        )
        soundex = (soundex + "0000")[:4]

        self.phonetic_cache[text] = soundex
        return soundex

    def _token_similarity(self, text1: str, text2: str) -> float:
        """Calculate token-based similarity"""
        try:
            tokens1 = set(text1.split())
            tokens2 = set(text2.split())

            if not tokens1 and not tokens2:
                return 1.0

            # Calculate token overlap
            common_tokens = tokens1 & tokens2
            all_tokens = tokens1 | tokens2

            if not all_tokens:
                return 0.0

            return len(common_tokens) / len(all_tokens)

        except Exception:
            return 0.0

    def _entity_specific_similarity(
        self, text1: str, text2: str, entity_type: EntityType
    ) -> float:
        """Calculate entity-type specific similarity"""
        try:
            if entity_type == EntityType.PERSON:
                return self._person_similarity(text1, text2)
            elif entity_type == EntityType.ORGANIZATION:
                return self._organization_similarity(text1, text2)
            elif entity_type == EntityType.ADDRESS:
                return self._address_similarity(text1, text2)
            elif entity_type in [EntityType.EMAIL, EntityType.PHONE]:
                return self._contact_similarity(text1, text2)
            else:
                return self._token_similarity(text1, text2)

        except Exception:
            return 0.0

    def _person_similarity(self, text1: str, text2: str) -> float:
        """Specialized similarity for person names"""
        try:
            words1 = text1.split()
            words2 = text2.split()

            # Handle initials
            if len(words1) != len(words2):
                # Check if one is an initial version of the other
                return self._initial_similarity(words1, words2)

            # Calculate name component similarity
            similarities = []
            for w1, w2 in zip(words1, words2):
                if len(w1) == 1 or len(w2) == 1:  # Initial
                    similarities.append(1.0 if w1[0] == w2[0] else 0.0)
                else:
                    similarities.append(SequenceMatcher(None, w1, w2).ratio())

            return sum(similarities) / len(similarities) if similarities else 0.0

        except Exception:
            return 0.0

    def _initial_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Handle similarity with initials"""
        try:
            shorter, longer = (
                (words1, words2) if len(words1) < len(words2) else (words2, words1)
            )

            if len(shorter) == 0:
                return 0.0

            matches = 0
            for i, short_word in enumerate(shorter):
                if i < len(longer):
                    long_word = longer[i]
                    if len(short_word) == 1:  # Initial
                        if short_word[0] == long_word[0]:
                            matches += 1
                    elif len(long_word) == 1:  # Other way around
                        if long_word[0] == short_word[0]:
                            matches += 1
                    else:
                        if SequenceMatcher(None, short_word, long_word).ratio() > 0.8:
                            matches += 1

            return matches / max(len(shorter), len(longer))

        except Exception:
            return 0.0

    def _organization_similarity(self, text1: str, text2: str) -> float:
        """Specialized similarity for organization names"""
        try:
            # Check for acronyms
            words1 = text1.split()
            words2 = text2.split()

            # Handle acronym cases
            if len(words1) == 1 and len(words1[0]) <= 6 and len(words2) > 1:
                acronym = "".join(word[0] for word in words2)
                if words1[0] == acronym:
                    return 0.9  # High similarity for acronym match

            elif len(words2) == 1 and len(words2[0]) <= 6 and len(words1) > 1:
                acronym = "".join(word[0] for word in words1)
                if words2[0] == acronym:
                    return 0.9

            # Regular token similarity for organizations
            return self._token_similarity(text1, text2)

        except Exception:
            return 0.0

    def _address_similarity(self, text1: str, text2: str) -> float:
        """Specialized similarity for addresses"""
        try:
            # Extract address components
            components1 = self._extract_address_components(text1)
            components2 = self._extract_address_components(text2)

            similarities = {}

            # Compare each component
            for component in ["number", "street", "city", "state", "zip"]:
                val1 = components1.get(component, "")
                val2 = components2.get(component, "")

                if val1 and val2:
                    similarities[component] = SequenceMatcher(None, val1, val2).ratio()
                elif not val1 and not val2:
                    similarities[component] = 1.0
                else:
                    similarities[component] = 0.0

            # Weighted average (street and number are most important)
            weights = {
                "number": 0.3,
                "street": 0.4,
                "city": 0.15,
                "state": 0.1,
                "zip": 0.05,
            }

            weighted_sim = sum(
                similarities[comp] * weight
                for comp, weight in weights.items()
                if comp in similarities
            )

            return weighted_sim

        except Exception:
            return 0.0

    def _extract_address_components(self, address: str) -> Dict[str, str]:
        """Extract components from address string"""
        components = {}

        # Simple regex patterns for address components
        zip_pattern = r"\b\d{5}(?:-\d{4})?\b"
        state_pattern = r"\b[A-Z]{2}\b"
        number_pattern = r"^\d+"

        # Extract ZIP
        zip_match = re.search(zip_pattern, address)
        if zip_match:
            components["zip"] = zip_match.group()
            address = address.replace(components["zip"], "").strip()

        # Extract state (2-letter code)
        state_match = re.search(state_pattern, address)
        if state_match:
            components["state"] = state_match.group()
            address = address.replace(components["state"], "").strip()

        # Extract number (at beginning)
        number_match = re.match(number_pattern, address)
        if number_match:
            components["number"] = number_match.group()
            address = address[len(components["number"]) :].strip()

        # Remaining parts
        parts = address.split(",")
        if len(parts) >= 2:
            components["street"] = parts[0].strip()
            components["city"] = parts[1].strip()
        elif len(parts) == 1:
            components["street"] = parts[0].strip()

        return components

    def _contact_similarity(self, text1: str, text2: str) -> float:
        """Specialized similarity for email/phone"""
        try:
            # For email and phone, normalize and use exact/fuzzy matching
            norm1 = re.sub(r"[^\w@\.]", "", text1.lower())
            norm2 = re.sub(r"[^\w@\.]", "", text2.lower())

            if norm1 == norm2:
                return 1.0

            # For emails, check domain similarity
            if "@" in norm1 and "@" in norm2:
                domain1 = norm1.split("@")[1]
                domain2 = norm2.split("@")[1]
                if domain1 == domain2:
                    user1 = norm1.split("@")[0]
                    user2 = norm2.split("@")[0]
                    return 0.5 + 0.5 * SequenceMatcher(None, user1, user2).ratio()

            return SequenceMatcher(None, norm1, norm2).ratio()

        except Exception:
            return 0.0

    def _get_similarity_weights(
        self, entity_type: EntityType = None
    ) -> Dict[str, float]:
        """Get weights for different similarity metrics based on entity type"""
        if entity_type == EntityType.PERSON:
            return {
                "exact": 0.3,
                "levenshtein": 0.2,
                "jaro_winkler": 0.25,
                "phonetic": 0.15,
                "entity_specific": 0.1,
            }
        elif entity_type == EntityType.ORGANIZATION:
            return {
                "exact": 0.25,
                "levenshtein": 0.2,
                "jaccard": 0.2,
                "token": 0.2,
                "entity_specific": 0.15,
            }
        elif entity_type == EntityType.ADDRESS:
            return {
                "exact": 0.2,
                "entity_specific": 0.5,
                "token": 0.2,
                "levenshtein": 0.1,
            }
        elif entity_type in [EntityType.EMAIL, EntityType.PHONE]:
            return {"exact": 0.4, "entity_specific": 0.4, "levenshtein": 0.2}
        else:
            # Generic weights
            return {
                "exact": 0.3,
                "levenshtein": 0.25,
                "jaccard": 0.2,
                "jaro_winkler": 0.15,
                "token": 0.1,
            }


class EntityResolver:
    """Main entity resolution system with fuzzy matching and graph-based linking"""

    def __init__(
        self, similarity_threshold: float = 0.8, cluster_threshold: float = 0.75
    ):
        """
        Initialize entity resolver

        Args:
            similarity_threshold: Threshold for considering entities as matches
            cluster_threshold: Threshold for clustering entities
        """
        self.similarity_threshold = similarity_threshold
        self.cluster_threshold = cluster_threshold

        # Core components
        self.fuzzy_matcher = FuzzyMatcher()

        # Storage for entities and matches
        self.entity_candidates: Dict[str, EntityCandidate] = {}
        self.entity_matches: Dict[str, EntityMatch] = {}
        self.entity_clusters: Dict[str, EntityCluster] = {}

        # Indexes for efficient lookup
        self.entity_index: Dict[EntityType, Dict[str, List[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.mention_index: Dict[str, List[str]] = defaultdict(
            list
        )  # document_id -> entity_ids

        # Statistics and caching
        self.resolution_stats = {
            "total_mentions": 0,
            "resolved_entities": 0,
            "created_clusters": 0,
            "disambiguation_cases": 0,
        }

        logger.info(
            f"Entity resolver initialized with thresholds: similarity={similarity_threshold}, cluster={cluster_threshold}"
        )

    def add_entity_mention(self, mention: EntityMention) -> str:
        """
        Add a new entity mention for resolution

        Args:
            mention: Entity mention to add

        Returns:
            Entity candidate ID
        """
        try:
            self.resolution_stats["total_mentions"] += 1

            # Try to find existing matches
            matching_candidates = self._find_matching_candidates(mention)

            if matching_candidates:
                # Add mention to best matching candidate
                best_match = max(
                    matching_candidates, key=lambda x: x[1]
                )  # highest similarity
                candidate_id = best_match[0]

                if candidate_id in self.entity_candidates:
                    candidate = self.entity_candidates[candidate_id]
                    candidate.mentions.append(mention)
                    candidate.last_updated = datetime.now()

                    # Update attributes from new mention
                    self._merge_attributes(candidate, mention)

                    # Update indexes
                    self._update_indexes(candidate_id, mention)

                    logger.debug(f"Added mention to existing candidate: {candidate_id}")
                    return candidate_id

            # Create new candidate entity
            candidate_id = str(uuid.uuid4())

            candidate = EntityCandidate(
                candidate_id=candidate_id,
                canonical_id=None,
                entity_type=mention.entity_type,
                canonical_name=mention.entity_text,
                aliases=set(),
                attributes=mention.extracted_attributes.copy(),
                mentions=[mention],
                confidence_score=mention.confidence,
                resolution_status=ResolutionStatus.CANDIDATE,
                last_updated=datetime.now(),
            )

            self.entity_candidates[candidate_id] = candidate

            # Update indexes
            self._update_indexes(candidate_id, mention)

            # Try to link with existing entities
            self._attempt_entity_linking(candidate_id)

            logger.debug(f"Created new entity candidate: {candidate_id}")
            return candidate_id

        except Exception as e:
            logger.error(f"Failed to add entity mention: {e}")
            # Create minimal candidate as fallback
            candidate_id = str(uuid.uuid4())
            self.entity_candidates[candidate_id] = EntityCandidate(
                candidate_id=candidate_id,
                canonical_id=None,
                entity_type=mention.entity_type,
                canonical_name=mention.entity_text,
                aliases=set(),
                attributes={},
                mentions=[mention],
                confidence_score=0.5,
                resolution_status=ResolutionStatus.UNRESOLVED,
                last_updated=datetime.now(),
            )
            return candidate_id

    def _find_matching_candidates(
        self, mention: EntityMention
    ) -> List[Tuple[str, float]]:
        """Find existing candidates that match the new mention"""
        try:
            matches = []

            # Look for candidates of the same entity type
            entity_type_candidates = self.entity_index.get(mention.entity_type, {})

            # Try different lookup strategies
            lookup_keys = [
                mention.entity_text.lower(),
                mention.entity_text.lower().strip(),
                " ".join(mention.entity_text.split()),  # normalized whitespace
            ]

            for key in lookup_keys:
                if key in entity_type_candidates:
                    for candidate_id in entity_type_candidates[key]:
                        if candidate_id in self.entity_candidates:
                            # Calculate similarity with existing candidate
                            candidate = self.entity_candidates[candidate_id]
                            similarity = self.fuzzy_matcher.calculate_similarity(
                                mention.entity_text,
                                candidate.canonical_name,
                                mention.entity_type,
                            )

                            if similarity >= self.similarity_threshold:
                                matches.append((candidate_id, similarity))

            # If no exact matches, try fuzzy matching against all candidates of same type
            if (
                not matches and len(entity_type_candidates) < 1000
            ):  # Avoid expensive computation for large sets
                for candidate_id in self.entity_candidates:
                    candidate = self.entity_candidates[candidate_id]

                    if candidate.entity_type != mention.entity_type:
                        continue

                    similarity = self.fuzzy_matcher.calculate_similarity(
                        mention.entity_text,
                        candidate.canonical_name,
                        mention.entity_type,
                    )

                    if similarity >= self.similarity_threshold:
                        matches.append((candidate_id, similarity))

            return sorted(matches, key=lambda x: x[1], reverse=True)

        except Exception as e:
            logger.error(f"Error finding matching candidates: {e}")
            return []

    def _merge_attributes(self, candidate: EntityCandidate, mention: EntityMention):
        """Merge attributes from new mention into candidate"""
        try:
            # Add entity text as alias if different
            if mention.entity_text.lower() != candidate.canonical_name.lower():
                candidate.aliases.add(mention.entity_text)

            # Merge extracted attributes
            for key, value in mention.extracted_attributes.items():
                if key not in candidate.attributes:
                    candidate.attributes[key] = value
                elif isinstance(value, list) and isinstance(
                    candidate.attributes[key], list
                ):
                    # Merge lists and remove duplicates
                    combined = candidate.attributes[key] + value
                    candidate.attributes[key] = list(set(combined))

            # Update confidence (weighted average)
            total_mentions = len(candidate.mentions)
            if total_mentions > 1:
                candidate.confidence_score = (
                    candidate.confidence_score * (total_mentions - 1)
                    + mention.confidence
                ) / total_mentions

        except Exception as e:
            logger.error(f"Error merging attributes: {e}")

    def _update_indexes(self, candidate_id: str, mention: EntityMention):
        """Update lookup indexes"""
        try:
            # Entity type index
            entity_type = mention.entity_type
            normalized_text = mention.entity_text.lower()

            if candidate_id not in self.entity_index[entity_type][normalized_text]:
                self.entity_index[entity_type][normalized_text].append(candidate_id)

            # Document mention index
            if candidate_id not in self.mention_index[mention.document_id]:
                self.mention_index[mention.document_id].append(candidate_id)

        except Exception as e:
            logger.error(f"Error updating indexes: {e}")

    def _attempt_entity_linking(self, candidate_id: str):
        """Attempt to link entity with existing clusters"""
        try:
            if candidate_id not in self.entity_candidates:
                return

            candidate = self.entity_candidates[candidate_id]

            # Look for potential matches in existing clusters
            potential_matches = []

            for cluster_id, cluster in self.entity_clusters.items():
                # Check similarity with canonical entity
                canonical_similarity = self.fuzzy_matcher.calculate_similarity(
                    candidate.canonical_name,
                    cluster.canonical_entity.canonical_name,
                    candidate.entity_type,
                )

                if canonical_similarity >= self.cluster_threshold:
                    potential_matches.append((cluster_id, canonical_similarity))

            if potential_matches:
                # Add to best matching cluster
                best_cluster_id = max(potential_matches, key=lambda x: x[1])[0]
                self._add_to_cluster(candidate_id, best_cluster_id)
            else:
                # Create new cluster if confidence is high enough
                if candidate.confidence_score >= 0.7:
                    self._create_new_cluster(candidate_id)

        except Exception as e:
            logger.error(f"Error in entity linking: {e}")

    def _add_to_cluster(self, candidate_id: str, cluster_id: str):
        """Add entity candidate to existing cluster"""
        try:
            if (
                cluster_id not in self.entity_clusters
                or candidate_id not in self.entity_candidates
            ):
                return

            cluster = self.entity_clusters[cluster_id]
            candidate = self.entity_candidates[candidate_id]

            # Add candidate to cluster
            cluster.merged_entities.append(candidate)
            cluster.updated_at = datetime.now()

            # Update candidate status
            candidate.canonical_id = cluster.canonical_entity.candidate_id
            candidate.resolution_status = ResolutionStatus.RESOLVED

            # Update cluster confidence
            all_entities = [cluster.canonical_entity] + cluster.merged_entities
            cluster.cluster_confidence = sum(
                e.confidence_score for e in all_entities
            ) / len(all_entities)

            # Merge aliases into canonical entity
            for alias in candidate.aliases:
                cluster.canonical_entity.aliases.add(alias)

            self.resolution_stats["resolved_entities"] += 1

            logger.debug(f"Added entity {candidate_id} to cluster {cluster_id}")

        except Exception as e:
            logger.error(f"Error adding to cluster: {e}")

    def _create_new_cluster(self, candidate_id: str):
        """Create new entity cluster"""
        try:
            if candidate_id not in self.entity_candidates:
                return

            candidate = self.entity_candidates[candidate_id]

            cluster_id = str(uuid.uuid4())

            cluster = EntityCluster(
                cluster_id=cluster_id,
                canonical_entity=candidate,
                merged_entities=[],
                cluster_confidence=candidate.confidence_score,
                resolution_method="new_cluster",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

            self.entity_clusters[cluster_id] = cluster

            # Update candidate
            candidate.canonical_id = candidate_id
            candidate.resolution_status = ResolutionStatus.RESOLVED

            self.resolution_stats["created_clusters"] += 1

            logger.debug(f"Created new cluster {cluster_id} for entity {candidate_id}")

        except Exception as e:
            logger.error(f"Error creating cluster: {e}")

    def resolve_entity_batch(self, mentions: List[EntityMention]) -> Dict[str, str]:
        """
        Resolve a batch of entity mentions

        Args:
            mentions: List of entity mentions to resolve

        Returns:
            Dictionary mapping mention_id to candidate_id
        """
        try:
            resolution_map = {}

            # Sort mentions by confidence (process high-confidence first)
            sorted_mentions = sorted(mentions, key=lambda x: x.confidence, reverse=True)

            for mention in sorted_mentions:
                candidate_id = self.add_entity_mention(mention)
                resolution_map[mention.mention_id] = candidate_id

            # Post-processing: identify potential disambiguation cases
            self._identify_disambiguation_cases()

            return resolution_map

        except Exception as e:
            logger.error(f"Batch resolution failed: {e}")
            return {}

    def _identify_disambiguation_cases(self):
        """Identify cases where entity disambiguation is needed"""
        try:
            # Group candidates by normalized name
            name_groups = defaultdict(list)

            for candidate_id, candidate in self.entity_candidates.items():
                normalized_name = candidate.canonical_name.lower().strip()
                name_groups[normalized_name].append(candidate_id)

            # Check for disambiguation needs
            for name, candidate_ids in name_groups.items():
                if len(candidate_ids) > 1:
                    # Multiple candidates with same name - needs disambiguation
                    self._handle_disambiguation(candidate_ids)

        except Exception as e:
            logger.error(f"Error in disambiguation identification: {e}")

    def _handle_disambiguation(self, candidate_ids: List[str]):
        """Handle entity disambiguation for candidates with similar names"""
        try:
            candidates = [
                self.entity_candidates[cid]
                for cid in candidate_ids
                if cid in self.entity_candidates
            ]

            if len(candidates) < 2:
                return

            # Analyze contexts and attributes for disambiguation
            for i, candidate1 in enumerate(candidates):
                for j, candidate2 in enumerate(candidates[i + 1 :], i + 1):
                    # Check if they should be merged based on additional evidence
                    should_merge = self._should_merge_candidates(candidate1, candidate2)

                    if should_merge:
                        # Merge the candidates
                        self._merge_candidates(
                            candidate1.candidate_id, candidate2.candidate_id
                        )
                    else:
                        # Mark as conflicted for manual review
                        candidate1.resolution_status = ResolutionStatus.CONFLICTED
                        candidate2.resolution_status = ResolutionStatus.CONFLICTED

            self.resolution_stats["disambiguation_cases"] += 1

        except Exception as e:
            logger.error(f"Error handling disambiguation: {e}")

    def _should_merge_candidates(
        self, candidate1: EntityCandidate, candidate2: EntityCandidate
    ) -> bool:
        """Determine if two candidates should be merged"""
        try:
            # Different entity types - don't merge
            if candidate1.entity_type != candidate2.entity_type:
                return False

            # Check attribute overlap
            attr_similarity = self._calculate_attribute_similarity(
                candidate1.attributes, candidate2.attributes
            )

            # Check context similarity (from mentions)
            context_similarity = self._calculate_context_similarity(
                candidate1, candidate2
            )

            # Check document co-occurrence
            doc_cooccurrence = self._check_document_cooccurrence(candidate1, candidate2)

            # Weighted decision
            merge_score = (
                attr_similarity * 0.4
                + context_similarity * 0.4
                + (1.0 if doc_cooccurrence else 0.0) * 0.2
            )

            return merge_score >= 0.7

        except Exception:
            return False

    def _calculate_attribute_similarity(
        self, attrs1: Dict[str, Any], attrs2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between attribute dictionaries"""
        try:
            if not attrs1 and not attrs2:
                return 1.0

            if not attrs1 or not attrs2:
                return 0.0

            common_keys = set(attrs1.keys()) & set(attrs2.keys())

            if not common_keys:
                return 0.0

            matches = 0
            for key in common_keys:
                val1, val2 = attrs1[key], attrs2[key]

                if val1 == val2:
                    matches += 1
                elif isinstance(val1, str) and isinstance(val2, str):
                    # Fuzzy match for string values
                    if SequenceMatcher(None, val1.lower(), val2.lower()).ratio() > 0.8:
                        matches += 1

            return matches / len(common_keys)

        except Exception:
            return 0.0

    def _calculate_context_similarity(
        self, candidate1: EntityCandidate, candidate2: EntityCandidate
    ) -> float:
        """Calculate similarity between mention contexts"""
        try:
            contexts1 = [
                mention.context for mention in candidate1.mentions if mention.context
            ]
            contexts2 = [
                mention.context for mention in candidate2.mentions if mention.context
            ]

            if not contexts1 or not contexts2:
                return 0.5  # Neutral when no context available

            # Calculate average similarity between contexts
            similarities = []

            for ctx1 in contexts1:
                for ctx2 in contexts2:
                    # Simple token overlap similarity
                    words1 = set(ctx1.lower().split())
                    words2 = set(ctx2.lower().split())

                    if words1 | words2:
                        similarity = len(words1 & words2) / len(words1 | words2)
                        similarities.append(similarity)

            return sum(similarities) / len(similarities) if similarities else 0.0

        except Exception:
            return 0.0

    def _check_document_cooccurrence(
        self, candidate1: EntityCandidate, candidate2: EntityCandidate
    ) -> bool:
        """Check if candidates co-occur in the same documents"""
        try:
            docs1 = set(mention.document_id for mention in candidate1.mentions)
            docs2 = set(mention.document_id for mention in candidate2.mentions)

            # If they appear in same documents, less likely to be same entity
            # (unless it's a complex document with multiple sections)
            return len(docs1 & docs2) == 0

        except Exception:
            return False

    def _merge_candidates(self, candidate1_id: str, candidate2_id: str):
        """Merge two entity candidates"""
        try:
            if (
                candidate1_id not in self.entity_candidates
                or candidate2_id not in self.entity_candidates
            ):
                return

            candidate1 = self.entity_candidates[candidate1_id]
            candidate2 = self.entity_candidates[candidate2_id]

            # Merge mentions
            candidate1.mentions.extend(candidate2.mentions)

            # Merge aliases
            candidate1.aliases.update(candidate2.aliases)
            candidate1.aliases.add(candidate2.canonical_name)

            # Merge attributes
            for key, value in candidate2.attributes.items():
                if key not in candidate1.attributes:
                    candidate1.attributes[key] = value
                elif isinstance(value, list) and isinstance(
                    candidate1.attributes[key], list
                ):
                    candidate1.attributes[key] = list(
                        set(candidate1.attributes[key] + value)
                    )

            # Update confidence and status
            total_mentions = len(candidate1.mentions)
            candidate1.confidence_score = (
                sum(m.confidence for m in candidate1.mentions) / total_mentions
            )
            candidate1.resolution_status = ResolutionStatus.MERGED
            candidate1.last_updated = datetime.now()

            # Remove second candidate
            del self.entity_candidates[candidate2_id]

            # Update indexes
            self._rebuild_indexes()

            logger.debug(f"Merged candidates {candidate1_id} and {candidate2_id}")

        except Exception as e:
            logger.error(f"Error merging candidates: {e}")

    def _rebuild_indexes(self):
        """Rebuild lookup indexes after merging"""
        try:
            # Clear existing indexes
            self.entity_index.clear()
            self.mention_index.clear()

            # Rebuild from current candidates
            for candidate_id, candidate in self.entity_candidates.items():
                for mention in candidate.mentions:
                    self._update_indexes(candidate_id, mention)

        except Exception as e:
            logger.error(f"Error rebuilding indexes: {e}")

    def get_entity_by_id(self, entity_id: str) -> Optional[EntityCandidate]:
        """Get entity candidate by ID"""
        return self.entity_candidates.get(entity_id)

    def get_entities_by_document(self, document_id: str) -> List[EntityCandidate]:
        """Get all entities mentioned in a document"""
        try:
            entity_ids = self.mention_index.get(document_id, [])
            return [
                self.entity_candidates[eid]
                for eid in entity_ids
                if eid in self.entity_candidates
            ]
        except Exception as e:
            logger.error(f"Error getting entities by document: {e}")
            return []

    def get_entities_by_type(self, entity_type: EntityType) -> List[EntityCandidate]:
        """Get all entities of a specific type"""
        try:
            return [
                candidate
                for candidate in self.entity_candidates.values()
                if candidate.entity_type == entity_type
            ]
        except Exception as e:
            logger.error(f"Error getting entities by type: {e}")
            return []

    def get_resolution_statistics(self) -> Dict[str, Any]:
        """Get entity resolution statistics"""
        try:
            stats = self.resolution_stats.copy()

            # Add current state statistics
            stats.update(
                {
                    "total_candidates": len(self.entity_candidates),
                    "total_clusters": len(self.entity_clusters),
                    "resolution_rate": (
                        self.resolution_stats["resolved_entities"]
                        / max(1, self.resolution_stats["total_mentions"])
                    ),
                    "entity_type_distribution": self._get_entity_type_distribution(),
                    "status_distribution": self._get_status_distribution(),
                }
            )

            return stats

        except Exception as e:
            logger.error(f"Error getting resolution statistics: {e}")
            return self.resolution_stats.copy()

    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types"""
        type_counts = defaultdict(int)
        for candidate in self.entity_candidates.values():
            type_counts[candidate.entity_type.value] += 1
        return dict(type_counts)

    def _get_status_distribution(self) -> Dict[str, int]:
        """Get distribution of resolution statuses"""
        status_counts = defaultdict(int)
        for candidate in self.entity_candidates.values():
            status_counts[candidate.resolution_status.value] += 1
        return dict(status_counts)
