"""
Real-Time Document Intelligence & Compliance Engine
Text Correction & Enhancement Module
Version: 1.0
Date: 2026-02-03

This module provides intelligent OCR text correction using language models,
spell checking, and context-aware error correction to improve text quality.
"""

import logging
import asyncio
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class CorrectionMethod(Enum):
    """Text correction method types"""

    SPELL_CHECK = "SPELL_CHECK"
    LANGUAGE_MODEL = "LANGUAGE_MODEL"
    CONTEXT_AWARE = "CONTEXT_AWARE"
    DICTIONARY_LOOKUP = "DICTIONARY_LOOKUP"
    PATTERN_MATCHING = "PATTERN_MATCHING"


class ConfidenceLevel(Enum):
    """Confidence levels for corrections"""

    HIGH = "HIGH"  # 90-100%
    MEDIUM = "MEDIUM"  # 70-89%
    LOW = "LOW"  # 50-69%
    UNCERTAIN = "UNCERTAIN"  # Below 50%


class TextCorrectionEngine:
    """
    Advanced text correction engine for improving OCR output quality
    """

    def __init__(self, use_language_models: bool = True):
        """
        Initialize text correction engine

        Args:
            use_language_models: Whether to use language models for correction
        """
        self.use_language_models = use_language_models

        # Correction models and services
        self.models = {
            "spell_checker": None,  # Would be loaded spell checking model
            "language_model": None,  # Would be loaded language model (BERT, etc.)
            "grammar_checker": None,  # Grammar correction model
        }

        # Dictionary and pattern resources
        self.dictionaries = {
            "general": set(),  # General English dictionary
            "legal": set(),  # Legal terminology
            "medical": set(),  # Medical terminology
            "financial": set(),  # Financial terminology
            "technical": set(),  # Technical terminology
        }

        # OCR error patterns
        self.error_patterns = {
            # Character substitution patterns
            "char_substitutions": {
                "rn": "m",  # Common OCR error
                "cl": "d",  # Common OCR error
                "li": "h",  # Common OCR error
                "vv": "w",  # Common OCR error
                "nn": "m",  # Common OCR error
                "0": "O",  # Number to letter
                "1": "l",  # Number to letter
                "5": "S",  # Number to letter
            },
            # Word-level common errors
            "word_corrections": {
                "teh": "the",
                "adn": "and",
                "hte": "the",
                "taht": "that",
                "thier": "their",
                "recieve": "receive",
                "seperate": "separate",
                "definately": "definitely",
            },
            # Legal document specific patterns
            "legal_patterns": {
                "WHEREAS": r"\bWHEREAS\b",
                "THEREFORE": r"\bTHEREFORE\b",
                "HEREBY": r"\bHEREBY\b",
                "WITNESSETH": r"\bWITNESSETH\b",
            },
        }

        # Context analysis settings
        self.context_settings = {
            "window_size": 10,  # Words to consider for context
            "min_confidence": 0.6,  # Minimum confidence for corrections
            "max_corrections_per_line": 5,  # Limit corrections per line
        }

        self._initialize_resources()

    def _initialize_resources(self):
        """Initialize dictionaries and correction resources"""
        try:
            # Load basic dictionaries (in real implementation, load from files)
            self.dictionaries["general"] = {
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
                "from",
                "up",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "between",
                "among",
                "under",
                "within",
            }

            self.dictionaries["legal"] = {
                "whereas",
                "therefore",
                "hereby",
                "witnesseth",
                "party",
                "parties",
                "agreement",
                "contract",
                "clause",
                "section",
                "article",
                "provision",
                "consideration",
                "covenant",
                "warranty",
                "liability",
                "indemnify",
                "jurisdiction",
                "governing",
                "breach",
                "default",
                "remedy",
                "damages",
            }

            self.dictionaries["financial"] = {
                "amount",
                "payment",
                "invoice",
                "receipt",
                "transaction",
                "balance",
                "credit",
                "debit",
                "interest",
                "principal",
                "premium",
                "fee",
                "expense",
                "revenue",
                "profit",
                "loss",
                "asset",
                "liability",
            }

            # In real implementation, initialize actual spell checker and language models
            # self.models["spell_checker"] = SpellChecker()
            # self.models["language_model"] = AutoModel.from_pretrained("bert-base-uncased")

            logger.info("Text correction resources initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize correction resources: {e}")

    async def correct_text(
        self,
        text: str,
        ocr_confidence: float = 0.8,
        document_type: str = "general",
        preserve_formatting: bool = True,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive text correction on OCR output

        Args:
            text: Text to correct
            ocr_confidence: Original OCR confidence score
            document_type: Type of document (legal, financial, medical, etc.)
            preserve_formatting: Whether to preserve original formatting

        Returns:
            Correction results with corrected text and metadata
        """
        try:
            correction_start = datetime.utcnow()

            # Initialize correction result
            correction_result = {
                "original_text": text,
                "corrected_text": text,
                "corrections": [],
                "confidence_score": ocr_confidence,
                "correction_metadata": {
                    "methods_applied": [],
                    "total_corrections": 0,
                    "processing_time_ms": 0,
                    "document_type": document_type,
                },
            }

            if not text or len(text.strip()) < 2:
                logger.debug("Skipping correction for empty or very short text")
                return correction_result

            # Apply correction methods in sequence
            corrected_text = text
            all_corrections = []

            # 1. Character-level corrections
            corrected_text, char_corrections = await self._apply_character_corrections(
                corrected_text, document_type
            )
            all_corrections.extend(char_corrections)
            if char_corrections:
                correction_result["correction_metadata"]["methods_applied"].append(
                    "character_correction"
                )

            # 2. Word-level corrections
            corrected_text, word_corrections = await self._apply_word_corrections(
                corrected_text, document_type
            )
            all_corrections.extend(word_corrections)
            if word_corrections:
                correction_result["correction_metadata"]["methods_applied"].append(
                    "word_correction"
                )

            # 3. Context-aware corrections
            if self.use_language_models:
                (
                    corrected_text,
                    context_corrections,
                ) = await self._apply_context_corrections(corrected_text, document_type)
                all_corrections.extend(context_corrections)
                if context_corrections:
                    correction_result["correction_metadata"]["methods_applied"].append(
                        "context_correction"
                    )

            # 4. Grammar and structure corrections
            corrected_text, grammar_corrections = await self._apply_grammar_corrections(
                corrected_text, document_type
            )
            all_corrections.extend(grammar_corrections)
            if grammar_corrections:
                correction_result["correction_metadata"]["methods_applied"].append(
                    "grammar_correction"
                )

            # 5. Document-specific pattern corrections
            corrected_text, pattern_corrections = await self._apply_pattern_corrections(
                corrected_text, document_type
            )
            all_corrections.extend(pattern_corrections)
            if pattern_corrections:
                correction_result["correction_metadata"]["methods_applied"].append(
                    "pattern_correction"
                )

            # Update result
            correction_result["corrected_text"] = corrected_text
            correction_result["corrections"] = all_corrections
            correction_result["correction_metadata"]["total_corrections"] = len(
                all_corrections
            )

            # Calculate overall confidence
            correction_result["confidence_score"] = (
                self._calculate_correction_confidence(ocr_confidence, all_corrections)
            )

            # Calculate processing time
            correction_end = datetime.utcnow()
            correction_result["correction_metadata"]["processing_time_ms"] = int(
                (correction_end - correction_start).total_seconds() * 1000
            )

            logger.info(
                f"Text correction completed: {len(all_corrections)} corrections, "
                f"confidence: {correction_result['confidence_score']:.2f}"
            )

            return correction_result

        except Exception as e:
            logger.error(f"Text correction failed: {e}")
            return {
                "original_text": text,
                "corrected_text": text,
                "error": str(e),
                "confidence_score": ocr_confidence,
            }

    async def _apply_character_corrections(
        self, text: str, document_type: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply character-level corrections for common OCR errors"""
        try:
            corrections = []
            corrected_text = text

            # Apply character substitutions
            for error_char, correct_char in self.error_patterns[
                "char_substitutions"
            ].items():
                if error_char in corrected_text:
                    # Find all occurrences and evaluate context
                    pattern = re.escape(error_char)
                    matches = list(re.finditer(pattern, corrected_text))

                    for match in reversed(
                        matches
                    ):  # Process in reverse to maintain positions
                        start, end = match.span()
                        context = self._get_character_context(
                            corrected_text, start, end
                        )

                        # Decide if this is likely an error
                        if self._should_correct_character(
                            error_char, correct_char, context
                        ):
                            corrected_text = (
                                corrected_text[:start]
                                + correct_char
                                + corrected_text[end:]
                            )

                            corrections.append(
                                {
                                    "type": "character",
                                    "method": CorrectionMethod.PATTERN_MATCHING.value,
                                    "original": error_char,
                                    "corrected": correct_char,
                                    "position": start,
                                    "confidence": 0.8,
                                    "context": context,
                                }
                            )

            return corrected_text, corrections

        except Exception as e:
            logger.warning(f"Character correction failed: {e}")
            return text, []

    def _get_character_context(
        self, text: str, start: int, end: int, window: int = 5
    ) -> str:
        """Get character context around position"""
        try:
            context_start = max(0, start - window)
            context_end = min(len(text), end + window)
            return text[context_start:context_end]
        except Exception:
            return ""

    def _should_correct_character(
        self, error_char: str, correct_char: str, context: str
    ) -> bool:
        """Determine if character should be corrected based on context"""
        try:
            # Simple heuristic: check if correction results in known words
            words_in_context = context.split()

            for word in words_in_context:
                if error_char in word:
                    corrected_word = word.replace(error_char, correct_char)
                    # Check if corrected word is in dictionary
                    if (
                        corrected_word.lower() in self.dictionaries["general"]
                        or corrected_word.lower() in self.dictionaries["legal"]
                    ):
                        return True

            return False

        except Exception:
            return False

    async def _apply_word_corrections(
        self, text: str, document_type: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply word-level corrections for common misspellings"""
        try:
            corrections = []
            corrected_text = text

            # Apply word-level corrections
            words = text.split()
            corrected_words = []

            for i, word in enumerate(words):
                original_word = word
                cleaned_word = re.sub(r"[^\w]", "", word.lower())

                # Check for direct word corrections
                if cleaned_word in self.error_patterns["word_corrections"]:
                    corrected_word = self.error_patterns["word_corrections"][
                        cleaned_word
                    ]

                    # Preserve original casing and punctuation
                    if word.isupper():
                        corrected_word = corrected_word.upper()
                    elif word.istitle():
                        corrected_word = corrected_word.title()

                    # Add back punctuation
                    punctuation = re.findall(r"[^\w]", word)
                    if punctuation:
                        corrected_word += "".join(punctuation)

                    corrected_words.append(corrected_word)

                    corrections.append(
                        {
                            "type": "word",
                            "method": CorrectionMethod.DICTIONARY_LOOKUP.value,
                            "original": original_word,
                            "corrected": corrected_word,
                            "position": i,
                            "confidence": 0.9,
                            "context": self._get_word_context(words, i),
                        }
                    )

                else:
                    # Check for potential spelling errors using edit distance
                    suggestion = self._find_spelling_suggestion(
                        cleaned_word, document_type
                    )
                    if suggestion and suggestion != cleaned_word:
                        # Apply same casing and punctuation logic
                        corrected_word = suggestion
                        if word.isupper():
                            corrected_word = corrected_word.upper()
                        elif word.istitle():
                            corrected_word = corrected_word.title()

                        punctuation = re.findall(r"[^\w]", word)
                        if punctuation:
                            corrected_word += "".join(punctuation)

                        corrected_words.append(corrected_word)

                        corrections.append(
                            {
                                "type": "spelling",
                                "method": CorrectionMethod.SPELL_CHECK.value,
                                "original": original_word,
                                "corrected": corrected_word,
                                "position": i,
                                "confidence": 0.7,
                                "context": self._get_word_context(words, i),
                            }
                        )
                    else:
                        corrected_words.append(original_word)

            corrected_text = " ".join(corrected_words)
            return corrected_text, corrections

        except Exception as e:
            logger.warning(f"Word correction failed: {e}")
            return text, []

    def _get_word_context(
        self, words: List[str], position: int, window: int = 3
    ) -> str:
        """Get word context around position"""
        try:
            start = max(0, position - window)
            end = min(len(words), position + window + 1)
            return " ".join(words[start:end])
        except Exception:
            return ""

    def _find_spelling_suggestion(self, word: str, document_type: str) -> Optional[str]:
        """Find spelling suggestions for misspelled words"""
        try:
            # Get relevant dictionary based on document type
            target_dict = self.dictionaries.get(
                document_type, self.dictionaries["general"]
            )

            # Combine with general dictionary
            combined_dict = target_dict.union(self.dictionaries["general"])

            # Simple edit distance calculation
            best_match = None
            best_distance = float("inf")

            for dict_word in combined_dict:
                if len(word) > 2 and len(dict_word) > 2:
                    distance = self._edit_distance(word, dict_word)
                    # Only suggest if word is reasonably close
                    if distance < len(word) * 0.3 and distance < best_distance:
                        best_distance = distance
                        best_match = dict_word

            return best_match if best_distance <= 2 else None

        except Exception:
            return None

    def _edit_distance(self, word1: str, word2: str) -> int:
        """Calculate Levenshtein edit distance between two words"""
        try:
            m, n = len(word1), len(word2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            # Initialize base cases
            for i in range(m + 1):
                dp[i][0] = i
            for j in range(n + 1):
                dp[0][j] = j

            # Fill the dp table
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if word1[i - 1] == word2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(
                            dp[i - 1][j],  # deletion
                            dp[i][j - 1],  # insertion
                            dp[i - 1][j - 1],
                        )  # substitution

            return dp[m][n]

        except Exception:
            return float("inf")

    async def _apply_context_corrections(
        self, text: str, document_type: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply context-aware corrections using language models"""
        try:
            corrections = []

            # Simulate language model-based corrections
            if self.use_language_models:
                # In real implementation, use actual language model
                await asyncio.sleep(0.1)  # Simulate model inference

                # Mock context corrections for demonstration
                sentences = text.split(".")
                corrected_sentences = []

                for sentence in sentences:
                    corrected_sentence = sentence

                    # Simple context-based corrections
                    if "teh contract" in sentence.lower():
                        corrected_sentence = sentence.replace("teh", "the")
                        corrections.append(
                            {
                                "type": "context",
                                "method": CorrectionMethod.CONTEXT_AWARE.value,
                                "original": "teh",
                                "corrected": "the",
                                "confidence": 0.95,
                                "context": sentence.strip(),
                            }
                        )

                    corrected_sentences.append(corrected_sentence)

                return ".".join(corrected_sentences), corrections

            return text, corrections

        except Exception as e:
            logger.warning(f"Context correction failed: {e}")
            return text, []

    async def _apply_grammar_corrections(
        self, text: str, document_type: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply grammar and structure corrections"""
        try:
            corrections = []
            corrected_text = text

            # Fix common grammar issues
            # 1. Multiple spaces
            if re.search(r"  +", corrected_text):
                corrected_text = re.sub(r"  +", " ", corrected_text)
                corrections.append(
                    {
                        "type": "grammar",
                        "method": CorrectionMethod.PATTERN_MATCHING.value,
                        "original": "multiple spaces",
                        "corrected": "single space",
                        "confidence": 1.0,
                        "description": "Removed extra spaces",
                    }
                )

            # 2. Missing spaces after punctuation
            pattern = r"([.!?])([A-Za-z])"
            if re.search(pattern, corrected_text):
                corrected_text = re.sub(pattern, r"\1 \2", corrected_text)
                corrections.append(
                    {
                        "type": "grammar",
                        "method": CorrectionMethod.PATTERN_MATCHING.value,
                        "original": "missing space after punctuation",
                        "corrected": "added space after punctuation",
                        "confidence": 0.9,
                        "description": "Added space after punctuation",
                    }
                )

            # 3. Incorrect capitalization after periods
            pattern = r"(\. )([a-z])"
            if re.search(pattern, corrected_text):

                def capitalize_match(match):
                    return match.group(1) + match.group(2).upper()

                corrected_text = re.sub(pattern, capitalize_match, corrected_text)
                corrections.append(
                    {
                        "type": "grammar",
                        "method": CorrectionMethod.PATTERN_MATCHING.value,
                        "original": "lowercase after period",
                        "corrected": "capitalized after period",
                        "confidence": 0.85,
                        "description": "Capitalized word after period",
                    }
                )

            return corrected_text, corrections

        except Exception as e:
            logger.warning(f"Grammar correction failed: {e}")
            return text, []

    async def _apply_pattern_corrections(
        self, text: str, document_type: str
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Apply document-specific pattern corrections"""
        try:
            corrections = []
            corrected_text = text

            # Apply legal document patterns
            if document_type in ["legal", "contract"]:
                for term, pattern in self.error_patterns["legal_patterns"].items():
                    # Find potential matches with slight variations
                    variations = [
                        term.lower(),
                        term.title(),
                        term.upper(),
                        term.replace("E", "3"),  # Common OCR error
                        term.replace("A", "4"),  # Common OCR error
                        term.replace("S", "5"),  # Common OCR error
                    ]

                    for variation in variations:
                        if variation != term and variation in corrected_text:
                            corrected_text = corrected_text.replace(variation, term)
                            corrections.append(
                                {
                                    "type": "pattern",
                                    "method": CorrectionMethod.PATTERN_MATCHING.value,
                                    "original": variation,
                                    "corrected": term,
                                    "confidence": 0.8,
                                    "document_type": document_type,
                                    "description": f"Corrected legal term: {term}",
                                }
                            )

            # Apply financial document patterns
            elif document_type == "financial":
                # Fix currency symbols and amounts
                currency_pattern = r"\$(\d+)[.,](\d{2})"
                matches = re.finditer(currency_pattern, corrected_text)

                for match in matches:
                    original = match.group(0)
                    amount = f"${match.group(1)}.{match.group(2)}"
                    if original != amount:
                        corrected_text = corrected_text.replace(original, amount)
                        corrections.append(
                            {
                                "type": "pattern",
                                "method": CorrectionMethod.PATTERN_MATCHING.value,
                                "original": original,
                                "corrected": amount,
                                "confidence": 0.9,
                                "document_type": document_type,
                                "description": "Standardized currency format",
                            }
                        )

            return corrected_text, corrections

        except Exception as e:
            logger.warning(f"Pattern correction failed: {e}")
            return text, []

    def _calculate_correction_confidence(
        self, original_confidence: float, corrections: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence after corrections"""
        try:
            if not corrections:
                return original_confidence

            # Base confidence improvement from corrections
            correction_boost = 0.0
            total_corrections = len(corrections)

            # Calculate weighted confidence boost
            for correction in corrections:
                correction_confidence = correction.get("confidence", 0.5)
                method = correction.get("method", "")

                # Weight corrections by method reliability
                if method == CorrectionMethod.DICTIONARY_LOOKUP.value:
                    correction_boost += 0.1 * correction_confidence
                elif method == CorrectionMethod.PATTERN_MATCHING.value:
                    correction_boost += 0.08 * correction_confidence
                elif method == CorrectionMethod.CONTEXT_AWARE.value:
                    correction_boost += 0.12 * correction_confidence
                elif method == CorrectionMethod.SPELL_CHECK.value:
                    correction_boost += 0.06 * correction_confidence

            # Cap the boost to prevent over-confidence
            max_boost = 0.15 if total_corrections < 5 else 0.20
            correction_boost = min(correction_boost, max_boost)

            # Calculate final confidence
            final_confidence = min(1.0, original_confidence + correction_boost)

            return final_confidence

        except Exception:
            return original_confidence

    async def correct_batch_text(
        self, text_blocks: List[Dict[str, Any]], document_type: str = "general"
    ) -> List[Dict[str, Any]]:
        """Correct multiple text blocks in batch"""
        try:
            corrected_blocks = []

            for block in text_blocks:
                text = block.get("text", "")
                confidence = block.get("confidence", 0.8)

                if text.strip():
                    correction_result = await self.correct_text(
                        text, confidence, document_type
                    )

                    corrected_block = block.copy()
                    corrected_block.update(
                        {
                            "corrected_text": correction_result["corrected_text"],
                            "corrections": correction_result["corrections"],
                            "correction_confidence": correction_result[
                                "confidence_score"
                            ],
                        }
                    )

                    corrected_blocks.append(corrected_block)
                else:
                    corrected_blocks.append(block)

            logger.info(
                f"Batch correction completed for {len(corrected_blocks)} text blocks"
            )
            return corrected_blocks

        except Exception as e:
            logger.error(f"Batch text correction failed: {e}")
            return text_blocks

    def get_correction_statistics(
        self, corrections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate correction statistics"""
        try:
            if not corrections:
                return {"total_corrections": 0}

            stats = {
                "total_corrections": len(corrections),
                "by_type": {},
                "by_method": {},
                "avg_confidence": 0.0,
                "confidence_distribution": {
                    "high": 0,  # >= 0.9
                    "medium": 0,  # 0.7-0.89
                    "low": 0,  # 0.5-0.69
                    "uncertain": 0,  # < 0.5
                },
            }

            # Count by type and method
            for correction in corrections:
                correction_type = correction.get("type", "unknown")
                method = correction.get("method", "unknown")
                confidence = correction.get("confidence", 0.0)

                stats["by_type"][correction_type] = (
                    stats["by_type"].get(correction_type, 0) + 1
                )
                stats["by_method"][method] = stats["by_method"].get(method, 0) + 1

                # Confidence distribution
                if confidence >= 0.9:
                    stats["confidence_distribution"]["high"] += 1
                elif confidence >= 0.7:
                    stats["confidence_distribution"]["medium"] += 1
                elif confidence >= 0.5:
                    stats["confidence_distribution"]["low"] += 1
                else:
                    stats["confidence_distribution"]["uncertain"] += 1

            # Calculate average confidence
            confidences = [c.get("confidence", 0.0) for c in corrections]
            stats["avg_confidence"] = sum(confidences) / len(confidences)

            return stats

        except Exception as e:
            logger.warning(f"Failed to generate correction statistics: {e}")
            return {"error": str(e)}
