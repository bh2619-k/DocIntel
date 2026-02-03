"""
Real-Time Document Intelligence & Compliance Engine
T5-Based Clause Extraction Module
Version: 1.0
Date: 2026-02-03

This module provides T5-based clause extraction for document-level analysis,
extracting key clauses, terms, and conditions with context understanding.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class ClauseType(Enum):
    """Types of clauses to extract"""

    # Legal clauses
    LIABILITY = "LIABILITY"
    INDEMNIFICATION = "INDEMNIFICATION"
    TERMINATION = "TERMINATION"
    CONFIDENTIALITY = "CONFIDENTIALITY"
    INTELLECTUAL_PROPERTY = "INTELLECTUAL_PROPERTY"
    FORCE_MAJEURE = "FORCE_MAJEURE"
    GOVERNING_LAW = "GOVERNING_LAW"
    DISPUTE_RESOLUTION = "DISPUTE_RESOLUTION"

    # Commercial clauses
    PAYMENT_TERMS = "PAYMENT_TERMS"
    DELIVERY_TERMS = "DELIVERY_TERMS"
    WARRANTY = "WARRANTY"
    SERVICE_LEVEL = "SERVICE_LEVEL"
    PRICING = "PRICING"
    RENEWAL = "RENEWAL"

    # Compliance clauses
    REGULATORY_COMPLIANCE = "REGULATORY_COMPLIANCE"
    DATA_PROTECTION = "DATA_PROTECTION"
    AUDIT_RIGHTS = "AUDIT_RIGHTS"
    REPORTING_REQUIREMENTS = "REPORTING_REQUIREMENTS"

    # General clauses
    DEFINITIONS = "DEFINITIONS"
    SCOPE_OF_WORK = "SCOPE_OF_WORK"
    MISCELLANEOUS = "MISCELLANEOUS"


class ExtractionMode(Enum):
    """Clause extraction modes"""

    GENERATIVE = "GENERATIVE"  # Generate clause summaries
    EXTRACTIVE = "EXTRACTIVE"  # Extract exact text spans
    HYBRID = "HYBRID"  # Combine both approaches


class RelevanceLevel(Enum):
    """Clause relevance levels"""

    CRITICAL = "CRITICAL"  # Must be reviewed
    HIGH = "HIGH"  # Important for review
    MEDIUM = "MEDIUM"  # Standard importance
    LOW = "LOW"  # Informational


class T5ClauseExtractor(nn.Module):
    """
    T5-based clause extraction system for document-level analysis
    """

    def __init__(
        self,
        model_variant: str = "t5-base",
        extraction_mode: ExtractionMode = ExtractionMode.HYBRID,
        max_input_length: int = 1024,
        max_output_length: int = 256,
        relevance_threshold: float = 0.7,
        cache_dir: str = "./model_cache",
    ):
        """
        Initialize T5 clause extractor

        Args:
            model_variant: T5 model variant (t5-base, t5-large, etc.)
            extraction_mode: Mode for clause extraction
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
            relevance_threshold: Minimum relevance score for extraction
            cache_dir: Model cache directory
        """
        super().__init__()

        self.model_variant = model_variant
        self.extraction_mode = extraction_mode
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.relevance_threshold = relevance_threshold
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model components
        self.tokenizer = None
        self.t5_model = None

        # Clause extraction prompts
        self.extraction_prompts = self._initialize_extraction_prompts()

        # Post-processing patterns
        self.clause_patterns = self._initialize_clause_patterns()

        # Performance metrics
        self.metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "average_relevance_score": 0.0,
            "clause_type_counts": {clause.value: 0 for clause in ClauseType},
            "average_processing_time": 0.0,
        }

        # Model state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_initialized = False

        # Initialize model
        self._initialize_model()

        logger.info(f"T5 clause extractor initialized with variant: {model_variant}")

    def _initialize_model(self):
        """Initialize T5 model and tokenizer"""
        try:
            # Mock initialization - in real implementation would load actual T5 model
            # from transformers import T5Tokenizer, T5ForConditionalGeneration
            # self.tokenizer = T5Tokenizer.from_pretrained(self.model_variant)
            # self.t5_model = T5ForConditionalGeneration.from_pretrained(self.model_variant)

            # Mock T5 components
            self.tokenizer = MockT5Tokenizer(self.model_variant)
            self.t5_model = MockT5Model(
                model_name=self.model_variant,
                max_input_length=self.max_input_length,
                max_output_length=self.max_output_length,
            )

            # Move to device
            self.t5_model = self.t5_model.to(self.device)

            self.is_initialized = True
            logger.info(f"T5 model loaded on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize T5 model: {e}")
            raise

    def _initialize_extraction_prompts(self) -> Dict[str, str]:
        """Initialize extraction prompts for different clause types"""
        prompts = {}

        # Legal clause prompts
        prompts[ClauseType.LIABILITY.value] = (
            "Extract the liability clause from this document. "
            "Focus on limitations of liability, caps, and exclusions: "
        )

        prompts[ClauseType.INDEMNIFICATION.value] = (
            "Extract the indemnification clause from this document. "
            "Include who indemnifies whom and for what: "
        )

        prompts[ClauseType.TERMINATION.value] = (
            "Extract the termination clause from this document. "
            "Include termination conditions and notice periods: "
        )

        prompts[ClauseType.CONFIDENTIALITY.value] = (
            "Extract the confidentiality/non-disclosure clause from this document. "
            "Include what constitutes confidential information: "
        )

        # Commercial clause prompts
        prompts[ClauseType.PAYMENT_TERMS.value] = (
            "Extract the payment terms from this document. "
            "Include payment amounts, schedules, and conditions: "
        )

        prompts[ClauseType.WARRANTY.value] = (
            "Extract the warranty clause from this document. "
            "Include warranties provided and disclaimed: "
        )

        # Compliance clause prompts
        prompts[ClauseType.REGULATORY_COMPLIANCE.value] = (
            "Extract regulatory compliance requirements from this document. "
            "Include specific regulations and compliance obligations: "
        )

        prompts[ClauseType.DATA_PROTECTION.value] = (
            "Extract data protection and privacy clauses from this document. "
            "Include data handling requirements and protections: "
        )

        # General prompts
        prompts[ClauseType.DEFINITIONS.value] = (
            "Extract key definitions from this document. "
            "List important terms and their definitions: "
        )

        prompts[ClauseType.SCOPE_OF_WORK.value] = (
            "Extract the scope of work or services from this document. "
            "Include what will be delivered or performed: "
        )

        return prompts

    def _initialize_clause_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for clause identification"""
        patterns = {}

        # Liability patterns
        patterns[ClauseType.LIABILITY.value] = [
            r"liability\s+(?:shall\s+be\s+)?limited",
            r"limitation\s+of\s+liability",
            r"in\s+no\s+event\s+shall.*liable",
            r"aggregate\s+liability.*shall\s+not\s+exceed",
            r"consequential\s+damages.*excluded",
        ]

        # Indemnification patterns
        patterns[ClauseType.INDEMNIFICATION.value] = [
            r"shall\s+indemnify",
            r"indemnification\s+obligations",
            r"hold\s+harmless",
            r"defend,?\s+indemnify,?\s+and\s+hold\s+harmless",
        ]

        # Termination patterns
        patterns[ClauseType.TERMINATION.value] = [
            r"terminate\s+this\s+agreement",
            r"termination\s+(?:for\s+)?(?:cause|convenience)",
            r"(?:thirty|60|90)\s+days?\s+(?:written\s+)?notice",
            r"agreement\s+may\s+be\s+terminated",
        ]

        # Payment patterns
        patterns[ClauseType.PAYMENT_TERMS.value] = [
            r"payment\s+(?:shall\s+be\s+)?due",
            r"invoice.*within.*days",
            r"net\s+\d+\s+days",
            r"payment\s+terms",
        ]

        # Confidentiality patterns
        patterns[ClauseType.CONFIDENTIALITY.value] = [
            r"confidential\s+information",
            r"non-disclosure",
            r"proprietary\s+information",
            r"shall\s+not\s+disclose",
        ]

        return patterns

    def extract_clauses(
        self,
        document_text: str,
        clause_types: Optional[List[ClauseType]] = None,
        document_context: Optional[Dict[str, Any]] = None,
        return_relevance_scores: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract clauses from document text

        Args:
            document_text: Full document text
            clause_types: Specific clause types to extract (None for all)
            document_context: Additional context about the document
            return_relevance_scores: Whether to return relevance scores

        Returns:
            Extracted clauses with metadata
        """
        try:
            extraction_start = datetime.now()

            # Initialize result structure
            result = {
                "extracted_clauses": [],
                "document_summary": "",
                "extraction_metadata": {
                    "total_clauses": 0,
                    "processing_time_ms": 0,
                    "extraction_mode": self.extraction_mode.value,
                    "model_variant": self.model_variant,
                },
            }

            if not document_text.strip():
                return result

            # Determine clause types to extract
            if clause_types is None:
                clause_types = list(ClauseType)

            # Preprocess document
            processed_text = self._preprocess_document(document_text)

            # Extract document summary
            result["document_summary"] = self._generate_document_summary(processed_text)

            # Extract clauses by type
            for clause_type in clause_types:
                clause_result = self._extract_clause_type(
                    processed_text,
                    clause_type,
                    document_context,
                    return_relevance_scores,
                )

                if clause_result:
                    result["extracted_clauses"].extend(clause_result)

            # Post-process and rank clauses
            result["extracted_clauses"] = self._post_process_clauses(
                result["extracted_clauses"], document_text
            )

            # Update metadata
            result["extraction_metadata"]["total_clauses"] = len(
                result["extracted_clauses"]
            )

            # Calculate processing time
            processing_time = (datetime.now() - extraction_start).total_seconds() * 1000
            result["extraction_metadata"]["processing_time_ms"] = processing_time

            # Update metrics
            self._update_metrics(result["extracted_clauses"], processing_time)

            logger.info(
                f"Extracted {len(result['extracted_clauses'])} clauses in {processing_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Clause extraction failed: {e}")
            return {
                "extracted_clauses": [],
                "error": str(e),
                "extraction_metadata": {"total_clauses": 0},
            }

    def _preprocess_document(self, document_text: str) -> str:
        """Preprocess document text for clause extraction"""
        try:
            # Clean up text
            text = document_text.strip()

            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove page numbers and headers/footers
            text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text, flags=re.IGNORECASE)

            # Remove confidentiality notices at top/bottom
            text = re.sub(
                r"^.*CONFIDENTIAL.*?(?=\n|$)",
                "",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )

            # Normalize paragraph breaks
            text = re.sub(r"\n\s*\n", "\n\n", text)

            return text

        except Exception:
            return document_text

    def _generate_document_summary(self, document_text: str) -> str:
        """Generate document summary using T5"""
        try:
            # Prepare summarization prompt
            summary_prompt = (
                f"Summarize this document in 2-3 sentences: {document_text[:2000]}"
            )

            # Generate summary using T5
            summary = self._generate_with_t5(
                summary_prompt, max_length=150, temperature=0.7
            )

            return summary.strip() if summary else "Document summary not available."

        except Exception as e:
            logger.warning(f"Document summary generation failed: {e}")
            return "Document summary not available."

    def _extract_clause_type(
        self,
        document_text: str,
        clause_type: ClauseType,
        document_context: Optional[Dict[str, Any]],
        return_relevance_scores: bool,
    ) -> List[Dict[str, Any]]:
        """Extract specific clause type from document"""
        try:
            extracted_clauses = []

            # Get extraction prompt for clause type
            prompt = self.extraction_prompts.get(clause_type.value, "")
            if not prompt:
                return []

            # Extractive approach: Find relevant sections first
            relevant_sections = self._find_relevant_sections(document_text, clause_type)

            for section in relevant_sections:
                # Generate clause extraction
                extraction_prompt = f"{prompt}{section['text']}"

                if self.extraction_mode in [
                    ExtractionMode.GENERATIVE,
                    ExtractionMode.HYBRID,
                ]:
                    # Generative extraction
                    generated_clause = self._generate_with_t5(
                        extraction_prompt,
                        max_length=self.max_output_length,
                        temperature=0.3,
                    )

                    if generated_clause and len(generated_clause.strip()) > 20:
                        clause_data = {
                            "clause_type": clause_type.value,
                            "extraction_method": "generative",
                            "generated_text": generated_clause.strip(),
                            "source_section": section,
                            "relevance_level": self._determine_relevance_level(
                                generated_clause, clause_type
                            ),
                        }

                        if return_relevance_scores:
                            clause_data["relevance_score"] = section.get(
                                "relevance_score", 0.5
                            )

                        extracted_clauses.append(clause_data)

                if self.extraction_mode in [
                    ExtractionMode.EXTRACTIVE,
                    ExtractionMode.HYBRID,
                ]:
                    # Extractive approach: identify specific clause text
                    extracted_spans = self._extract_clause_spans(
                        section["text"], clause_type
                    )

                    for span in extracted_spans:
                        clause_data = {
                            "clause_type": clause_type.value,
                            "extraction_method": "extractive",
                            "extracted_text": span["text"],
                            "start_position": span["start"]
                            + section.get("start_position", 0),
                            "end_position": span["end"]
                            + section.get("start_position", 0),
                            "relevance_level": self._determine_relevance_level(
                                span["text"], clause_type
                            ),
                        }

                        if return_relevance_scores:
                            clause_data["relevance_score"] = span.get("confidence", 0.5)

                        extracted_clauses.append(clause_data)

            return extracted_clauses

        except Exception as e:
            logger.warning(f"Clause extraction for {clause_type.value} failed: {e}")
            return []

    def _find_relevant_sections(
        self, document_text: str, clause_type: ClauseType
    ) -> List[Dict[str, Any]]:
        """Find document sections relevant to clause type"""
        try:
            relevant_sections = []
            patterns = self.clause_patterns.get(clause_type.value, [])

            # Split document into paragraphs
            paragraphs = document_text.split("\n\n")
            current_position = 0

            for paragraph in paragraphs:
                paragraph = paragraph.strip()
                if not paragraph:
                    current_position += 2  # Account for double newline
                    continue

                # Check if paragraph contains relevant patterns
                relevance_score = 0.0
                pattern_matches = 0

                for pattern in patterns:
                    matches = re.findall(pattern, paragraph, re.IGNORECASE)
                    if matches:
                        pattern_matches += len(matches)
                        relevance_score += 0.3 * len(matches)

                # Check for clause type keywords
                keywords = self._get_clause_keywords(clause_type)
                for keyword in keywords:
                    if keyword.lower() in paragraph.lower():
                        relevance_score += 0.1

                # Include section if relevance score is above threshold
                if relevance_score >= self.relevance_threshold or pattern_matches > 0:
                    relevant_sections.append(
                        {
                            "text": paragraph,
                            "start_position": current_position,
                            "end_position": current_position + len(paragraph),
                            "relevance_score": min(1.0, relevance_score),
                            "pattern_matches": pattern_matches,
                        }
                    )

                current_position += len(paragraph) + 2

            # Sort by relevance score
            relevant_sections.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Return top sections (limit to avoid processing too much text)
            return relevant_sections[:5]

        except Exception as e:
            logger.warning(f"Failed to find relevant sections: {e}")
            return [
                {
                    "text": document_text[:2000],
                    "relevance_score": 0.5,
                    "start_position": 0,
                }
            ]

    def _get_clause_keywords(self, clause_type: ClauseType) -> List[str]:
        """Get keywords associated with clause type"""
        keywords = {
            ClauseType.LIABILITY: [
                "liability",
                "liable",
                "damages",
                "limitation",
                "exclude",
            ],
            ClauseType.INDEMNIFICATION: [
                "indemnify",
                "indemnification",
                "hold harmless",
                "defend",
            ],
            ClauseType.TERMINATION: [
                "terminate",
                "termination",
                "end",
                "expire",
                "cancel",
            ],
            ClauseType.CONFIDENTIALITY: [
                "confidential",
                "non-disclosure",
                "proprietary",
                "secret",
            ],
            ClauseType.PAYMENT_TERMS: ["payment", "pay", "invoice", "due", "remit"],
            ClauseType.WARRANTY: ["warranty", "warrant", "guarantee", "represent"],
            ClauseType.GOVERNING_LAW: [
                "governing law",
                "jurisdiction",
                "court",
                "venue",
            ],
            ClauseType.DATA_PROTECTION: [
                "data",
                "privacy",
                "personal information",
                "GDPR",
            ],
        }

        return keywords.get(clause_type, [clause_type.value.lower().replace("_", " ")])

    def _extract_clause_spans(
        self, text: str, clause_type: ClauseType
    ) -> List[Dict[str, Any]]:
        """Extract specific clause text spans from section"""
        try:
            spans = []
            patterns = self.clause_patterns.get(clause_type.value, [])

            for pattern in patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))

                for match in matches:
                    # Extend match to include full sentence/clause
                    start = match.start()
                    end = match.end()

                    # Find sentence boundaries
                    sentence_start = text.rfind(".", 0, start)
                    if sentence_start == -1:
                        sentence_start = 0
                    else:
                        sentence_start += 1

                    sentence_end = text.find(".", end)
                    if sentence_end == -1:
                        sentence_end = len(text)
                    else:
                        sentence_end += 1

                    extracted_text = text[sentence_start:sentence_end].strip()

                    if len(extracted_text) > 20:  # Minimum clause length
                        spans.append(
                            {
                                "text": extracted_text,
                                "start": sentence_start,
                                "end": sentence_end,
                                "confidence": 0.8,  # Pattern-based confidence
                                "pattern": pattern,
                            }
                        )

            return spans

        except Exception as e:
            logger.warning(f"Failed to extract clause spans: {e}")
            return []

    def _generate_with_t5(
        self, prompt: str, max_length: int = 256, temperature: float = 0.7
    ) -> str:
        """Generate text using T5 model"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                prompt, max_length=self.max_input_length, truncation=True
            )

            # Generate with T5 model
            with torch.no_grad():
                outputs = self.t5_model.generate(
                    input_ids=torch.tensor([inputs]).to(self.device),
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True if temperature > 0 else False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return generated_text

        except Exception as e:
            logger.warning(f"T5 generation failed: {e}")
            return ""

    def _determine_relevance_level(
        self, clause_text: str, clause_type: ClauseType
    ) -> str:
        """Determine relevance level of extracted clause"""
        try:
            text_lower = clause_text.lower()

            # Critical clauses - legal/financial impact
            critical_indicators = [
                "shall",
                "must",
                "required",
                "liable",
                "penalty",
                "breach",
                "indemnify",
                "terminate",
                "damages",
                "governing law",
            ]

            # High importance indicators
            high_indicators = [
                "may",
                "should",
                "warranty",
                "confidential",
                "payment",
                "delivery",
                "compliance",
                "audit",
            ]

            critical_count = sum(
                1 for indicator in critical_indicators if indicator in text_lower
            )
            high_count = sum(
                1 for indicator in high_indicators if indicator in text_lower
            )

            # Determine relevance level
            if critical_count >= 2 or "shall" in text_lower:
                return RelevanceLevel.CRITICAL.value
            elif critical_count >= 1 or high_count >= 2:
                return RelevanceLevel.HIGH.value
            elif high_count >= 1:
                return RelevanceLevel.MEDIUM.value
            else:
                return RelevanceLevel.LOW.value

        except Exception:
            return RelevanceLevel.MEDIUM.value

    def _post_process_clauses(
        self, clauses: List[Dict[str, Any]], document_text: str
    ) -> List[Dict[str, Any]]:
        """Post-process extracted clauses"""
        try:
            processed_clauses = []

            for clause in clauses:
                # Clean clause text
                if "generated_text" in clause:
                    clause["generated_text"] = self._clean_clause_text(
                        clause["generated_text"]
                    )
                if "extracted_text" in clause:
                    clause["extracted_text"] = self._clean_clause_text(
                        clause["extracted_text"]
                    )

                # Add clause analysis
                clause_text = clause.get("generated_text") or clause.get(
                    "extracted_text", ""
                )
                clause["analysis"] = self._analyze_clause(
                    clause_text, clause["clause_type"]
                )

                # Add risk assessment
                clause["risk_level"] = self._assess_clause_risk(
                    clause_text, clause["clause_type"]
                )

                processed_clauses.append(clause)

            # Remove duplicates
            processed_clauses = self._remove_duplicate_clauses(processed_clauses)

            # Sort by relevance and importance
            processed_clauses.sort(
                key=lambda x: (
                    self._get_relevance_level_score(x.get("relevance_level", "MEDIUM")),
                    x.get("relevance_score", 0.5),
                ),
                reverse=True,
            )

            return processed_clauses

        except Exception as e:
            logger.warning(f"Clause post-processing failed: {e}")
            return clauses

    def _clean_clause_text(self, text: str) -> str:
        """Clean and normalize clause text"""
        try:
            # Remove excessive whitespace
            text = re.sub(r"\s+", " ", text)

            # Remove artifacts from generation
            text = re.sub(
                r"^(extract|clause|the following)[:,\s]*", "", text, flags=re.IGNORECASE
            )

            # Ensure proper capitalization
            if text and not text[0].isupper():
                text = text[0].upper() + text[1:]

            return text.strip()

        except Exception:
            return text

    def _analyze_clause(self, clause_text: str, clause_type: str) -> Dict[str, Any]:
        """Analyze clause for key components"""
        try:
            analysis = {
                "word_count": len(clause_text.split()),
                "key_terms": [],
                "obligations": [],
                "conditions": [],
            }

            # Extract key terms (simplified)
            if clause_type == "LIABILITY":
                key_terms = re.findall(r"\$[\d,]+", clause_text)  # Dollar amounts
                analysis["key_terms"] = key_terms

            elif clause_type == "PAYMENT_TERMS":
                payment_terms = re.findall(r"\d+\s+days?", clause_text)
                analysis["key_terms"] = payment_terms

            # Identify obligations (simplified)
            obligations = re.findall(r"shall\s+([^.]*)", clause_text, re.IGNORECASE)
            analysis["obligations"] = [o.strip() for o in obligations[:3]]  # Limit to 3

            # Identify conditions
            conditions = re.findall(
                r"(?:if|unless|provided that)\s+([^.]*)", clause_text, re.IGNORECASE
            )
            analysis["conditions"] = [c.strip() for c in conditions[:3]]  # Limit to 3

            return analysis

        except Exception:
            return {
                "word_count": 0,
                "key_terms": [],
                "obligations": [],
                "conditions": [],
            }

    def _assess_clause_risk(self, clause_text: str, clause_type: str) -> str:
        """Assess risk level of clause"""
        try:
            text_lower = clause_text.lower()

            # High risk indicators
            high_risk_terms = [
                "unlimited liability",
                "personal guarantee",
                "liquidated damages",
                "immediate termination",
                "no limitation",
                "gross negligence",
            ]

            # Medium risk indicators
            medium_risk_terms = [
                "limited liability",
                "indemnification",
                "confidentiality",
                "termination for cause",
                "material breach",
            ]

            if any(term in text_lower for term in high_risk_terms):
                return "HIGH"
            elif any(term in text_lower for term in medium_risk_terms):
                return "MEDIUM"
            else:
                return "LOW"

        except Exception:
            return "MEDIUM"

    def _remove_duplicate_clauses(
        self, clauses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate clauses based on text similarity"""
        try:
            unique_clauses = []
            seen_texts = set()

            for clause in clauses:
                clause_text = clause.get("generated_text") or clause.get(
                    "extracted_text", ""
                )

                # Simple deduplication based on text similarity
                text_words = set(clause_text.lower().split())

                is_duplicate = False
                for seen_text in seen_texts:
                    seen_words = set(seen_text.lower().split())

                    # Check for high overlap
                    if (
                        len(text_words & seen_words)
                        / max(len(text_words), len(seen_words))
                        > 0.8
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    unique_clauses.append(clause)
                    seen_texts.add(clause_text)

            return unique_clauses

        except Exception:
            return clauses

    def _get_relevance_level_score(self, relevance_level: str) -> int:
        """Convert relevance level to numeric score for sorting"""
        scores = {
            RelevanceLevel.CRITICAL.value: 4,
            RelevanceLevel.HIGH.value: 3,
            RelevanceLevel.MEDIUM.value: 2,
            RelevanceLevel.LOW.value: 1,
        }
        return scores.get(relevance_level, 2)

    def _update_metrics(self, clauses: List[Dict[str, Any]], processing_time: float):
        """Update performance metrics"""
        try:
            self.metrics["total_extractions"] += 1
            self.metrics["successful_extractions"] += 1 if clauses else 0

            # Update clause type counts
            for clause in clauses:
                clause_type = clause.get("clause_type", "UNKNOWN")
                if clause_type in self.metrics["clause_type_counts"]:
                    self.metrics["clause_type_counts"][clause_type] += 1

            # Update average relevance score
            if clauses:
                relevance_scores = [c.get("relevance_score", 0.5) for c in clauses]
                avg_relevance = sum(relevance_scores) / len(relevance_scores)

                current_avg = self.metrics["average_relevance_score"]
                total_extractions = self.metrics["total_extractions"]
                new_avg = (
                    (current_avg * (total_extractions - 1)) + avg_relevance
                ) / total_extractions
                self.metrics["average_relevance_score"] = new_avg

            # Update average processing time
            current_avg_time = self.metrics["average_processing_time"]
            total_extractions = self.metrics["total_extractions"]
            new_avg_time = (
                (current_avg_time * (total_extractions - 1)) + processing_time
            ) / total_extractions
            self.metrics["average_processing_time"] = new_avg_time

        except Exception:
            pass

    def fine_tune_extractor(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        num_epochs: int = 3,
        learning_rate: float = 1e-4,
        batch_size: int = 8,
    ) -> Dict[str, Any]:
        """
        Fine-tune T5 model for clause extraction

        Args:
            training_data: Training data with documents and labeled clauses
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size

        Returns:
            Fine-tuning results
        """
        try:
            logger.info(f"Starting T5 fine-tuning with {len(training_data)} samples")

            # Prepare training data (mock implementation)
            train_loader = self._prepare_training_data(training_data, batch_size)
            val_loader = None
            if validation_data:
                val_loader = self._prepare_training_data(validation_data, batch_size)

            # Setup optimizer
            optimizer = torch.optim.AdamW(self.t5_model.parameters(), lr=learning_rate)

            # Training loop
            training_history = {
                "train_losses": [],
                "val_losses": [],
                "epochs_completed": 0,
            }

            for epoch in range(num_epochs):
                # Training phase
                train_loss = self._train_t5_epoch(train_loader, optimizer)
                training_history["train_losses"].append(train_loss)

                # Validation phase
                if val_loader:
                    val_loss = self._validate_t5_epoch(val_loader)
                    training_history["val_losses"].append(val_loss)

                training_history["epochs_completed"] = epoch + 1
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"
                )

            return {
                "status": "success",
                "training_history": training_history,
                "final_train_loss": training_history["train_losses"][-1],
                "final_val_loss": training_history["val_losses"][-1]
                if training_history["val_losses"]
                else None,
            }

        except Exception as e:
            logger.error(f"T5 fine-tuning failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _prepare_training_data(self, data: List[Dict[str, Any]], batch_size: int):
        """Prepare training data loader"""
        # Mock implementation
        return MockT5DataLoader(data, batch_size)

    def _train_t5_epoch(self, train_loader, optimizer) -> float:
        """Train T5 for one epoch"""
        self.t5_model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            # Mock training step
            loss = torch.tensor(1.5 + torch.randn(1).item() * 0.2)  # Simulated loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def _validate_t5_epoch(self, val_loader) -> float:
        """Validate T5 for one epoch"""
        self.t5_model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # Mock validation step
                loss = torch.tensor(1.3 + torch.randn(1).item() * 0.15)
                total_loss += loss.item()

        return total_loss / num_batches

    def get_extractor_info(self) -> Dict[str, Any]:
        """Get comprehensive extractor information"""
        return {
            "model_variant": self.model_variant,
            "extraction_mode": self.extraction_mode.value,
            "configuration": {
                "max_input_length": self.max_input_length,
                "max_output_length": self.max_output_length,
                "relevance_threshold": self.relevance_threshold,
            },
            "supported_clause_types": [clause.value for clause in ClauseType],
            "performance_metrics": self.metrics,
            "model_status": {
                "is_initialized": self.is_initialized,
                "device": str(self.device),
                "parameters": self._count_parameters(),
            },
        }

    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        try:
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )

            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
            }
        except Exception:
            return {"total_parameters": 0, "trainable_parameters": 0}


# Mock classes for demonstration
class MockT5Tokenizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.pad_token_id = 0

    def encode(
        self, text: str, max_length: int = 512, truncation: bool = True
    ) -> List[int]:
        # Mock tokenization
        words = text.split()[: max_length // 4]  # Rough approximation
        return list(range(1, len(words) + 1))

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        # Mock decoding
        return f"Generated clause text based on input (length: {len(token_ids)})"


class MockT5Model(nn.Module):
    def __init__(self, model_name: str, max_input_length: int, max_output_length: int):
        super().__init__()
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        # Mock model components
        self.encoder = nn.Linear(512, 512)
        self.decoder = nn.Linear(512, 32000)  # Vocab size

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 256,
        temperature: float = 0.7,
        do_sample: bool = False,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        # Mock generation
        batch_size = input_ids.shape[0]
        return torch.randint(1, 1000, (batch_size, min(max_length, 50)))


class MockT5DataLoader:
    def __init__(self, data: List[Dict[str, Any]], batch_size: int):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size + (
            1 if len(self.data) % self.batch_size else 0
        )

    def __iter__(self):
        for i in range(len(self)):
            yield {"batch": i}
