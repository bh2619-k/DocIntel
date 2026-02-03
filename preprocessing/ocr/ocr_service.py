"""
Real-Time Document Intelligence & Compliance Engine
OCR Service Integration
Version: 1.0
Date: 2026-02-03

This module provides OCR services with AWS Textract as primary and Tesseract as fallback,
including confidence scoring, result validation, and adaptive quality optimization.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OCRProvider(Enum):
    """OCR service providers"""

    TEXTRACT = "TEXTRACT"  # AWS Textract (primary)
    TESSERACT = "TESSERACT"  # Tesseract OCR (fallback)
    AZURE_OCR = "AZURE_OCR"  # Azure Computer Vision OCR
    GOOGLE_OCR = "GOOGLE_OCR"  # Google Cloud Vision OCR


class OCRMode(Enum):
    """OCR processing modes"""

    TEXT_DETECTION = "TEXT_DETECTION"  # Basic text extraction
    LAYOUT_ANALYSIS = "LAYOUT_ANALYSIS"  # Layout-aware extraction
    TABLE_EXTRACTION = "TABLE_EXTRACTION"  # Table structure extraction
    FORM_EXTRACTION = "FORM_EXTRACTION"  # Form field extraction
    HANDWRITING = "HANDWRITING"  # Handwritten text extraction


class OCRQuality(Enum):
    """OCR quality levels"""

    FAST = "FAST"  # Quick processing, lower quality
    STANDARD = "STANDARD"  # Balanced speed and quality
    HIGH = "HIGH"  # High quality, slower processing
    PREMIUM = "PREMIUM"  # Maximum quality, slowest


class OCRServiceManager:
    """
    Manages OCR services with intelligent provider selection and fallback handling
    """

    def __init__(
        self,
        primary_provider: OCRProvider = OCRProvider.TEXTRACT,
        enable_fallback: bool = True,
        confidence_threshold: float = 0.7,
    ):
        """
        Initialize OCR service manager

        Args:
            primary_provider: Primary OCR provider to use
            enable_fallback: Whether to enable fallback providers
            confidence_threshold: Minimum confidence score for accepting results
        """
        self.primary_provider = primary_provider
        self.enable_fallback = enable_fallback
        self.confidence_threshold = confidence_threshold

        # Provider configurations
        self.provider_configs = {
            OCRProvider.TEXTRACT: {
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "supported_formats": ["pdf", "png", "jpg", "jpeg", "tiff"],
                "max_pages": 3000,
                "cost_per_page": 0.0015,
                "average_quality": 0.92,
            },
            OCRProvider.TESSERACT: {
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "supported_formats": ["png", "jpg", "jpeg", "tiff", "bmp"],
                "max_pages": 1000,
                "cost_per_page": 0.0,
                "average_quality": 0.75,
            },
            OCRProvider.AZURE_OCR: {
                "max_file_size": 50 * 1024 * 1024,  # 50MB
                "supported_formats": ["pdf", "png", "jpg", "jpeg", "tiff"],
                "max_pages": 2000,
                "cost_per_page": 0.001,
                "average_quality": 0.88,
            },
        }

        # Initialize providers
        self.providers = {}
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize available OCR providers"""
        try:
            # Initialize Textract provider
            self.providers[OCRProvider.TEXTRACT] = TextractProvider()

            # Initialize Tesseract provider
            self.providers[OCRProvider.TESSERACT] = TesseractProvider()

            # Initialize Azure OCR provider (if configured)
            # self.providers[OCRProvider.AZURE_OCR] = AzureOCRProvider()

            logger.info(f"Initialized {len(self.providers)} OCR providers")

        except Exception as e:
            logger.error(f"Failed to initialize OCR providers: {e}")

    async def extract_text(
        self,
        file_content: bytes,
        filename: str,
        ocr_mode: OCRMode = OCRMode.TEXT_DETECTION,
        quality: OCRQuality = OCRQuality.STANDARD,
        language_hints: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract text using optimal OCR provider with fallback

        Args:
            file_content: Document binary content
            filename: Original filename for format detection
            ocr_mode: OCR processing mode
            quality: Quality level for processing
            language_hints: Language codes to optimize OCR

        Returns:
            OCR extraction results with text, confidence, and metadata
        """
        try:
            extraction_start = datetime.utcnow()

            # Determine file format
            file_format = self._get_file_format(filename)
            file_size = len(file_content)

            # Select optimal provider
            selected_provider = await self._select_optimal_provider(
                file_format, file_size, ocr_mode, quality
            )

            # Prepare extraction request
            extraction_request = {
                "file_content": file_content,
                "filename": filename,
                "file_format": file_format,
                "file_size": file_size,
                "ocr_mode": ocr_mode,
                "quality": quality,
                "language_hints": language_hints or ["en"],
                "provider": selected_provider,
            }

            # Attempt extraction with primary provider
            extraction_result = await self._extract_with_provider(
                selected_provider, extraction_request
            )

            # Check if fallback is needed
            if (
                self.enable_fallback
                and extraction_result.get("confidence_score", 0)
                < self.confidence_threshold
            ):
                fallback_provider = await self._get_fallback_provider(
                    selected_provider, file_format, file_size
                )

                if fallback_provider:
                    logger.info(f"Using fallback provider: {fallback_provider.value}")
                    extraction_request["provider"] = fallback_provider

                    fallback_result = await self._extract_with_provider(
                        fallback_provider, extraction_request
                    )

                    # Use better result
                    if fallback_result.get(
                        "confidence_score", 0
                    ) > extraction_result.get("confidence_score", 0):
                        extraction_result = fallback_result
                        extraction_result["used_fallback"] = True
                        extraction_result["fallback_provider"] = fallback_provider.value

            # Calculate processing time
            extraction_end = datetime.utcnow()
            extraction_result["processing_time_ms"] = int(
                (extraction_end - extraction_start).total_seconds() * 1000
            )

            # Validate and enhance results
            extraction_result = await self._validate_and_enhance_results(
                extraction_result
            )

            logger.info(
                f"OCR completed: provider={extraction_result.get('provider')}, "
                f"confidence={extraction_result.get('confidence_score', 0):.2f}, "
                f"words={extraction_result.get('word_count', 0)}"
            )

            return extraction_result

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": None,
                "confidence_score": 0.0,
                "text": "",
                "word_count": 0,
                "processing_time_ms": 0,
            }

    async def _select_optimal_provider(
        self, file_format: str, file_size: int, ocr_mode: OCRMode, quality: OCRQuality
    ) -> OCRProvider:
        """Select optimal OCR provider based on document characteristics"""
        try:
            # Score each available provider
            provider_scores = {}

            for provider, config in self.provider_configs.items():
                if provider not in self.providers:
                    continue

                score = 0

                # Format support
                if file_format in config["supported_formats"]:
                    score += 30

                # File size compatibility
                if file_size <= config["max_file_size"]:
                    score += 20
                else:
                    continue  # Skip if file too large

                # Quality preference
                if quality == OCRQuality.PREMIUM and provider == OCRProvider.TEXTRACT:
                    score += 25
                elif quality == OCRQuality.FAST and provider == OCRProvider.TESSERACT:
                    score += 25

                # Mode-specific optimization
                if (
                    ocr_mode == OCRMode.TABLE_EXTRACTION
                    and provider == OCRProvider.TEXTRACT
                ):
                    score += 20
                elif (
                    ocr_mode == OCRMode.HANDWRITING
                    and provider == OCRProvider.AZURE_OCR
                ):
                    score += 20

                # Base quality score
                score += int(config["average_quality"] * 20)

                # Cost consideration (prefer lower cost for large documents)
                if file_size > 1024 * 1024:  # >1MB
                    score += int((1 - config["cost_per_page"]) * 10)

                provider_scores[provider] = score

            # Select highest scoring provider
            if provider_scores:
                best_provider = max(provider_scores, key=provider_scores.get)
                logger.debug(
                    f"Selected provider: {best_provider} (score: {provider_scores[best_provider]})"
                )
                return best_provider
            else:
                return self.primary_provider

        except Exception as e:
            logger.warning(f"Provider selection failed: {e}")
            return self.primary_provider

    async def _extract_with_provider(
        self, provider: OCRProvider, extraction_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract text using specified provider"""
        try:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not available")

            provider_instance = self.providers[provider]
            result = await provider_instance.extract_text(extraction_request)

            # Add provider metadata
            result["provider"] = provider.value
            result["provider_config"] = self.provider_configs[provider]

            return result

        except Exception as e:
            logger.error(f"Extraction failed with provider {provider}: {e}")
            return {
                "success": False,
                "error": str(e),
                "provider": provider.value,
                "confidence_score": 0.0,
                "text": "",
                "word_count": 0,
            }

    async def _get_fallback_provider(
        self, primary: OCRProvider, file_format: str, file_size: int
    ) -> Optional[OCRProvider]:
        """Get appropriate fallback provider"""
        fallback_options = [
            OCRProvider.TESSERACT,
            OCRProvider.AZURE_OCR,
            OCRProvider.TEXTRACT,
        ]

        # Remove primary provider from fallback options
        if primary in fallback_options:
            fallback_options.remove(primary)

        # Find compatible fallback
        for provider in fallback_options:
            if provider in self.providers:
                config = self.provider_configs[provider]
                if (
                    file_format in config["supported_formats"]
                    and file_size <= config["max_file_size"]
                ):
                    return provider

        return None

    async def _validate_and_enhance_results(
        self, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and enhance OCR results"""
        try:
            if not result.get("success", False):
                return result

            extracted_text = result.get("text", "")

            # Basic validation
            result["word_count"] = len(extracted_text.split())
            result["character_count"] = len(extracted_text)
            result["line_count"] = len(extracted_text.splitlines())

            # Confidence validation
            if result.get("confidence_score", 0) < 0.3:
                result["quality_warning"] = "Low confidence OCR results"

            # Text quality analysis
            result["text_analysis"] = await self._analyze_text_quality(extracted_text)

            # Detect potential OCR errors
            result["potential_errors"] = await self._detect_ocr_errors(extracted_text)

            return result

        except Exception as e:
            logger.warning(f"Result validation failed: {e}")
            return result

    async def _analyze_text_quality(self, text: str) -> Dict[str, Any]:
        """Analyze quality of extracted text"""
        try:
            analysis = {
                "readability_score": 0.0,
                "formatting_preserved": False,
                "special_characters_ratio": 0.0,
                "suspected_errors": 0,
            }

            if not text:
                return analysis

            # Count special characters that might indicate OCR errors
            special_chars = sum(
                1
                for c in text
                if not (c.isalnum() or c.isspace() or c in ".,!?;:()[]{}\"'-")
            )
            analysis["special_characters_ratio"] = (
                special_chars / len(text) if text else 0
            )

            # Check for formatting preservation
            analysis["formatting_preserved"] = "\n" in text and (
                "  " in text or "\t" in text
            )

            # Simple readability estimate
            words = text.split()
            avg_word_length = (
                sum(len(word) for word in words) / len(words) if words else 0
            )
            analysis["readability_score"] = min(
                1.0, max(0.0, (10 - avg_word_length) / 10)
            )

            return analysis

        except Exception as e:
            logger.warning(f"Text quality analysis failed: {e}")
            return {"error": str(e)}

    async def _detect_ocr_errors(self, text: str) -> List[str]:
        """Detect potential OCR errors in text"""
        errors = []

        try:
            # Common OCR error patterns
            error_patterns = [
                (r"[0-9]+[a-zA-Z]+[0-9]+", "mixed_alphanumeric_sequences"),
                (r"\b[A-Z]{1}[a-z]*[A-Z]+[a-z]*\b", "irregular_capitalization"),
                (r'[^\w\s.,!?;:()\[\]{}"\'-]', "unexpected_special_characters"),
                (r"\b\w{1,2}\b.*\b\w{1,2}\b", "excessive_short_words"),
                (r"[il1|]{2,}", "character_confusion_il1"),
            ]

            import re

            for pattern, error_type in error_patterns:
                matches = re.findall(pattern, text)
                if len(matches) > len(text.split()) * 0.1:  # More than 10% of words
                    errors.append(error_type)

            return errors

        except Exception:
            return []

    def _get_file_format(self, filename: str) -> str:
        """Extract file format from filename"""
        return filename.split(".")[-1].lower() if "." in filename else "unknown"


class TextractProvider:
    """AWS Textract OCR provider implementation"""

    def __init__(self):
        self.service_name = "textract"
        self.max_retries = 3

    async def extract_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text using AWS Textract"""
        try:
            # Mock Textract implementation
            # In real implementation, use boto3 Textract client

            # file_content = request["file_content"]
            # ocr_mode = request["ocr_mode"]
            quality = request["quality"]

            # Simulate processing delay
            await asyncio.sleep(0.5)

            # Mock extraction results
            extracted_text = (
                "Sample extracted text from AWS Textract OCR service. " * 10
            )

            result = {
                "success": True,
                "text": extracted_text,
                "confidence_score": 0.92,
                "word_count": len(extracted_text.split()),
                "blocks": [],  # Textract block structure
                "tables": [],  # Detected tables
                "forms": [],  # Detected form fields
                "lines": [],  # Line-level results
                "service_metadata": {
                    "job_id": "textract-job-12345",
                    "model_version": "1.0",
                    "pages_processed": 1,
                },
            }

            # Adjust confidence based on quality setting
            quality_multipliers = {
                OCRQuality.FAST: 0.85,
                OCRQuality.STANDARD: 1.0,
                OCRQuality.HIGH: 1.05,
                OCRQuality.PREMIUM: 1.1,
            }

            result["confidence_score"] *= quality_multipliers.get(quality, 1.0)
            result["confidence_score"] = min(0.99, result["confidence_score"])

            return result

        except Exception as e:
            logger.error(f"Textract extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence_score": 0.0,
                "text": "",
                "word_count": 0,
            }


class TesseractProvider:
    """Tesseract OCR provider implementation"""

    def __init__(self):
        self.service_name = "tesseract"
        self.default_config = "--oem 3 --psm 6"

    async def extract_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text using Tesseract OCR"""
        try:
            # Mock Tesseract implementation
            # In real implementation, use pytesseract

            # file_content = request["file_content"]
            language_hints = request.get("language_hints", ["en"])
            quality = request["quality"]

            # Simulate processing delay
            await asyncio.sleep(1.0)

            # Mock extraction results
            extracted_text = "Sample extracted text from Tesseract OCR engine. " * 8

            # Simulate language-specific results
            lang_code = "+".join(language_hints[:3])  # Tesseract format

            result = {
                "success": True,
                "text": extracted_text,
                "confidence_score": 0.78,
                "word_count": len(extracted_text.split()),
                "language_used": lang_code,
                "service_metadata": {
                    "tesseract_version": "5.3.0",
                    "config_used": self.default_config,
                    "preprocessing_applied": ["deskew", "noise_removal"],
                },
            }

            # Adjust confidence based on quality
            quality_adjustments = {
                OCRQuality.FAST: -0.1,
                OCRQuality.STANDARD: 0.0,
                OCRQuality.HIGH: +0.05,
                OCRQuality.PREMIUM: +0.1,
            }

            result["confidence_score"] += quality_adjustments.get(quality, 0.0)
            result["confidence_score"] = max(0.1, min(0.95, result["confidence_score"]))

            return result

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "confidence_score": 0.0,
                "text": "",
                "word_count": 0,
            }


class OCRResultProcessor:
    """
    Post-processes OCR results for quality improvement
    """

    def __init__(self):
        self.correction_enabled = True
        self.language_detection_enabled = True

    async def process_ocr_results(
        self, ocr_result: Dict[str, Any], enable_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Process and enhance OCR results

        Args:
            ocr_result: Raw OCR extraction results
            enable_correction: Whether to apply text corrections

        Returns:
            Enhanced OCR results
        """
        try:
            if not ocr_result.get("success", False):
                return ocr_result

            processed_result = ocr_result.copy()
            extracted_text = ocr_result.get("text", "")

            if not extracted_text:
                return processed_result

            # Apply text corrections if enabled
            if enable_correction and self.correction_enabled:
                corrected_text = await self._apply_text_corrections(extracted_text)
                processed_result["corrected_text"] = corrected_text
                processed_result["correction_applied"] = True

                # Update confidence if corrections were significant
                correction_ratio = self._calculate_correction_ratio(
                    extracted_text, corrected_text
                )
                if correction_ratio > 0.1:  # More than 10% corrections
                    processed_result["confidence_score"] *= 1 - correction_ratio * 0.5

            # Detect and validate language
            if self.language_detection_enabled:
                language_info = await self._detect_and_validate_language(extracted_text)
                processed_result["detected_language"] = language_info

            # Extract structured information
            structured_data = await self._extract_structured_data(extracted_text)
            processed_result["structured_data"] = structured_data

            # Generate quality metrics
            quality_metrics = await self._generate_quality_metrics(processed_result)
            processed_result["quality_metrics"] = quality_metrics

            return processed_result

        except Exception as e:
            logger.error(f"OCR result processing failed: {e}")
            ocr_result["processing_error"] = str(e)
            return ocr_result

    async def _apply_text_corrections(self, text: str) -> str:
        """Apply text corrections to OCR output"""
        try:
            corrected_text = text

            # Common OCR error corrections
            corrections = {
                # Character confusions
                r"\bl\b": "I",  # Lowercase l to uppercase I
                r"\b0\b": "O",  # Zero to letter O in context
                r"\brn\b": "m",  # rn to m
                r"\bvv\b": "w",  # vv to w
                # Word-level corrections
                r"\bteh\b": "the",
                r"\badn\b": "and",
                r"\bfrom\b": "from",
                r"\bwith\b": "with",
            }

            import re

            for pattern, replacement in corrections.items():
                corrected_text = re.sub(
                    pattern, replacement, corrected_text, flags=re.IGNORECASE
                )

            # Remove excessive whitespace
            corrected_text = re.sub(r"\s+", " ", corrected_text)
            corrected_text = corrected_text.strip()

            return corrected_text

        except Exception as e:
            logger.warning(f"Text correction failed: {e}")
            return text

    async def _detect_and_validate_language(self, text: str) -> Dict[str, Any]:
        """Detect and validate document language"""
        try:
            # Simple language detection (would use langdetect in real implementation)
            language_info = {
                "detected_language": "en",
                "confidence": 0.85,
                "alternative_languages": ["es", "fr"],
                "is_multilingual": False,
            }

            # Check for multilingual indicators
            if len(text) > 100:
                # Simple check for mixed language indicators
                english_words = [
                    "the",
                    "and",
                    "or",
                    "for",
                    "with",
                    "this",
                    "that",
                    "from",
                ]
                spanish_words = ["el", "y", "o", "para", "con", "este", "de", "la"]

                text_lower = text.lower()
                en_count = sum(1 for word in english_words if word in text_lower)
                es_count = sum(1 for word in spanish_words if word in text_lower)

                if en_count > 0 and es_count > 0:
                    language_info["is_multilingual"] = True
                    language_info["detected_language"] = "mixed"

            return language_info

        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return {"error": str(e)}

    async def _extract_structured_data(self, text: str) -> Dict[str, Any]:
        """Extract structured information from text"""
        try:
            structured_data = {
                "emails": [],
                "phone_numbers": [],
                "dates": [],
                "monetary_amounts": [],
                "addresses": [],
            }

            import re

            # Email extraction
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            structured_data["emails"] = re.findall(email_pattern, text)

            # Phone number extraction
            phone_pattern = r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"
            structured_data["phone_numbers"] = re.findall(phone_pattern, text)

            # Date extraction (simple patterns)
            date_patterns = [
                r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
                r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
                r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b",
            ]

            for pattern in date_patterns:
                structured_data["dates"].extend(re.findall(pattern, text))

            # Monetary amount extraction
            money_pattern = r"\$\s*\d{1,3}(?:,\d{3})*(?:\.\d{2})?"
            structured_data["monetary_amounts"] = re.findall(money_pattern, text)

            return structured_data

        except Exception as e:
            logger.warning(f"Structured data extraction failed: {e}")
            return {}

    async def _generate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate quality metrics for OCR results"""
        try:
            text = result.get("text", "")
            confidence = result.get("confidence_score", 0.0)

            metrics = {
                "overall_quality": "unknown",
                "text_density": 0.0,
                "structure_preservation": "unknown",
                "error_indicators": 0,
                "recommendation": "review_recommended",
            }

            if text:
                # Text density (characters per "page" estimate)
                estimated_pages = max(1, len(text) // 3000)
                metrics["text_density"] = len(text) / estimated_pages

                # Structure preservation check
                if "\n" in text and any(char in text for char in ["\t", "  "]):
                    metrics["structure_preservation"] = "good"
                elif "\n" in text:
                    metrics["structure_preservation"] = "partial"
                else:
                    metrics["structure_preservation"] = "poor"

                # Overall quality assessment
                if confidence >= 0.9 and metrics["text_density"] > 1000:
                    metrics["overall_quality"] = "excellent"
                    metrics["recommendation"] = "ready_for_processing"
                elif confidence >= 0.7:
                    metrics["overall_quality"] = "good"
                    metrics["recommendation"] = "minor_review_suggested"
                elif confidence >= 0.5:
                    metrics["overall_quality"] = "fair"
                    metrics["recommendation"] = "review_recommended"
                else:
                    metrics["overall_quality"] = "poor"
                    metrics["recommendation"] = "manual_review_required"

            return metrics

        except Exception as e:
            logger.warning(f"Quality metrics generation failed: {e}")
            return {"error": str(e)}

    def _calculate_correction_ratio(self, original: str, corrected: str) -> float:
        """Calculate ratio of corrections made to text"""
        try:
            if not original or not corrected:
                return 0.0

            # Simple character-based difference ratio
            import difflib

            matcher = difflib.SequenceMatcher(None, original, corrected)
            similarity = matcher.ratio()
            return 1.0 - similarity

        except Exception:
            return 0.0
