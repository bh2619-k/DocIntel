"""
Real-Time Document Intelligence & Compliance Engine
Document Type Detection & Preprocessing
Version: 1.0
Date: 2026-02-03

This module handles hybrid document detection, splitting multi-document files,
and determining optimal processing strategies for different document types.
"""

import logging
import hashlib
from typing import Dict, List, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classifications"""

    DIGITAL_PDF = "DIGITAL_PDF"  # Born-digital PDF with selectable text
    SCANNED_PDF = "SCANNED_PDF"  # Scanned PDF requiring OCR
    HYBRID_PDF = "HYBRID_PDF"  # Mixed digital and scanned pages
    IMAGE_DOCUMENT = "IMAGE_DOCUMENT"  # Image files (PNG, JPEG, TIFF)
    OFFICE_DOCUMENT = "OFFICE_DOCUMENT"  # Word, Excel, PowerPoint
    TEXT_DOCUMENT = "TEXT_DOCUMENT"  # Plain text files
    MULTI_DOCUMENT = "MULTI_DOCUMENT"  # Multiple documents in one file
    UNKNOWN = "UNKNOWN"  # Unable to classify


class ProcessingStrategy(Enum):
    """Processing strategy based on document type"""

    TEXT_EXTRACTION = "TEXT_EXTRACTION"  # Extract text directly
    OCR_PRIMARY = "OCR_PRIMARY"  # Use OCR as primary method
    OCR_FALLBACK = "OCR_FALLBACK"  # Try text extraction, fallback to OCR
    LAYOUT_ANALYSIS = "LAYOUT_ANALYSIS"  # Complex layout analysis required
    SPLITTING_REQUIRED = "SPLITTING_REQUIRED"  # Document splitting needed


class DocumentDetector:
    """
    Hybrid document detector for determining document types and processing strategies
    """

    def __init__(self, use_advanced_detection: bool = True):
        """
        Initialize document detector

        Args:
            use_advanced_detection: Whether to use advanced ML-based detection
        """
        self.use_advanced_detection = use_advanced_detection

        # Text extraction confidence thresholds
        self.digital_text_threshold = (
            0.7  # Minimum text coverage for digital classification
        )
        self.hybrid_threshold = 0.3  # Minimum text for hybrid classification

        # Page analysis thresholds
        self.max_pages_for_analysis = 20  # Maximum pages to analyze for type detection
        self.splitting_confidence_threshold = (
            0.8  # Confidence threshold for document splitting
        )

    async def detect_document_type(
        self, file_content: bytes, filename: str, mime_type: str
    ) -> Dict[str, Any]:
        """
        Detect document type and determine processing strategy

        Args:
            file_content: Binary document content
            filename: Original filename
            mime_type: MIME content type

        Returns:
            Detection results with type, strategy, and metadata
        """
        try:
            detection_start = datetime.utcnow()

            # Basic file analysis
            file_size = len(file_content)
            file_hash = hashlib.sha256(file_content).hexdigest()

            # Initialize detection results
            detection_result = {
                "document_type": DocumentType.UNKNOWN.value,
                "processing_strategy": ProcessingStrategy.OCR_PRIMARY.value,
                "confidence_score": 0.0,
                "page_count": 0,
                "has_text_layer": False,
                "text_coverage_percentage": 0.0,
                "requires_splitting": False,
                "language_hints": [],
                "layout_complexity": "simple",
                "file_metadata": {
                    "filename": filename,
                    "mime_type": mime_type,
                    "file_size": file_size,
                    "file_hash": file_hash,
                },
                "processing_recommendations": [],
                "detection_metadata": {
                    "detection_time_ms": 0,
                    "analysis_depth": "basic",
                },
            }

            # Route to appropriate detector based on MIME type
            if mime_type == "application/pdf":
                detection_result = await self._analyze_pdf_document(
                    file_content, detection_result
                )
            elif mime_type.startswith("image/"):
                detection_result = await self._analyze_image_document(
                    file_content, detection_result
                )
            elif (
                "word" in mime_type
                or mime_type
                == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ):
                detection_result = await self._analyze_office_document(
                    file_content, detection_result
                )
            elif mime_type == "text/plain":
                detection_result = await self._analyze_text_document(
                    file_content, detection_result
                )
            else:
                detection_result["document_type"] = DocumentType.UNKNOWN.value
                detection_result["processing_strategy"] = (
                    ProcessingStrategy.OCR_PRIMARY.value
                )

            # Calculate detection time
            detection_end = datetime.utcnow()
            detection_result["detection_metadata"]["detection_time_ms"] = int(
                (detection_end - detection_start).total_seconds() * 1000
            )

            # Add processing recommendations
            detection_result["processing_recommendations"] = (
                self._generate_processing_recommendations(detection_result)
            )

            logger.info(
                f"Document type detected: {detection_result['document_type']} "
                f"(confidence: {detection_result['confidence_score']:.2f})"
            )

            return detection_result

        except Exception as e:
            logger.error(f"Document type detection failed: {e}")
            detection_result["document_type"] = DocumentType.UNKNOWN.value
            detection_result["error"] = str(e)
            return detection_result

    async def _analyze_pdf_document(
        self, file_content: bytes, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze PDF document for type and characteristics"""
        try:
            # Use PyPDF2 or similar for PDF analysis
            text_analysis = await self._extract_pdf_text_sample(file_content)

            result["page_count"] = text_analysis["page_count"]
            result["has_text_layer"] = text_analysis["has_text"]
            result["text_coverage_percentage"] = text_analysis["text_coverage"]

            # Determine document type based on text coverage
            if text_analysis["text_coverage"] >= self.digital_text_threshold:
                result["document_type"] = DocumentType.DIGITAL_PDF.value
                result["processing_strategy"] = ProcessingStrategy.TEXT_EXTRACTION.value
                result["confidence_score"] = min(0.95, text_analysis["text_coverage"])

            elif text_analysis["text_coverage"] >= self.hybrid_threshold:
                result["document_type"] = DocumentType.HYBRID_PDF.value
                result["processing_strategy"] = ProcessingStrategy.OCR_FALLBACK.value
                result["confidence_score"] = 0.7

            else:
                result["document_type"] = DocumentType.SCANNED_PDF.value
                result["processing_strategy"] = ProcessingStrategy.OCR_PRIMARY.value
                result["confidence_score"] = 0.8

            # Check for multi-document indicators
            if await self._check_for_document_splitting(file_content, text_analysis):
                result["requires_splitting"] = True
                result["document_type"] = DocumentType.MULTI_DOCUMENT.value
                result["processing_strategy"] = (
                    ProcessingStrategy.SPLITTING_REQUIRED.value
                )

            # Analyze layout complexity
            result["layout_complexity"] = await self._analyze_layout_complexity(
                file_content
            )

            # Detect language hints
            result["language_hints"] = await self._detect_language_hints(
                text_analysis.get("sample_text", "")
            )

        except Exception as e:
            logger.warning(f"PDF analysis failed: {e}")
            result["document_type"] = DocumentType.SCANNED_PDF.value
            result["processing_strategy"] = ProcessingStrategy.OCR_PRIMARY.value
            result["confidence_score"] = 0.5

        return result

    async def _analyze_image_document(
        self, file_content: bytes, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze image document"""
        try:
            # Analyze image properties
            image_analysis = await self._analyze_image_properties(file_content)

            result["document_type"] = DocumentType.IMAGE_DOCUMENT.value
            result["processing_strategy"] = ProcessingStrategy.OCR_PRIMARY.value
            result["confidence_score"] = 0.9
            result["page_count"] = 1

            # Update metadata with image properties
            result["file_metadata"].update(
                {
                    "image_width": image_analysis.get("width", 0),
                    "image_height": image_analysis.get("height", 0),
                    "image_format": image_analysis.get("format", "unknown"),
                    "color_mode": image_analysis.get("color_mode", "unknown"),
                }
            )

            # Determine if layout analysis is needed based on image characteristics
            if image_analysis.get("complexity", "simple") == "complex":
                result["processing_strategy"] = ProcessingStrategy.LAYOUT_ANALYSIS.value
                result["layout_complexity"] = "complex"

        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            result["confidence_score"] = 0.5

        return result

    async def _analyze_office_document(
        self, file_content: bytes, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze Microsoft Office document"""
        try:
            result["document_type"] = DocumentType.OFFICE_DOCUMENT.value
            result["processing_strategy"] = ProcessingStrategy.TEXT_EXTRACTION.value
            result["confidence_score"] = 0.9
            result["has_text_layer"] = True
            result["text_coverage_percentage"] = (
                95.0  # Assume high text coverage for office docs
            )

            # Office documents typically don't need OCR
            result["processing_recommendations"].append("convert_to_pdf_first")

        except Exception as e:
            logger.warning(f"Office document analysis failed: {e}")
            result["confidence_score"] = 0.5

        return result

    async def _analyze_text_document(
        self, file_content: bytes, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze plain text document"""
        try:
            text_content = file_content.decode("utf-8", errors="ignore")

            result["document_type"] = DocumentType.TEXT_DOCUMENT.value
            result["processing_strategy"] = ProcessingStrategy.TEXT_EXTRACTION.value
            result["confidence_score"] = 1.0
            result["page_count"] = max(1, len(text_content) // 3000)  # Estimate pages
            result["has_text_layer"] = True
            result["text_coverage_percentage"] = 100.0

            # Detect language from text content
            result["language_hints"] = await self._detect_language_hints(
                text_content[:1000]
            )

        except Exception as e:
            logger.warning(f"Text document analysis failed: {e}")
            result["confidence_score"] = 0.5

        return result

    async def _extract_pdf_text_sample(self, file_content: bytes) -> Dict[str, Any]:
        """Extract text sample from PDF to analyze content"""
        try:
            # This would use PyPDF2, pdfplumber, or similar
            # For now, return a mock analysis

            # Simulate PDF text extraction
            sample_analysis = {
                "page_count": 1,
                "has_text": True,
                "text_coverage": 0.8,
                "sample_text": "Sample extracted text from PDF document",
                "fonts_detected": ["Arial", "Times"],
                "images_count": 0,
            }

            # Basic heuristics for text coverage estimation
            # In real implementation, this would analyze actual PDF structure
            if len(file_content) < 50000:  # Small file, likely text-based
                sample_analysis["text_coverage"] = 0.9
            elif b"/Font" in file_content and b"/Text" in file_content:
                sample_analysis["text_coverage"] = 0.8
            elif b"/Image" in file_content and b"/Font" not in file_content:
                sample_analysis["text_coverage"] = 0.1
            else:
                sample_analysis["text_coverage"] = 0.5

            return sample_analysis

        except Exception as e:
            logger.warning(f"PDF text extraction failed: {e}")
            return {
                "page_count": 1,
                "has_text": False,
                "text_coverage": 0.0,
                "sample_text": "",
                "fonts_detected": [],
                "images_count": 0,
            }

    async def _check_for_document_splitting(
        self, file_content: bytes, text_analysis: Dict[str, Any]
    ) -> bool:
        """Check if document contains multiple separate documents"""
        try:
            sample_text = text_analysis.get("sample_text", "")

            # Look for document boundary indicators
            boundary_indicators = [
                "page 1 of",
                "document begins",
                "end of document",
                "next document",
                "attachment:",
                "exhibit a",
                "exhibit b",
                "schedule",
            ]

            boundary_count = sum(
                1
                for indicator in boundary_indicators
                if indicator.lower() in sample_text.lower()
            )

            # If multiple boundaries found, likely multi-document
            return boundary_count >= 2

        except Exception:
            return False

    async def _analyze_layout_complexity(self, file_content: bytes) -> str:
        """Analyze layout complexity of document"""
        try:
            # Simple heuristics for layout complexity
            # In real implementation, this would use computer vision

            if len(file_content) > 1000000:  # Large file
                return "complex"
            elif b"/Table" in file_content or b"/Form" in file_content:
                return "complex"
            else:
                return "simple"

        except Exception:
            return "simple"

    async def _analyze_image_properties(self, file_content: bytes) -> Dict[str, Any]:
        """Analyze image properties for processing optimization"""
        try:
            # This would use PIL/Pillow for actual image analysis
            # For now, return mock properties

            properties = {
                "width": 2100,
                "height": 2970,
                "format": "PNG",
                "color_mode": "RGB",
                "dpi": 300,
                "complexity": "simple",
            }

            # Estimate complexity based on file size
            if len(file_content) > 5000000:  # Large image
                properties["complexity"] = "complex"
                properties["dpi"] = 600

            return properties

        except Exception as e:
            logger.warning(f"Image property analysis failed: {e}")
            return {
                "width": 0,
                "height": 0,
                "format": "unknown",
                "color_mode": "unknown",
                "dpi": 300,
                "complexity": "simple",
            }

    async def _detect_language_hints(self, text_sample: str) -> List[str]:
        """Detect language hints from text sample"""
        try:
            if not text_sample or len(text_sample) < 10:
                return ["en"]  # Default to English

            # Simple language detection heuristics
            # In real implementation, use langdetect or similar

            language_indicators = {
                "en": ["the", "and", "or", "for", "with", "this", "that"],
                "es": ["el", "la", "y", "o", "con", "para", "este", "esta"],
                "fr": ["le", "la", "et", "ou", "avec", "pour", "ce", "cette"],
                "de": ["der", "die", "das", "und", "oder", "mit", "für"],
                "it": ["il", "la", "e", "o", "con", "per", "questo", "questa"],
            }

            text_lower = text_sample.lower()
            language_scores = {}

            for lang, indicators in language_indicators.items():
                score = sum(1 for indicator in indicators if indicator in text_lower)
                if score > 0:
                    language_scores[lang] = score

            # Return top detected languages
            if language_scores:
                sorted_langs = sorted(
                    language_scores.items(), key=lambda x: x[1], reverse=True
                )
                return [lang for lang, score in sorted_langs[:2]]
            else:
                return ["en"]  # Default fallback

        except Exception:
            return ["en"]

    def _generate_processing_recommendations(
        self, detection_result: Dict[str, Any]
    ) -> List[str]:
        """Generate processing recommendations based on detection results"""
        recommendations = []

        doc_type = detection_result["document_type"]
        confidence = detection_result["confidence_score"]
        page_count = detection_result["page_count"]

        # Confidence-based recommendations
        if confidence < 0.6:
            recommendations.append("manual_review_recommended")

        # Document type specific recommendations
        if doc_type == DocumentType.HYBRID_PDF.value:
            recommendations.append("use_ocr_fallback")
            recommendations.append("page_level_analysis")

        if doc_type == DocumentType.SCANNED_PDF.value:
            recommendations.append("high_quality_ocr")
            recommendations.append("text_correction_enabled")

        if page_count > 50:
            recommendations.append("batch_processing")
            recommendations.append("progress_checkpoints")

        if detection_result["layout_complexity"] == "complex":
            recommendations.append("layout_analysis_required")
            recommendations.append("structure_preservation")

        if detection_result["requires_splitting"]:
            recommendations.append("document_splitting_first")
            recommendations.append("separate_processing_jobs")

        # Language-based recommendations
        languages = detection_result["language_hints"]
        if len(languages) > 1:
            recommendations.append("multi_language_processing")
        if "en" not in languages:
            recommendations.append("translation_consideration")

        return recommendations


class DocumentSplitter:
    """
    Splits multi-document files into individual documents
    """

    def __init__(self):
        self.splitting_strategies = {
            DocumentType.MULTI_DOCUMENT: self._split_multi_document,
            DocumentType.HYBRID_PDF: self._split_by_page_type,
        }

    async def split_document(
        self, file_content: bytes, detection_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split document into individual components

        Args:
            file_content: Original document content
            detection_result: Result from document detection

        Returns:
            List of split document information
        """
        try:
            doc_type = DocumentType(detection_result["document_type"])

            if doc_type in self.splitting_strategies:
                strategy = self.splitting_strategies[doc_type]
                return await strategy(file_content, detection_result)
            else:
                # No splitting needed, return original document
                return [
                    {
                        "document_part": 1,
                        "total_parts": 1,
                        "content": file_content,
                        "page_range": "1-{}".format(
                            detection_result.get("page_count", 1)
                        ),
                        "processing_strategy": detection_result["processing_strategy"],
                        "metadata": detection_result,
                    }
                ]

        except Exception as e:
            logger.error(f"Document splitting failed: {e}")
            # Return original as single document on error
            return [
                {
                    "document_part": 1,
                    "total_parts": 1,
                    "content": file_content,
                    "error": str(e),
                    "metadata": detection_result,
                }
            ]

    async def _split_multi_document(
        self, file_content: bytes, detection_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split file containing multiple separate documents"""
        # This would implement actual document boundary detection and splitting
        # For now, return mock split
        return [
            {
                "document_part": 1,
                "total_parts": 2,
                "content": file_content[: len(file_content) // 2],
                "page_range": "1-5",
                "processing_strategy": ProcessingStrategy.OCR_PRIMARY.value,
            },
            {
                "document_part": 2,
                "total_parts": 2,
                "content": file_content[len(file_content) // 2 :],
                "page_range": "6-10",
                "processing_strategy": ProcessingStrategy.OCR_PRIMARY.value,
            },
        ]

    async def _split_by_page_type(
        self, file_content: bytes, detection_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Split hybrid PDF by page type (digital vs scanned)"""
        # This would analyze each page and group by processing type
        return [
            {
                "document_part": 1,
                "total_parts": 1,
                "content": file_content,
                "page_range": "1-{}".format(detection_result.get("page_count", 1)),
                "processing_strategy": detection_result["processing_strategy"],
                "split_reason": "hybrid_analysis_complete",
            }
        ]
