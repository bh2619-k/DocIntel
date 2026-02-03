"""
Real-Time Document Intelligence & Compliance Engine
Layout Analysis Engine
Version: 1.0
Date: 2026-02-03

This module provides advanced document layout analysis using LayoutLMv3 and rule-based
approaches to understand document structure, extract regions, and preserve formatting.
"""

import logging
import asyncio
from typing import Dict, List, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LayoutElement(Enum):
    """Document layout element types"""

    TITLE = "TITLE"
    HEADING = "HEADING"
    PARAGRAPH = "PARAGRAPH"
    LIST_ITEM = "LIST_ITEM"
    TABLE = "TABLE"
    TABLE_CELL = "TABLE_CELL"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    SIGNATURE = "SIGNATURE"
    WATERMARK = "WATERMARK"
    FORM_FIELD = "FORM_FIELD"
    CHECKBOX = "CHECKBOX"
    SEPARATOR = "SEPARATOR"


class ReadingOrder(Enum):
    """Document reading order patterns"""

    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"
    TOP_TO_BOTTOM = "TOP_TO_BOTTOM"
    COLUMN_BASED = "COLUMN_BASED"
    MIXED = "MIXED"


class LayoutAnalysisEngine:
    """
    Advanced layout analysis engine using ML models and rule-based approaches
    """

    def __init__(self, use_ml_models: bool = True, confidence_threshold: float = 0.7):
        """
        Initialize layout analysis engine

        Args:
            use_ml_models: Whether to use ML models (LayoutLMv3) for analysis
            confidence_threshold: Minimum confidence for element classification
        """
        self.use_ml_models = use_ml_models
        self.confidence_threshold = confidence_threshold

        # Layout analysis models
        self.models = {
            "layoutlmv3": None,  # Would be loaded LayoutLMv3 model
            "table_detector": None,  # Table detection model
            "reading_order": None,  # Reading order prediction model
        }

        # Rule-based analysis configurations
        self.rule_configs = {
            "font_size_threshold": 2.0,  # Font size difference for headings
            "line_spacing_threshold": 1.5,  # Line spacing threshold
            "margin_threshold": 20,  # Margin detection threshold
            "table_detection_threshold": 0.8,  # Table structure confidence
            "column_gap_threshold": 30,  # Column separation detection
        }

        # Initialize components
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models for layout analysis"""
        try:
            if self.use_ml_models:
                # In real implementation, load actual models
                # self.models["layoutlmv3"] = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
                logger.info("LayoutLMv3 model would be loaded here")

                # self.models["table_detector"] = load_table_detection_model()
                logger.info("Table detection model would be loaded here")

            logger.info("Layout analysis engine initialized")

        except Exception as e:
            logger.warning(
                f"Failed to initialize ML models: {e}, falling back to rule-based analysis"
            )
            self.use_ml_models = False

    async def analyze_layout(
        self, ocr_results: Dict[str, Any], image_data: bytes = None, filename: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze document layout and structure

        Args:
            ocr_results: OCR extraction results with bounding boxes
            image_data: Original document image data
            filename: Document filename for context

        Returns:
            Layout analysis results with elements, structure, and reading order
        """
        try:
            analysis_start = datetime.utcnow()

            # Initialize analysis result
            layout_result = {
                "layout_elements": [],
                "document_structure": {},
                "reading_order": [],
                "page_regions": [],
                "tables": [],
                "forms": [],
                "text_blocks": [],
                "analysis_metadata": {
                    "method_used": "rule_based",
                    "confidence_score": 0.0,
                    "processing_time_ms": 0,
                    "page_count": 1,
                },
            }

            # Extract text blocks and bounding boxes from OCR results
            text_blocks = self._extract_text_blocks(ocr_results)
            if not text_blocks:
                logger.warning("No text blocks found for layout analysis")
                return layout_result

            layout_result["text_blocks"] = text_blocks

            # Choose analysis method
            if self.use_ml_models and self.models.get("layoutlmv3"):
                layout_result = await self._analyze_with_ml_models(
                    text_blocks, image_data, layout_result
                )
                layout_result["analysis_metadata"]["method_used"] = "ml_based"
            else:
                layout_result = await self._analyze_with_rules(
                    text_blocks, layout_result
                )
                layout_result["analysis_metadata"]["method_used"] = "rule_based"

            # Post-process and validate results
            layout_result = await self._post_process_layout(layout_result)

            # Calculate processing time
            analysis_end = datetime.utcnow()
            layout_result["analysis_metadata"]["processing_time_ms"] = int(
                (analysis_end - analysis_start).total_seconds() * 1000
            )

            logger.info(
                f"Layout analysis completed: {len(layout_result['layout_elements'])} elements, "
                f"confidence: {layout_result['analysis_metadata']['confidence_score']:.2f}"
            )

            return layout_result

        except Exception as e:
            logger.error(f"Layout analysis failed: {e}")
            return {
                "layout_elements": [],
                "error": str(e),
                "analysis_metadata": {"method_used": "failed", "confidence_score": 0.0},
            }

    def _extract_text_blocks(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text blocks with bounding boxes from OCR results"""
        try:
            text_blocks = []

            # Extract from different OCR result formats
            if "blocks" in ocr_results:
                # Textract-style blocks
                for block in ocr_results.get("blocks", []):
                    if block.get("BlockType") == "LINE":
                        text_blocks.append(
                            {
                                "text": block.get("Text", ""),
                                "confidence": block.get("Confidence", 0.0) / 100,
                                "bbox": self._normalize_bbox(block.get("Geometry", {})),
                                "block_type": "line",
                                "page_number": block.get("Page", 1),
                            }
                        )

            elif "lines" in ocr_results:
                # Generic line-based results
                for line in ocr_results.get("lines", []):
                    text_blocks.append(
                        {
                            "text": line.get("text", ""),
                            "confidence": line.get("confidence", 0.8),
                            "bbox": line.get("bbox", [0, 0, 0, 0]),
                            "block_type": "line",
                            "page_number": line.get("page", 1),
                        }
                    )

            else:
                # Fallback: create single block from text
                text = ocr_results.get("text", "")
                if text:
                    lines = text.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip():
                            text_blocks.append(
                                {
                                    "text": line.strip(),
                                    "confidence": ocr_results.get(
                                        "confidence_score", 0.8
                                    ),
                                    "bbox": [
                                        0,
                                        i * 20,
                                        500,
                                        (i + 1) * 20,
                                    ],  # Estimated bbox
                                    "block_type": "line",
                                    "page_number": 1,
                                }
                            )

            logger.debug(
                f"Extracted {len(text_blocks)} text blocks for layout analysis"
            )
            return text_blocks

        except Exception as e:
            logger.error(f"Failed to extract text blocks: {e}")
            return []

    def _normalize_bbox(self, geometry: Dict[str, Any]) -> List[float]:
        """Normalize bounding box to [x1, y1, x2, y2] format"""
        try:
            if "BoundingBox" in geometry:
                bbox = geometry["BoundingBox"]
                return [
                    bbox.get("Left", 0),
                    bbox.get("Top", 0),
                    bbox.get("Left", 0) + bbox.get("Width", 0),
                    bbox.get("Top", 0) + bbox.get("Height", 0),
                ]
            else:
                return [0, 0, 1, 1]  # Default normalized coordinates

        except Exception:
            return [0, 0, 1, 1]

    async def _analyze_with_ml_models(
        self,
        text_blocks: List[Dict[str, Any]],
        image_data: bytes,
        layout_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze layout using ML models (LayoutLMv3)"""
        try:
            # Simulate ML model processing
            await asyncio.sleep(0.3)  # Simulate model inference time

            # Mock ML-based layout analysis
            layout_elements = []

            for i, block in enumerate(text_blocks):
                text = block["text"]
                bbox = block["bbox"]

                # Simulate LayoutLMv3 classification
                element_type = self._classify_element_ml(text, bbox, i)
                confidence = 0.85 + (0.1 * (i % 3))  # Mock confidence variation

                layout_elements.append(
                    {
                        "element_id": f"ml_element_{i}",
                        "element_type": element_type.value,
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence,
                        "page_number": block.get("page_number", 1),
                        "reading_order": i,
                        "font_info": {
                            "estimated_size": self._estimate_font_size(bbox),
                            "is_bold": "bold" in text.lower() or text.isupper(),
                            "is_italic": False,
                        },
                    }
                )

            layout_result["layout_elements"] = layout_elements
            layout_result["analysis_metadata"]["confidence_score"] = 0.87

            # Generate document structure
            layout_result["document_structure"] = self._build_document_structure(
                layout_elements
            )

            # Determine reading order
            layout_result["reading_order"] = self._determine_reading_order(
                layout_elements
            )

            return layout_result

        except Exception as e:
            logger.error(f"ML-based layout analysis failed: {e}")
            # Fallback to rule-based analysis
            return await self._analyze_with_rules(text_blocks, layout_result)

    async def _analyze_with_rules(
        self, text_blocks: List[Dict[str, Any]], layout_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze layout using rule-based approaches"""
        try:
            layout_elements = []

            # Sort text blocks by position (top to bottom, left to right)
            sorted_blocks = sorted(
                text_blocks, key=lambda x: (x["bbox"][1], x["bbox"][0])
            )

            for i, block in enumerate(sorted_blocks):
                text = block["text"]
                bbox = block["bbox"]

                # Rule-based element classification
                element_type = self._classify_element_rules(
                    text, bbox, i, sorted_blocks
                )
                confidence = 0.75  # Base confidence for rule-based classification

                layout_elements.append(
                    {
                        "element_id": f"rule_element_{i}",
                        "element_type": element_type.value,
                        "text": text,
                        "bbox": bbox,
                        "confidence": confidence,
                        "page_number": block.get("page_number", 1),
                        "reading_order": i,
                        "font_info": self._analyze_font_characteristics(text, bbox),
                    }
                )

            layout_result["layout_elements"] = layout_elements
            layout_result["analysis_metadata"]["confidence_score"] = 0.75

            # Detect page regions
            layout_result["page_regions"] = self._detect_page_regions(layout_elements)

            # Detect tables
            layout_result["tables"] = self._detect_tables_rule_based(layout_elements)

            # Generate document structure
            layout_result["document_structure"] = self._build_document_structure(
                layout_elements
            )

            # Determine reading order
            layout_result["reading_order"] = self._determine_reading_order(
                layout_elements
            )

            return layout_result

        except Exception as e:
            logger.error(f"Rule-based layout analysis failed: {e}")
            layout_result["error"] = str(e)
            return layout_result

    def _classify_element_ml(
        self, text: str, bbox: List[float], position: int
    ) -> LayoutElement:
        """Simulate ML-based element classification"""
        # Mock LayoutLMv3 classification
        text_lower = text.lower().strip()

        # Simple heuristics to simulate ML classification
        if position == 0 or (
            len(text) < 100
            and any(word in text_lower for word in ["title", "heading", "chapter"])
        ):
            return LayoutElement.TITLE
        elif text.startswith(("1.", "2.", "3.", "•", "-", "○")):
            return LayoutElement.LIST_ITEM
        elif len(text.split()) < 10 and text.endswith(":"):
            return LayoutElement.HEADING
        elif "|" in text or "│" in text:
            return LayoutElement.TABLE_CELL
        else:
            return LayoutElement.PARAGRAPH

    def _classify_element_rules(
        self,
        text: str,
        bbox: List[float],
        position: int,
        all_blocks: List[Dict[str, Any]],
    ) -> LayoutElement:
        """Classify layout element using rule-based approach"""
        try:
            text_stripped = text.strip()
            text_lower = text_stripped.lower()

            # Empty or very short text
            if len(text_stripped) < 3:
                return LayoutElement.SEPARATOR

            # Title detection (first element, short, centered, or specific keywords)
            if position == 0 or (
                len(text_stripped) < 100
                and any(
                    keyword in text_lower
                    for keyword in ["title", "agreement", "contract", "report"]
                )
            ):
                return LayoutElement.TITLE

            # Heading detection
            if len(text_stripped) < 80 and (
                text_stripped.endswith(":")
                or text_stripped.isupper()
                or any(
                    text_lower.startswith(prefix)
                    for prefix in ["section", "chapter", "article"]
                )
            ):
                return LayoutElement.HEADING

            # List item detection
            if text_stripped.startswith(
                (
                    "1.",
                    "2.",
                    "3.",
                    "4.",
                    "5.",
                    "a.",
                    "b.",
                    "c.",
                    "•",
                    "-",
                    "○",
                    "►",
                    "▪",
                    "▫",
                )
            ) or text_lower.startswith(("whereas", "therefore")):
                return LayoutElement.LIST_ITEM

            # Table detection
            if (
                "|" in text
                or "│" in text
                or text.count("\t") > 2
                or self._is_tabular_data(text)
            ):
                return LayoutElement.TABLE_CELL

            # Form field detection
            if (
                "_____" in text
                or text.count("_") > 5
                or any(
                    keyword in text_lower
                    for keyword in ["date:", "name:", "signature:", "amount:"]
                )
            ):
                return LayoutElement.FORM_FIELD

            # Header/Footer detection (based on position)
            bbox_height = bbox[3] - bbox[1]
            if bbox[1] < 0.1 or bbox_height < 0.05:  # Top of page
                return LayoutElement.HEADER
            elif bbox[3] > 0.9:  # Bottom of page
                return LayoutElement.FOOTER

            # Default to paragraph
            return LayoutElement.PARAGRAPH

        except Exception as e:
            logger.warning(f"Element classification failed: {e}")
            return LayoutElement.PARAGRAPH

    def _is_tabular_data(self, text: str) -> bool:
        """Detect if text represents tabular data"""
        try:
            lines = text.split("\n")
            if len(lines) < 2:
                return False

            # Check for consistent spacing patterns
            space_patterns = []
            for line in lines:
                if line.strip():
                    # Count sequences of spaces
                    import re

                    spaces = re.findall(r" {2,}", line)
                    space_patterns.append(len(spaces))

            # If multiple lines have similar space patterns, likely tabular
            if len(space_patterns) >= 2:
                avg_spaces = sum(space_patterns) / len(space_patterns)
                return avg_spaces > 1.5

            return False

        except Exception:
            return False

    def _analyze_font_characteristics(
        self, text: str, bbox: List[float]
    ) -> Dict[str, Any]:
        """Analyze font characteristics from text and bounding box"""
        try:
            font_info = {
                "estimated_size": self._estimate_font_size(bbox),
                "is_bold": False,
                "is_italic": False,
                "is_underlined": False,
                "is_capitalized": False,
            }

            # Detect formatting from text characteristics
            if text.isupper() and len(text) > 5:
                font_info["is_capitalized"] = True
                font_info["is_bold"] = True  # Often uppercase implies bold

            # Detect potential bold text (heuristics)
            if any(
                keyword in text.lower()
                for keyword in ["bold", "important", "note:", "warning:"]
            ):
                font_info["is_bold"] = True

            # Detect potential italic text
            if any(
                keyword in text.lower() for keyword in ["italic", "emphasis", "note"]
            ):
                font_info["is_italic"] = True

            return font_info

        except Exception:
            return {"estimated_size": 12, "is_bold": False, "is_italic": False}

    def _estimate_font_size(self, bbox: List[float]) -> float:
        """Estimate font size from bounding box height"""
        try:
            height = abs(bbox[3] - bbox[1])
            # Convert normalized height to approximate font size
            estimated_size = height * 1000  # Rough conversion
            return max(8, min(72, estimated_size))  # Clamp to reasonable range
        except Exception:
            return 12  # Default font size

    def _detect_page_regions(
        self, layout_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect page regions (header, body, footer, sidebar)"""
        try:
            regions = []

            if not layout_elements:
                return regions

            # Sort elements by vertical position
            sorted_elements = sorted(layout_elements, key=lambda x: x["bbox"][1])

            # Detect header region (top 15% of page)
            header_elements = [e for e in sorted_elements if e["bbox"][1] < 0.15]
            if header_elements:
                regions.append(
                    {
                        "region_type": "header",
                        "elements": [e["element_id"] for e in header_elements],
                        "bbox": self._calculate_region_bbox(header_elements),
                    }
                )

            # Detect footer region (bottom 15% of page)
            footer_elements = [e for e in sorted_elements if e["bbox"][3] > 0.85]
            if footer_elements:
                regions.append(
                    {
                        "region_type": "footer",
                        "elements": [e["element_id"] for e in footer_elements],
                        "bbox": self._calculate_region_bbox(footer_elements),
                    }
                )

            # Detect body region (middle area)
            body_elements = [e for e in sorted_elements if 0.15 <= e["bbox"][1] <= 0.85]
            if body_elements:
                regions.append(
                    {
                        "region_type": "body",
                        "elements": [e["element_id"] for e in body_elements],
                        "bbox": self._calculate_region_bbox(body_elements),
                    }
                )

            # Detect potential sidebar (elements with consistent x-position)
            # This is a simplified detection
            left_elements = [e for e in body_elements if e["bbox"][0] < 0.3]
            right_elements = [e for e in body_elements if e["bbox"][0] > 0.7]

            if len(left_elements) > 2 and len(right_elements) > 2:
                regions.append(
                    {
                        "region_type": "sidebar_left",
                        "elements": [e["element_id"] for e in left_elements],
                        "bbox": self._calculate_region_bbox(left_elements),
                    }
                )

                regions.append(
                    {
                        "region_type": "sidebar_right",
                        "elements": [e["element_id"] for e in right_elements],
                        "bbox": self._calculate_region_bbox(right_elements),
                    }
                )

            return regions

        except Exception as e:
            logger.warning(f"Page region detection failed: {e}")
            return []

    def _calculate_region_bbox(self, elements: List[Dict[str, Any]]) -> List[float]:
        """Calculate bounding box encompassing all elements in region"""
        try:
            if not elements:
                return [0, 0, 1, 1]

            bboxes = [e["bbox"] for e in elements]
            min_x = min(bbox[0] for bbox in bboxes)
            min_y = min(bbox[1] for bbox in bboxes)
            max_x = max(bbox[2] for bbox in bboxes)
            max_y = max(bbox[3] for bbox in bboxes)

            return [min_x, min_y, max_x, max_y]

        except Exception:
            return [0, 0, 1, 1]

    def _detect_tables_rule_based(
        self, layout_elements: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect tables using rule-based approach"""
        try:
            tables = []
            table_elements = [
                e
                for e in layout_elements
                if e["element_type"] == LayoutElement.TABLE_CELL.value
            ]

            if not table_elements:
                return tables

            # Group nearby table elements
            table_groups = self._group_table_elements(table_elements)

            for i, group in enumerate(table_groups):
                if len(group) >= 2:  # Minimum elements for a table
                    table_bbox = self._calculate_region_bbox(group)
                    tables.append(
                        {
                            "table_id": f"table_{i}",
                            "elements": [e["element_id"] for e in group],
                            "bbox": table_bbox,
                            "estimated_rows": self._estimate_table_rows(group),
                            "estimated_columns": self._estimate_table_columns(group),
                            "confidence": 0.7,
                        }
                    )

            return tables

        except Exception as e:
            logger.warning(f"Table detection failed: {e}")
            return []

    def _group_table_elements(
        self, table_elements: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group nearby table elements into table structures"""
        try:
            groups = []
            remaining_elements = table_elements.copy()

            while remaining_elements:
                current_group = [remaining_elements.pop(0)]
                current_bbox = current_group[0]["bbox"]

                # Find nearby elements
                to_remove = []
                for element in remaining_elements:
                    element_bbox = element["bbox"]

                    # Check if elements are close enough to be in same table
                    if (
                        abs(element_bbox[1] - current_bbox[1]) < 0.1  # Same row
                        or abs(element_bbox[0] - current_bbox[0]) < 0.1
                    ):  # Same column
                        current_group.append(element)
                        to_remove.append(element)

                # Remove grouped elements
                for element in to_remove:
                    remaining_elements.remove(element)

                if len(current_group) > 1:
                    groups.append(current_group)

            return groups

        except Exception:
            return []

    def _estimate_table_rows(self, table_elements: List[Dict[str, Any]]) -> int:
        """Estimate number of rows in table"""
        try:
            y_positions = [e["bbox"][1] for e in table_elements]
            unique_rows = len(set(round(y, 2) for y in y_positions))
            return max(1, unique_rows)
        except Exception:
            return 1

    def _estimate_table_columns(self, table_elements: List[Dict[str, Any]]) -> int:
        """Estimate number of columns in table"""
        try:
            x_positions = [e["bbox"][0] for e in table_elements]
            unique_cols = len(set(round(x, 2) for x in x_positions))
            return max(1, unique_cols)
        except Exception:
            return 1

    def _build_document_structure(
        self, layout_elements: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build hierarchical document structure"""
        try:
            structure = {
                "document_type": "unknown",
                "sections": [],
                "hierarchy": [],
                "content_flow": [],
            }

            current_section = None
            section_counter = 0

            for element in layout_elements:
                element_type = element["element_type"]
                text = element["text"]

                # Detect document type from content
                if element_type == LayoutElement.TITLE.value:
                    structure["document_type"] = self._classify_document_type(text)

                # Build sections based on headings
                if element_type in [
                    LayoutElement.TITLE.value,
                    LayoutElement.HEADING.value,
                ]:
                    if current_section:
                        structure["sections"].append(current_section)

                    section_counter += 1
                    current_section = {
                        "section_id": f"section_{section_counter}",
                        "title": text,
                        "type": element_type,
                        "elements": [element["element_id"]],
                        "bbox": element["bbox"],
                    }

                elif current_section:
                    current_section["elements"].append(element["element_id"])

                # Build content flow
                structure["content_flow"].append(
                    {
                        "element_id": element["element_id"],
                        "type": element_type,
                        "reading_order": element["reading_order"],
                    }
                )

            # Add final section
            if current_section:
                structure["sections"].append(current_section)

            return structure

        except Exception as e:
            logger.warning(f"Document structure building failed: {e}")
            return {"error": str(e)}

    def _classify_document_type(self, title_text: str) -> str:
        """Classify document type from title"""
        title_lower = title_text.lower()

        if any(word in title_lower for word in ["contract", "agreement", "terms"]):
            return "contract"
        elif any(word in title_lower for word in ["invoice", "bill", "receipt"]):
            return "invoice"
        elif any(word in title_lower for word in ["report", "analysis", "summary"]):
            return "report"
        elif any(word in title_lower for word in ["policy", "procedure", "manual"]):
            return "policy_document"
        else:
            return "general_document"

    def _determine_reading_order(
        self, layout_elements: List[Dict[str, Any]]
    ) -> List[str]:
        """Determine optimal reading order for document elements"""
        try:
            # Sort elements primarily by Y position (top to bottom)
            # Secondary sort by X position (left to right)
            sorted_elements = sorted(
                layout_elements, key=lambda x: (x["bbox"][1], x["bbox"][0])
            )

            reading_order = []

            for element in sorted_elements:
                reading_order.append(
                    {
                        "element_id": element["element_id"],
                        "reading_position": len(reading_order) + 1,
                        "confidence": 0.8,
                    }
                )

            return reading_order

        except Exception as e:
            logger.warning(f"Reading order determination failed: {e}")
            return []

    async def _post_process_layout(
        self, layout_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Post-process and validate layout analysis results"""
        try:
            # Validate element relationships
            layout_result = self._validate_element_relationships(layout_result)

            # Enhance with semantic information
            layout_result = self._add_semantic_information(layout_result)

            # Calculate overall confidence
            elements = layout_result.get("layout_elements", [])
            if elements:
                avg_confidence = sum(e.get("confidence", 0) for e in elements) / len(
                    elements
                )
                layout_result["analysis_metadata"]["confidence_score"] = avg_confidence

            return layout_result

        except Exception as e:
            logger.warning(f"Layout post-processing failed: {e}")
            return layout_result

    def _validate_element_relationships(
        self, layout_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate spatial and logical relationships between elements"""
        try:
            elements = layout_result.get("layout_elements", [])

            # Check for overlapping elements
            for i, elem1 in enumerate(elements):
                for j, elem2 in enumerate(elements[i + 1 :], i + 1):
                    if self._bboxes_overlap(elem1["bbox"], elem2["bbox"]):
                        # Handle overlapping elements
                        logger.debug(
                            f"Overlapping elements detected: {elem1['element_id']} and {elem2['element_id']}"
                        )

            return layout_result

        except Exception:
            return layout_result

    def _bboxes_overlap(self, bbox1: List[float], bbox2: List[float]) -> bool:
        """Check if two bounding boxes overlap"""
        try:
            return not (
                bbox1[2] < bbox2[0]  # bbox1 is to the left of bbox2
                or bbox1[0] > bbox2[2]  # bbox1 is to the right of bbox2
                or bbox1[3] < bbox2[1]  # bbox1 is above bbox2
                or bbox1[1] > bbox2[3]
            )  # bbox1 is below bbox2
        except Exception:
            return False

    def _add_semantic_information(
        self, layout_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add semantic information to layout elements"""
        try:
            elements = layout_result.get("layout_elements", [])

            for element in elements:
                text = element.get("text", "")
                # element_type = element.get("element_type")  # Reserved for future use

                # Add semantic tags based on content
                semantic_tags = []

                if any(
                    keyword in text.lower()
                    for keyword in ["whereas", "therefore", "hereby"]
                ):
                    semantic_tags.append("legal_language")

                if any(
                    keyword in text.lower()
                    for keyword in ["$", "amount", "price", "cost"]
                ):
                    semantic_tags.append("financial_content")

                if any(
                    keyword in text.lower() for keyword in ["date", "time", "deadline"]
                ):
                    semantic_tags.append("temporal_content")

                element["semantic_tags"] = semantic_tags

            return layout_result

        except Exception:
            return layout_result
