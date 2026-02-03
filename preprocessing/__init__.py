"""
Real-Time Document Intelligence & Compliance Engine
Document Preprocessing Module Integration
Version: 1.0
Date: 2026-02-03

This module integrates all preprocessing components into a unified preprocessing pipeline
with coordinated processing stages, error handling, and checkpoint management.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from .detection.document_detector import DocumentDetector, DocumentSplitter
from .ocr.ocr_service import OCRServiceManager
from .layout.layout_analyzer import LayoutAnalysisEngine
from .correction.text_corrector import TextCorrectionEngine
from .language.language_processor import LanguageProcessor
from .checkpoint.checkpoint_manager import CheckpointManager, ProcessingStage

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Unified document preprocessing pipeline integrating all preprocessing components
    """

    def __init__(
        self,
        use_ml_models: bool = True,
        checkpoint_storage_path: str = "./checkpoints",
        ocr_confidence_threshold: float = 0.7,
    ):
        """
        Initialize preprocessing pipeline

        Args:
            use_ml_models: Whether to use ML models for processing
            checkpoint_storage_path: Path for checkpoint storage
            ocr_confidence_threshold: Minimum confidence for OCR results
        """
        # Initialize all preprocessing components
        self.document_detector = DocumentDetector()
        self.document_splitter = DocumentSplitter()
        self.ocr_manager = OCRServiceManager(
            confidence_threshold=ocr_confidence_threshold
        )
        self.layout_analyzer = LayoutAnalysisEngine(use_ml_models=use_ml_models)
        self.text_corrector = TextCorrectionEngine(use_language_models=use_ml_models)
        self.language_processor = LanguageProcessor()
        self.checkpoint_manager = CheckpointManager(
            storage_path=checkpoint_storage_path
        )

        # Pipeline configuration
        self.config = {
            "use_ml_models": use_ml_models,
            "ocr_confidence_threshold": ocr_confidence_threshold,
            "enable_checkpoints": True,
            "enable_error_recovery": True,
            "max_retry_attempts": 3,
            "batch_processing_enabled": True,
        }

        # Processing statistics
        self.stats = {
            "documents_processed": 0,
            "successful_completions": 0,
            "failed_processing": 0,
            "recovered_documents": 0,
            "average_processing_time": 0.0,
        }

        logger.info("Document preprocessing pipeline initialized")

    async def process_document(
        self,
        document_data: bytes,
        document_id: str,
        filename: str = "",
        document_type: str = "general",
        target_language: str = "en",
    ) -> Dict[str, Any]:
        """
        Process single document through complete preprocessing pipeline

        Args:
            document_data: Document binary data
            document_id: Unique document identifier
            filename: Original filename
            document_type: Document type hint
            target_language: Target language for translation

        Returns:
            Complete preprocessing results
        """
        processing_start = datetime.now(timezone.utc)

        try:
            # Initialize processing result
            result = {
                "document_id": document_id,
                "filename": filename,
                "processing_stages": {},
                "final_results": {},
                "error_info": None,
                "processing_metadata": {
                    "start_time": processing_start.isoformat(),
                    "completion_time": None,
                    "total_processing_time_ms": 0,
                    "stages_completed": [],
                    "pipeline_version": "1.0",
                },
            }

            # Check for resumable processing
            resumable_checkpoint = (
                await self.checkpoint_manager.resume_document_processing(document_id)
            )
            if resumable_checkpoint:
                logger.info(
                    f"Resuming document processing from stage: {resumable_checkpoint.stage.value}"
                )
                result["processing_metadata"]["resumed_from"] = (
                    resumable_checkpoint.stage.value
                )
                self.stats["recovered_documents"] += 1

            # Stage 1: Document Detection and Splitting
            detection_checkpoint = await self._execute_detection_stage(
                document_data, document_id, filename, result
            )

            if not detection_checkpoint:
                return self._create_error_result(result, "Document detection failed")

            # Stage 2: OCR Processing
            ocr_checkpoint = await self._execute_ocr_stage(
                document_data, document_id, result
            )

            if not ocr_checkpoint:
                return self._create_error_result(result, "OCR processing failed")

            # Stage 3: Layout Analysis
            layout_checkpoint = await self._execute_layout_analysis_stage(
                document_id, result
            )

            if not layout_checkpoint:
                return self._create_error_result(result, "Layout analysis failed")

            # Stage 4: Text Correction
            correction_checkpoint = await self._execute_text_correction_stage(
                document_id, document_type, result
            )

            if not correction_checkpoint:
                return self._create_error_result(result, "Text correction failed")

            # Stage 5: Language Processing
            language_checkpoint = await self._execute_language_processing_stage(
                document_id, target_language, result
            )

            if not language_checkpoint:
                return self._create_error_result(result, "Language processing failed")

            # Stage 6: Finalization
            await self._finalize_processing(document_id, result)

            # Update statistics
            processing_end = datetime.now(timezone.utc)
            processing_time = (processing_end - processing_start).total_seconds() * 1000

            result["processing_metadata"]["completion_time"] = (
                processing_end.isoformat()
            )
            result["processing_metadata"]["total_processing_time_ms"] = processing_time

            self.stats["documents_processed"] += 1
            self.stats["successful_completions"] += 1
            self._update_average_processing_time(processing_time)

            logger.info(
                f"Document preprocessing completed for {document_id} in {processing_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Document preprocessing failed for {document_id}: {e}")
            self.stats["failed_processing"] += 1
            return self._create_error_result(result, str(e))

    async def _execute_detection_stage(
        self,
        document_data: bytes,
        document_id: str,
        filename: str,
        result: Dict[str, Any],
    ) -> Optional[str]:
        """Execute document detection and splitting stage"""
        try:
            # Create checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.DETECTED
            )

            # Update progress
            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id, 10, {"stage": "document_detection_started"}
            )

            # Detect document properties
            detection_result = await self.document_detector.detect_document_properties(
                document_data, filename
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id,
                50,
                intermediate_results={"detection_result": detection_result},
            )

            # Split multi-documents if needed
            if detection_result.get("is_multi_document", False):
                splitting_result = await self.document_splitter.split_document(
                    document_data, detection_result
                )
                await self.checkpoint_manager.update_checkpoint(
                    checkpoint.checkpoint_id,
                    90,
                    intermediate_results={"splitting_result": splitting_result},
                )

            # Complete checkpoint
            final_detection_result = {
                "detection": detection_result,
                "splitting": splitting_result
                if detection_result.get("is_multi_document")
                else None,
            }

            await self.checkpoint_manager.complete_checkpoint(
                checkpoint.checkpoint_id, {"detection_stage": final_detection_result}
            )

            # Store in result
            result["processing_stages"]["detection"] = final_detection_result
            result["processing_metadata"]["stages_completed"].append("detection")

            return checkpoint.checkpoint_id

        except Exception as e:
            logger.error(f"Detection stage failed: {e}")
            if "checkpoint" in locals():
                await self.checkpoint_manager.fail_checkpoint(
                    checkpoint.checkpoint_id, e
                )
            return None

    async def _execute_ocr_stage(
        self, document_data: bytes, document_id: str, result: Dict[str, Any]
    ) -> Optional[str]:
        """Execute OCR processing stage"""
        try:
            # Create checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.OCR_STARTED
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id, 10, {"stage": "ocr_processing_started"}
            )

            # Get document properties from previous stage
            detection_result = result["processing_stages"]["detection"]["detection"]

            # Process with OCR
            ocr_result = await self.ocr_manager.process_document(
                document_data,
                document_id,
                detection_result.get("processing_strategy", "auto"),
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id,
                90,
                intermediate_results={"ocr_result": ocr_result},
            )

            # Update checkpoint to completed
            await self.checkpoint_manager.complete_checkpoint(
                checkpoint.checkpoint_id, {"ocr_stage": ocr_result}
            )

            # Create completed stage checkpoint
            completed_checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.OCR_COMPLETED
            )

            await self.checkpoint_manager.complete_checkpoint(
                completed_checkpoint.checkpoint_id, {"ocr_completed": ocr_result}
            )

            # Store in result
            result["processing_stages"]["ocr"] = ocr_result
            result["processing_metadata"]["stages_completed"].extend(
                ["ocr_started", "ocr_completed"]
            )

            return completed_checkpoint.checkpoint_id

        except Exception as e:
            logger.error(f"OCR stage failed: {e}")
            if "checkpoint" in locals():
                await self.checkpoint_manager.fail_checkpoint(
                    checkpoint.checkpoint_id, e
                )
            return None

    async def _execute_layout_analysis_stage(
        self, document_id: str, result: Dict[str, Any]
    ) -> Optional[str]:
        """Execute layout analysis stage"""
        try:
            # Create checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.LAYOUT_ANALYZED
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id, 10, {"stage": "layout_analysis_started"}
            )

            # Get OCR results from previous stage
            ocr_result = result["processing_stages"]["ocr"]

            # Analyze layout
            layout_result = await self.layout_analyzer.analyze_layout(
                ocr_result, filename=result.get("filename", "")
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id,
                90,
                intermediate_results={"layout_result": layout_result},
            )

            # Complete checkpoint
            await self.checkpoint_manager.complete_checkpoint(
                checkpoint.checkpoint_id, {"layout_stage": layout_result}
            )

            # Store in result
            result["processing_stages"]["layout"] = layout_result
            result["processing_metadata"]["stages_completed"].append("layout_analysis")

            return checkpoint.checkpoint_id

        except Exception as e:
            logger.error(f"Layout analysis stage failed: {e}")
            if "checkpoint" in locals():
                await self.checkpoint_manager.fail_checkpoint(
                    checkpoint.checkpoint_id, e
                )
            return None

    async def _execute_text_correction_stage(
        self, document_id: str, document_type: str, result: Dict[str, Any]
    ) -> Optional[str]:
        """Execute text correction stage"""
        try:
            # Create checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.TEXT_CORRECTED
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id, 10, {"stage": "text_correction_started"}
            )

            # Get OCR and layout results
            ocr_result = result["processing_stages"]["ocr"]
            layout_result = result["processing_stages"]["layout"]

            # Extract text blocks for correction
            text_blocks = layout_result.get("text_blocks", [])
            if not text_blocks and ocr_result.get("text"):
                # Fallback to raw OCR text
                text_blocks = [
                    {
                        "text": ocr_result["text"],
                        "confidence": ocr_result.get("confidence_score", 0.8),
                    }
                ]

            # Apply text correction
            corrected_blocks = await self.text_corrector.correct_batch_text(
                text_blocks, document_type
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id,
                90,
                intermediate_results={"corrected_blocks": corrected_blocks},
            )

            # Create correction summary
            correction_result = {
                "corrected_text_blocks": corrected_blocks,
                "correction_statistics": self.text_corrector.get_correction_statistics(
                    [
                        correction
                        for block in corrected_blocks
                        for correction in block.get("corrections", [])
                    ]
                ),
            }

            # Complete checkpoint
            await self.checkpoint_manager.complete_checkpoint(
                checkpoint.checkpoint_id, {"correction_stage": correction_result}
            )

            # Store in result
            result["processing_stages"]["text_correction"] = correction_result
            result["processing_metadata"]["stages_completed"].append("text_correction")

            return checkpoint.checkpoint_id

        except Exception as e:
            logger.error(f"Text correction stage failed: {e}")
            if "checkpoint" in locals():
                await self.checkpoint_manager.fail_checkpoint(
                    checkpoint.checkpoint_id, e
                )
            return None

    async def _execute_language_processing_stage(
        self, document_id: str, target_language: str, result: Dict[str, Any]
    ) -> Optional[str]:
        """Execute language detection and processing stage"""
        try:
            # Create checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.LANGUAGE_PROCESSED
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id, 10, {"stage": "language_processing_started"}
            )

            # Get corrected text blocks
            correction_result = result["processing_stages"]["text_correction"]
            text_blocks = correction_result["corrected_text_blocks"]

            # Process multilingual content
            language_result = (
                await self.language_processor.process_multilingual_document(
                    text_blocks, target_language
                )
            )

            await self.checkpoint_manager.update_checkpoint(
                checkpoint.checkpoint_id,
                90,
                intermediate_results={"language_result": language_result},
            )

            # Complete checkpoint
            await self.checkpoint_manager.complete_checkpoint(
                checkpoint.checkpoint_id, {"language_stage": language_result}
            )

            # Store in result
            result["processing_stages"]["language_processing"] = language_result
            result["processing_metadata"]["stages_completed"].append(
                "language_processing"
            )

            return checkpoint.checkpoint_id

        except Exception as e:
            logger.error(f"Language processing stage failed: {e}")
            if "checkpoint" in locals():
                await self.checkpoint_manager.fail_checkpoint(
                    checkpoint.checkpoint_id, e
                )
            return None

    async def _finalize_processing(self, document_id: str, result: Dict[str, Any]):
        """Finalize document preprocessing"""
        try:
            # Create final checkpoint
            checkpoint = await self.checkpoint_manager.create_checkpoint(
                document_id, ProcessingStage.PREPROCESSING_COMPLETED
            )

            # Compile final results
            final_results = {
                "document_properties": result["processing_stages"]["detection"][
                    "detection"
                ],
                "extracted_text": self._compile_final_text(result),
                "document_structure": result["processing_stages"]["layout"].get(
                    "document_structure", {}
                ),
                "language_information": result["processing_stages"][
                    "language_processing"
                ]["processing_metadata"],
                "quality_metrics": self._calculate_quality_metrics(result),
                "processing_summary": {
                    "stages_completed": result["processing_metadata"][
                        "stages_completed"
                    ],
                    "total_processing_time": result["processing_metadata"].get(
                        "total_processing_time_ms", 0
                    ),
                    "confidence_score": self._calculate_overall_confidence(result),
                },
            }

            # Complete final checkpoint
            await self.checkpoint_manager.complete_checkpoint(
                checkpoint.checkpoint_id, {"final_results": final_results}
            )

            # Store final results
            result["final_results"] = final_results
            result["processing_metadata"]["stages_completed"].append(
                "preprocessing_completed"
            )

            logger.info(f"Finalized preprocessing for document {document_id}")

        except Exception as e:
            logger.error(f"Failed to finalize processing: {e}")
            if "checkpoint" in locals():
                await self.checkpoint_manager.fail_checkpoint(
                    checkpoint.checkpoint_id, e
                )

    def _compile_final_text(self, result: Dict[str, Any]) -> str:
        """Compile final corrected and processed text"""
        try:
            language_result = result["processing_stages"]["language_processing"]
            processed_blocks = language_result["processed_blocks"]

            final_text_parts = []
            for block in processed_blocks:
                # Use translated text if available, otherwise use corrected text
                text = (
                    block.get("translated_text")
                    or block.get("corrected_text")
                    or block.get("text", "")
                )
                if text.strip():
                    final_text_parts.append(text.strip())

            return "\n\n".join(final_text_parts)

        except Exception as e:
            logger.warning(f"Failed to compile final text: {e}")
            # Fallback to OCR text
            return result.get("processing_stages", {}).get("ocr", {}).get("text", "")

    def _calculate_quality_metrics(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics"""
        try:
            metrics = {
                "ocr_confidence": 0.0,
                "layout_confidence": 0.0,
                "correction_confidence": 0.0,
                "language_confidence": 0.0,
                "overall_confidence": 0.0,
            }

            # OCR confidence
            ocr_result = result.get("processing_stages", {}).get("ocr", {})
            metrics["ocr_confidence"] = ocr_result.get("confidence_score", 0.0)

            # Layout confidence
            layout_result = result.get("processing_stages", {}).get("layout", {})
            metrics["layout_confidence"] = layout_result.get(
                "analysis_metadata", {}
            ).get("confidence_score", 0.0)

            # Correction confidence
            correction_result = result.get("processing_stages", {}).get(
                "text_correction", {}
            )
            correction_stats = correction_result.get("correction_statistics", {})
            metrics["correction_confidence"] = correction_stats.get(
                "avg_confidence", 0.0
            )

            # Language confidence
            language_result = result.get("processing_stages", {}).get(
                "language_processing", {}
            )
            # Calculate average detection confidence
            processed_blocks = language_result.get("processed_blocks", [])
            if processed_blocks:
                total_confidence = sum(
                    block.get("detection_confidence", 0.0) for block in processed_blocks
                )
                metrics["language_confidence"] = total_confidence / len(
                    processed_blocks
                )

            # Overall confidence (weighted average)
            weights = {"ocr": 0.3, "layout": 0.2, "correction": 0.25, "language": 0.25}
            weighted_sum = (
                metrics["ocr_confidence"] * weights["ocr"]
                + metrics["layout_confidence"] * weights["layout"]
                + metrics["correction_confidence"] * weights["correction"]
                + metrics["language_confidence"] * weights["language"]
            )
            metrics["overall_confidence"] = weighted_sum

            return metrics

        except Exception as e:
            logger.warning(f"Failed to calculate quality metrics: {e}")
            return {"overall_confidence": 0.0}

    def _calculate_overall_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate overall processing confidence"""
        try:
            quality_metrics = self._calculate_quality_metrics(result)
            return quality_metrics.get("overall_confidence", 0.0)
        except Exception:
            return 0.0

    def _create_error_result(
        self, result: Dict[str, Any], error_message: str
    ) -> Dict[str, Any]:
        """Create error result structure"""
        result["error_info"] = {
            "error_message": error_message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages_completed": result["processing_metadata"]["stages_completed"],
        }
        result["processing_metadata"]["completion_time"] = datetime.now(
            timezone.utc
        ).isoformat()
        return result

    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time statistics"""
        try:
            current_avg = self.stats["average_processing_time"]
            total_processed = self.stats["documents_processed"]

            # Calculate new average
            new_avg = (
                (current_avg * (total_processed - 1)) + processing_time
            ) / total_processed
            self.stats["average_processing_time"] = new_avg

        except Exception:
            pass

    async def process_batch_documents(
        self, documents: List[Dict[str, Any]], max_concurrent: int = 3
    ) -> List[Dict[str, Any]]:
        """Process multiple documents concurrently"""
        try:
            logger.info(f"Starting batch processing of {len(documents)} documents")

            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_single_doc(doc_info):
                async with semaphore:
                    return await self.process_document(
                        doc_info["data"],
                        doc_info["id"],
                        doc_info.get("filename", ""),
                        doc_info.get("document_type", "general"),
                        doc_info.get("target_language", "en"),
                    )

            # Process documents concurrently
            tasks = [process_single_doc(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_result = {
                        "document_id": documents[i]["id"],
                        "error_info": {"error_message": str(result)},
                        "processing_stages": {},
                    }
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)

            logger.info(f"Batch processing completed for {len(documents)} documents")
            return processed_results

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        try:
            checkpoint_stats = self.checkpoint_manager.get_system_statistics()

            return {
                "processing_statistics": self.stats,
                "checkpoint_statistics": checkpoint_stats,
                "pipeline_configuration": self.config,
                "component_status": {
                    "document_detector": "active",
                    "ocr_manager": "active",
                    "layout_analyzer": "active",
                    "text_corrector": "active",
                    "language_processor": "active",
                    "checkpoint_manager": "active",
                },
            }

        except Exception as e:
            logger.error(f"Failed to get pipeline statistics: {e}")
            return {"error": str(e)}
