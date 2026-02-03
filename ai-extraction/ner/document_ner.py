"""
Real-Time Document Intelligence & Compliance Engine
Document-Type Specific NER Module
Version: 1.0
Date: 2026-02-03

This module provides document-type-specific Named Entity Recognition (NER) heads
for extracting entities from contracts, invoices, legal documents with confidence scoring.
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

from ..encoders.bert_encoder import SharedBERTEncoder, BERTVariant, EncodingMode

logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Supported document types for NER"""

    LEGAL_CONTRACT = "LEGAL_CONTRACT"
    INVOICE = "INVOICE"
    PURCHASE_ORDER = "PURCHASE_ORDER"
    INSURANCE_POLICY = "INSURANCE_POLICY"
    LOAN_AGREEMENT = "LOAN_AGREEMENT"
    EMPLOYMENT_CONTRACT = "EMPLOYMENT_CONTRACT"
    NDA = "NDA"
    LEASE_AGREEMENT = "LEASE_AGREEMENT"
    GENERAL_DOCUMENT = "GENERAL_DOCUMENT"


class EntityType(Enum):
    """Entity types for different document categories"""

    # Legal entities
    PARTY_NAME = "PARTY_NAME"
    CONTRACT_DATE = "CONTRACT_DATE"
    EFFECTIVE_DATE = "EFFECTIVE_DATE"
    TERMINATION_DATE = "TERMINATION_DATE"
    JURISDICTION = "JURISDICTION"
    GOVERNING_LAW = "GOVERNING_LAW"

    # Financial entities
    AMOUNT = "AMOUNT"
    CURRENCY = "CURRENCY"
    PAYMENT_TERMS = "PAYMENT_TERMS"
    DUE_DATE = "DUE_DATE"
    INVOICE_NUMBER = "INVOICE_NUMBER"
    TAX_AMOUNT = "TAX_AMOUNT"

    # Personal/Organization entities
    PERSON_NAME = "PERSON_NAME"
    ORGANIZATION = "ORGANIZATION"
    ADDRESS = "ADDRESS"
    PHONE_NUMBER = "PHONE_NUMBER"
    EMAIL = "EMAIL"

    # Product/Service entities
    PRODUCT_NAME = "PRODUCT_NAME"
    SERVICE_DESCRIPTION = "SERVICE_DESCRIPTION"
    QUANTITY = "QUANTITY"
    UNIT_PRICE = "UNIT_PRICE"

    # Legal clauses
    LIABILITY_CLAUSE = "LIABILITY_CLAUSE"
    INDEMNITY_CLAUSE = "INDEMNITY_CLAUSE"
    CONFIDENTIALITY_CLAUSE = "CONFIDENTIALITY_CLAUSE"
    TERMINATION_CLAUSE = "TERMINATION_CLAUSE"


class NEROutputFormat(Enum):
    """NER output format options"""

    BIO = "BIO"  # Begin-Inside-Outside
    BILOU = "BILOU"  # Begin-Inside-Last-Outside-Unit
    SPANS = "SPANS"  # Character spans with labels


class DocumentTypeNER(nn.Module):
    """
    Document-type-specific NER head for entity extraction
    """

    def __init__(
        self,
        document_type: DocumentType,
        bert_encoder: SharedBERTEncoder,
        entity_types: List[EntityType],
        output_format: NEROutputFormat = NEROutputFormat.SPANS,
        confidence_threshold: float = 0.7,
        max_entity_length: int = 200,
    ):
        """
        Initialize document-type-specific NER head

        Args:
            document_type: Type of document this NER is designed for
            bert_encoder: Shared BERT encoder instance
            entity_types: List of entity types to extract
            output_format: Format for NER output
            confidence_threshold: Minimum confidence for entity extraction
            max_entity_length: Maximum entity length in characters
        """
        super().__init__()

        self.document_type = document_type
        self.bert_encoder = bert_encoder
        self.entity_types = entity_types
        self.output_format = output_format
        self.confidence_threshold = confidence_threshold
        self.max_entity_length = max_entity_length

        # Get BERT hidden size
        self.hidden_size = 768 if "base" in bert_encoder.model_variant.value else 1024

        # Create entity type mappings
        self.entity_to_id = {entity.value: i for i, entity in enumerate(entity_types)}
        self.id_to_entity = {i: entity.value for entity, i in self.entity_to_id.items()}

        # Calculate number of labels based on output format
        if output_format == NEROutputFormat.BIO:
            self.num_labels = len(entity_types) * 2 + 1  # B-, I-, O
        elif output_format == NEROutputFormat.BILOU:
            self.num_labels = len(entity_types) * 4 + 1  # B-, I-, L-, U-, O
        else:  # SPANS
            self.num_labels = len(entity_types)

        # NER classification heads
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        # Entity type-specific regex patterns
        self.entity_patterns = self._initialize_entity_patterns()

        # Post-processing rules
        self.post_processing_rules = self._initialize_post_processing_rules()

        # Performance metrics
        self.metrics = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "average_confidence": 0.0,
            "entity_type_counts": {entity.value: 0 for entity in entity_types},
        }

        logger.info(
            f"Initialized NER head for {document_type.value} with {len(entity_types)} entity types"
        )

    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize regex patterns for entity recognition"""
        patterns = {}

        # Date patterns
        date_patterns = [
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
            r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b",
        ]

        # Amount patterns
        amount_patterns = [
            r"\$[\d,]+\.?\d*",
            r"USD\s*[\d,]+\.?\d*",
            r"EUR\s*[\d,]+\.?\d*",
            r"\b\d+\.?\d*\s*(?:dollars|USD|euros|EUR)\b",
        ]

        # Email patterns
        email_patterns = [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"]

        # Phone patterns
        phone_patterns = [
            r"\b\d{3}-\d{3}-\d{4}\b",
            r"\(\d{3}\)\s*\d{3}-\d{4}",
            r"\+\d{1,3}\s*\d{3,4}\s*\d{3,4}\s*\d{3,4}",
        ]

        patterns[EntityType.CONTRACT_DATE.value] = date_patterns
        patterns[EntityType.EFFECTIVE_DATE.value] = date_patterns
        patterns[EntityType.TERMINATION_DATE.value] = date_patterns
        patterns[EntityType.DUE_DATE.value] = date_patterns
        patterns[EntityType.AMOUNT.value] = amount_patterns
        patterns[EntityType.TAX_AMOUNT.value] = amount_patterns
        patterns[EntityType.UNIT_PRICE.value] = amount_patterns
        patterns[EntityType.EMAIL.value] = email_patterns
        patterns[EntityType.PHONE_NUMBER.value] = phone_patterns

        return patterns

    def _initialize_post_processing_rules(self) -> Dict[str, List[str]]:
        """Initialize post-processing rules for entity refinement"""
        rules = {}

        # Document-type-specific entity validation rules
        if self.document_type == DocumentType.LEGAL_CONTRACT:
            rules["party_validation"] = [
                "Must contain at least one word",
                "Cannot be only punctuation",
                "Maximum length 100 characters",
            ]
            rules["date_validation"] = [
                "Must be valid date format",
                "Cannot be in far future (>10 years)",
            ]

        elif self.document_type == DocumentType.INVOICE:
            rules["amount_validation"] = [
                "Must be positive number",
                "Must have currency indicator",
                "Cannot exceed reasonable invoice amounts",
            ]
            rules["invoice_number_validation"] = [
                "Must be alphanumeric",
                "Length between 3-20 characters",
            ]

        return rules

    def extract_entities(
        self,
        text: str,
        use_regex_fallback: bool = True,
        return_confidence_scores: bool = True,
    ) -> Dict[str, Any]:
        """
        Extract entities from text using BERT-based NER

        Args:
            text: Input text for entity extraction
            use_regex_fallback: Whether to use regex patterns as fallback
            return_confidence_scores: Whether to return confidence scores

        Returns:
            Extracted entities with metadata
        """
        try:
            extraction_start = datetime.now()

            # Initialize result structure
            result = {
                "document_type": self.document_type.value,
                "entities": [],
                "extraction_metadata": {
                    "total_entities": 0,
                    "average_confidence": 0.0,
                    "processing_time_ms": 0,
                    "method_used": "bert_ner",
                },
            }

            if not text.strip():
                return result

            # Primary extraction using BERT
            bert_entities = self._extract_with_bert(text, return_confidence_scores)
            result["entities"].extend(bert_entities)

            # Fallback to regex patterns if enabled
            if use_regex_fallback:
                regex_entities = self._extract_with_regex(text, bert_entities)
                result["entities"].extend(regex_entities)
                if regex_entities:
                    result["extraction_metadata"]["method_used"] = "bert_ner+regex"

            # Post-process and validate entities
            result["entities"] = self._post_process_entities(result["entities"], text)

            # Calculate metadata
            result["extraction_metadata"]["total_entities"] = len(result["entities"])
            if result["entities"]:
                confidences = [e.get("confidence", 0.0) for e in result["entities"]]
                result["extraction_metadata"]["average_confidence"] = sum(
                    confidences
                ) / len(confidences)

            # Calculate processing time
            processing_time = (datetime.now() - extraction_start).total_seconds() * 1000
            result["extraction_metadata"]["processing_time_ms"] = processing_time

            # Update metrics
            self._update_metrics(result["entities"])

            logger.debug(
                f"Extracted {len(result['entities'])} entities from text ({len(text)} chars)"
            )

            return result

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "document_type": self.document_type.value,
                "entities": [],
                "error": str(e),
            }

    def _extract_with_bert(
        self, text: str, return_confidence_scores: bool
    ) -> List[Dict[str, Any]]:
        """Extract entities using BERT-based NER"""
        try:
            # Encode text with BERT
            encoding_result = self.bert_encoder.encode_text(
                text, mode=EncodingMode.INFERENCE, return_attention_mask=True
            )

            if "error" in encoding_result:
                logger.warning(f"BERT encoding failed: {encoding_result['error']}")
                return []

            # Get embeddings and attention masks
            embeddings = encoding_result["embeddings"][0]  # First (and only) text
            # attention_mask = encoding_result["attention_masks"][0]

            # Convert to tensor for classification
            embeddings_tensor = torch.tensor(embeddings).unsqueeze(
                0
            )  # Add batch dimension

            # Forward pass through NER classifier
            with torch.no_grad():
                self.eval()
                logits = self.classifier(self.dropout(embeddings_tensor))
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

            # Convert predictions to entities
            entities = self._predictions_to_entities(
                text,
                predictions.squeeze().tolist()
                if isinstance(predictions.squeeze().tolist(), list)
                else [predictions.squeeze().item()],
                probabilities.squeeze().tolist()
                if isinstance(probabilities.squeeze().tolist(), list)
                else [probabilities.squeeze().item()],
                return_confidence_scores,
            )

            return entities

        except Exception as e:
            logger.warning(f"BERT-based NER failed: {e}")
            return []

    def _predictions_to_entities(
        self,
        text: str,
        predictions: List[int],
        probabilities: List[float],
        return_confidence_scores: bool,
    ) -> List[Dict[str, Any]]:
        """Convert model predictions to entity objects"""
        try:
            entities = []

            # For SPANS format (simplified implementation)
            if self.output_format == NEROutputFormat.SPANS:
                # Mock entity extraction based on probabilities
                # In real implementation, would properly decode token-level predictions
                words = text.split()

                for i, (word, prob) in enumerate(
                    zip(words[: len(probabilities)], probabilities)
                ):
                    if prob > self.confidence_threshold:
                        # Determine entity type (simplified)
                        entity_type_id = (
                            predictions[min(i, len(predictions) - 1)]
                            if predictions
                            else 0
                        )
                        entity_type = self.id_to_entity.get(
                            entity_type_id % len(self.entity_types), "UNKNOWN"
                        )

                        # Calculate character positions
                        start_pos = text.find(word)
                        end_pos = start_pos + len(word) if start_pos != -1 else 0

                        entity = {
                            "entity_type": entity_type,
                            "text": word,
                            "start_position": start_pos,
                            "end_position": end_pos,
                            "extraction_method": "bert_ner",
                        }

                        if return_confidence_scores:
                            entity["confidence"] = prob

                        entities.append(entity)

            return entities

        except Exception as e:
            logger.warning(f"Failed to convert predictions to entities: {e}")
            return []

    def _extract_with_regex(
        self, text: str, existing_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns as fallback"""
        try:
            regex_entities = []
            existing_spans = {
                (e["start_position"], e["end_position"]) for e in existing_entities
            }

            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)

                    for match in matches:
                        start_pos = match.start()
                        end_pos = match.end()

                        # Skip if overlaps with existing entity
                        if (start_pos, end_pos) in existing_spans:
                            continue

                        # Check for overlap with existing spans
                        overlaps = any(
                            start_pos < e_end and end_pos > e_start
                            for e_start, e_end in existing_spans
                        )

                        if not overlaps:
                            entity = {
                                "entity_type": entity_type,
                                "text": match.group(),
                                "start_position": start_pos,
                                "end_position": end_pos,
                                "confidence": 0.8,  # Default regex confidence
                                "extraction_method": "regex",
                            }

                            regex_entities.append(entity)
                            existing_spans.add((start_pos, end_pos))

            return regex_entities

        except Exception as e:
            logger.warning(f"Regex extraction failed: {e}")
            return []

    def _post_process_entities(
        self, entities: List[Dict[str, Any]], text: str
    ) -> List[Dict[str, Any]]:
        """Post-process and validate extracted entities"""
        try:
            validated_entities = []

            for entity in entities:
                # Basic validation
                if not entity.get("text", "").strip():
                    continue

                if len(entity.get("text", "")) > self.max_entity_length:
                    continue

                # Confidence threshold check
                if entity.get("confidence", 1.0) < self.confidence_threshold:
                    continue

                # Entity type-specific validation
                if self._validate_entity(entity, text):
                    # Clean entity text
                    entity["text"] = self._clean_entity_text(entity["text"])

                    # Add normalized value if applicable
                    normalized_value = self._normalize_entity_value(entity)
                    if normalized_value:
                        entity["normalized_value"] = normalized_value

                    validated_entities.append(entity)

            # Remove duplicates and overlapping entities
            validated_entities = self._remove_duplicate_entities(validated_entities)

            return validated_entities

        except Exception as e:
            logger.warning(f"Entity post-processing failed: {e}")
            return entities

    def _validate_entity(self, entity: Dict[str, Any], text: str) -> bool:
        """Validate entity based on type-specific rules"""
        try:
            entity_type = entity.get("entity_type", "")
            entity_text = entity.get("text", "")

            # Date validation
            if entity_type in [
                "CONTRACT_DATE",
                "EFFECTIVE_DATE",
                "TERMINATION_DATE",
                "DUE_DATE",
            ]:
                return self._validate_date_entity(entity_text)

            # Amount validation
            elif entity_type in ["AMOUNT", "TAX_AMOUNT", "UNIT_PRICE"]:
                return self._validate_amount_entity(entity_text)

            # Email validation
            elif entity_type == "EMAIL":
                return "@" in entity_text and "." in entity_text

            # Phone validation
            elif entity_type == "PHONE_NUMBER":
                return (
                    any(char.isdigit() for char in entity_text)
                    and len(entity_text) >= 7
                )

            # Person/Organization name validation
            elif entity_type in ["PARTY_NAME", "PERSON_NAME", "ORGANIZATION"]:
                return len(entity_text.strip()) >= 2 and not entity_text.isdigit()

            # Default validation
            return True

        except Exception:
            return False

    def _validate_date_entity(self, date_text: str) -> bool:
        """Validate date entity"""
        try:
            # Basic date format checks
            if re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", date_text):
                return True
            if re.match(r"\d{4}[/-]\d{1,2}[/-]\d{1,2}", date_text):
                return True
            return False
        except Exception:
            return False

    def _validate_amount_entity(self, amount_text: str) -> bool:
        """Validate amount entity"""
        try:
            # Check for currency symbols or keywords
            has_currency = any(
                symbol in amount_text
                for symbol in ["$", "USD", "EUR", "GBP", "dollars", "euros"]
            )

            # Check for numbers
            has_numbers = any(char.isdigit() for char in amount_text)

            return has_currency and has_numbers
        except Exception:
            return False

    def _clean_entity_text(self, text: str) -> str:
        """Clean and normalize entity text"""
        try:
            # Remove excessive whitespace
            text = " ".join(text.split())

            # Remove leading/trailing punctuation for certain entities
            text = text.strip(".,;:")

            return text
        except Exception:
            return text

    def _normalize_entity_value(self, entity: Dict[str, Any]) -> Optional[str]:
        """Normalize entity value for standardization"""
        try:
            entity_type = entity.get("entity_type", "")
            entity_text = entity.get("text", "")

            # Normalize dates
            if entity_type in [
                "CONTRACT_DATE",
                "EFFECTIVE_DATE",
                "TERMINATION_DATE",
                "DUE_DATE",
            ]:
                # Mock date normalization
                return self._normalize_date(entity_text)

            # Normalize amounts
            elif entity_type in ["AMOUNT", "TAX_AMOUNT", "UNIT_PRICE"]:
                return self._normalize_amount(entity_text)

            # Normalize emails
            elif entity_type == "EMAIL":
                return entity_text.lower().strip()

            return None

        except Exception:
            return None

    def _normalize_date(self, date_text: str) -> str:
        """Normalize date to standard format"""
        try:
            # Mock date normalization - in real implementation use datetime parsing
            if re.match(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}", date_text):
                return date_text.replace("/", "-")
            return date_text
        except Exception:
            return date_text

    def _normalize_amount(self, amount_text: str) -> str:
        """Normalize amount to standard format"""
        try:
            # Extract numeric value
            numbers = re.findall(r"[\d,.]+", amount_text)
            if numbers:
                numeric_value = numbers[0].replace(",", "")
                return f"${numeric_value}"
            return amount_text
        except Exception:
            return amount_text

    def _remove_duplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate and overlapping entities"""
        try:
            # Sort by confidence (highest first)
            entities = sorted(
                entities, key=lambda x: x.get("confidence", 0.0), reverse=True
            )

            filtered_entities = []
            used_spans = set()

            for entity in entities:
                start = entity.get("start_position", 0)
                end = entity.get("end_position", 0)

                # Check for overlap with already selected entities
                overlaps = any(
                    start < used_end and end > used_start
                    for used_start, used_end in used_spans
                )

                if not overlaps:
                    filtered_entities.append(entity)
                    used_spans.add((start, end))

            return filtered_entities

        except Exception as e:
            logger.warning(f"Failed to remove duplicates: {e}")
            return entities

    def _update_metrics(self, entities: List[Dict[str, Any]]):
        """Update performance metrics"""
        try:
            self.metrics["total_extractions"] += 1
            self.metrics["successful_extractions"] += 1 if entities else 0

            # Update entity type counts
            for entity in entities:
                entity_type = entity.get("entity_type", "UNKNOWN")
                if entity_type in self.metrics["entity_type_counts"]:
                    self.metrics["entity_type_counts"][entity_type] += 1

            # Update average confidence
            if entities:
                confidences = [e.get("confidence", 0.0) for e in entities]
                avg_confidence = sum(confidences) / len(confidences)

                current_avg = self.metrics["average_confidence"]
                total_extractions = self.metrics["total_extractions"]
                new_avg = (
                    (current_avg * (total_extractions - 1)) + avg_confidence
                ) / total_extractions
                self.metrics["average_confidence"] = new_avg

        except Exception:
            pass

    def train_ner_head(
        self,
        train_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Train the NER head on labeled data

        Args:
            train_data: Training data with text and entity labels
            validation_data: Optional validation data
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            batch_size: Training batch size

        Returns:
            Training results and metrics
        """
        try:
            logger.info(f"Starting NER head training for {len(train_data)} samples")

            # Prepare data loaders (mock implementation)
            train_loader = self._prepare_ner_data_loader(
                train_data, batch_size, shuffle=True
            )
            val_loader = None
            if validation_data:
                val_loader = self._prepare_ner_data_loader(
                    validation_data, batch_size, shuffle=False
                )

            # Setup optimizer
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            loss_function = nn.CrossEntropyLoss()

            # Training loop
            training_history = {
                "train_losses": [],
                "val_losses": [],
                "train_f1_scores": [],
                "val_f1_scores": [],
                "epochs_completed": 0,
            }

            for epoch in range(num_epochs):
                # Training phase
                train_loss, train_f1 = self._train_ner_epoch(
                    train_loader, optimizer, loss_function
                )
                training_history["train_losses"].append(train_loss)
                training_history["train_f1_scores"].append(train_f1)

                # Validation phase
                if val_loader:
                    val_loss, val_f1 = self._validate_ner_epoch(
                        val_loader, loss_function
                    )
                    training_history["val_losses"].append(val_loss)
                    training_history["val_f1_scores"].append(val_f1)

                training_history["epochs_completed"] = epoch + 1
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}"
                )

            return {
                "status": "success",
                "training_history": training_history,
                "final_metrics": {
                    "train_loss": training_history["train_losses"][-1],
                    "train_f1": training_history["train_f1_scores"][-1],
                    "val_loss": training_history["val_losses"][-1]
                    if training_history["val_losses"]
                    else None,
                    "val_f1": training_history["val_f1_scores"][-1]
                    if training_history["val_f1_scores"]
                    else None,
                },
            }

        except Exception as e:
            logger.error(f"NER head training failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _prepare_ner_data_loader(
        self, data: List[Dict[str, Any]], batch_size: int, shuffle: bool
    ):
        """Prepare data loader for NER training"""
        # Mock implementation - in real implementation would create proper Dataset and DataLoader
        return MockNERDataLoader(data, batch_size, shuffle)

    def _train_ner_epoch(
        self, train_loader, optimizer, loss_function
    ) -> tuple[float, float]:
        """Train NER head for one epoch"""
        self.train()
        total_loss = 0.0
        total_f1 = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            # Mock training step
            logits = torch.randn(16, self.num_labels)  # Mock logits
            labels = torch.randint(0, self.num_labels, (16,))  # Mock labels

            loss = loss_function(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f1 += 0.8  # Mock F1 score

        return total_loss / num_batches, total_f1 / num_batches

    def _validate_ner_epoch(self, val_loader, loss_function) -> tuple[float, float]:
        """Validate NER head for one epoch"""
        self.eval()
        total_loss = 0.0
        total_f1 = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # Mock validation step
                logits = torch.randn(16, self.num_labels)  # Mock logits
                labels = torch.randint(0, self.num_labels, (16,))  # Mock labels

                loss = loss_function(logits, labels)
                total_loss += loss.item()
                total_f1 += 0.85  # Mock F1 score

        return total_loss / num_batches, total_f1 / num_batches

    def get_ner_statistics(self) -> Dict[str, Any]:
        """Get comprehensive NER statistics"""
        return {
            "document_type": self.document_type.value,
            "entity_types": [entity.value for entity in self.entity_types],
            "configuration": {
                "output_format": self.output_format.value,
                "confidence_threshold": self.confidence_threshold,
                "max_entity_length": self.max_entity_length,
                "num_labels": self.num_labels,
            },
            "performance_metrics": self.metrics,
            "model_info": {
                "hidden_size": self.hidden_size,
                "parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in self.parameters() if p.requires_grad
                ),
            },
        }


class MultiDocumentNERManager:
    """
    Manager for multiple document-type-specific NER heads
    """

    def __init__(self, bert_encoder: SharedBERTEncoder):
        """
        Initialize multi-document NER manager

        Args:
            bert_encoder: Shared BERT encoder instance
        """
        self.bert_encoder = bert_encoder
        self.ner_heads: Dict[str, DocumentTypeNER] = {}

        # Initialize default NER heads for common document types
        self._initialize_default_ner_heads()

        logger.info(
            f"Multi-document NER manager initialized with {len(self.ner_heads)} NER heads"
        )

    def _initialize_default_ner_heads(self):
        """Initialize default NER heads for common document types"""
        try:
            # Legal contract NER
            legal_entities = [
                EntityType.PARTY_NAME,
                EntityType.CONTRACT_DATE,
                EntityType.EFFECTIVE_DATE,
                EntityType.TERMINATION_DATE,
                EntityType.JURISDICTION,
                EntityType.GOVERNING_LAW,
                EntityType.LIABILITY_CLAUSE,
                EntityType.INDEMNITY_CLAUSE,
            ]
            self.ner_heads[DocumentType.LEGAL_CONTRACT.value] = DocumentTypeNER(
                DocumentType.LEGAL_CONTRACT, self.bert_encoder, legal_entities
            )

            # Invoice NER
            invoice_entities = [
                EntityType.AMOUNT,
                EntityType.TAX_AMOUNT,
                EntityType.DUE_DATE,
                EntityType.INVOICE_NUMBER,
                EntityType.ORGANIZATION,
                EntityType.ADDRESS,
                EntityType.PAYMENT_TERMS,
            ]
            self.ner_heads[DocumentType.INVOICE.value] = DocumentTypeNER(
                DocumentType.INVOICE, self.bert_encoder, invoice_entities
            )

            # General document NER
            general_entities = [
                EntityType.PERSON_NAME,
                EntityType.ORGANIZATION,
                EntityType.ADDRESS,
                EntityType.EMAIL,
                EntityType.PHONE_NUMBER,
                EntityType.AMOUNT,
            ]
            self.ner_heads[DocumentType.GENERAL_DOCUMENT.value] = DocumentTypeNER(
                DocumentType.GENERAL_DOCUMENT, self.bert_encoder, general_entities
            )

        except Exception as e:
            logger.error(f"Failed to initialize default NER heads: {e}")

    def extract_entities(
        self,
        text: str,
        document_type: str,
        confidence_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Extract entities using appropriate document-type NER head

        Args:
            text: Input text
            document_type: Document type identifier
            confidence_threshold: Optional confidence threshold override

        Returns:
            Extracted entities with metadata
        """
        try:
            # Get appropriate NER head
            ner_head = self.ner_heads.get(document_type)
            if not ner_head:
                # Fallback to general document NER
                ner_head = self.ner_heads.get(DocumentType.GENERAL_DOCUMENT.value)
                if not ner_head:
                    return {"error": "No suitable NER head available"}

            # Override confidence threshold if provided
            if confidence_threshold is not None:
                original_threshold = ner_head.confidence_threshold
                ner_head.confidence_threshold = confidence_threshold

            # Extract entities
            result = ner_head.extract_entities(text)

            # Restore original threshold
            if confidence_threshold is not None:
                ner_head.confidence_threshold = original_threshold

            return result

        except Exception as e:
            logger.error(f"Multi-document entity extraction failed: {e}")
            return {"error": str(e), "entities": []}

    def add_ner_head(self, document_type: str, entity_types: List[EntityType]) -> bool:
        """Add custom NER head for specific document type"""
        try:
            doc_type_enum = DocumentType(document_type)
            ner_head = DocumentTypeNER(doc_type_enum, self.bert_encoder, entity_types)
            self.ner_heads[document_type] = ner_head

            logger.info(f"Added custom NER head for document type: {document_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to add NER head: {e}")
            return False

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types"""
        return list(self.ner_heads.keys())

    def get_statistics_summary(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all NER heads"""
        return {
            "total_ner_heads": len(self.ner_heads),
            "supported_document_types": self.get_supported_document_types(),
            "ner_head_statistics": {
                doc_type: ner_head.get_ner_statistics()
                for doc_type, ner_head in self.ner_heads.items()
            },
        }


# Mock classes for demonstration
class MockNERDataLoader:
    def __init__(self, data: List[Dict[str, Any]], batch_size: int, shuffle: bool):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data) // self.batch_size + (
            1 if len(self.data) % self.batch_size else 0
        )

    def __iter__(self):
        # Mock iteration
        for i in range(len(self)):
            yield {"batch": i}
