"""
Real-Time Document Intelligence & Compliance Engine
Shared BERT Encoder Module
Version: 1.0
Date: 2026-02-03

This module provides a shared BERT encoder for document understanding with
configurable architectures, fine-tuning capabilities, and support for multiple variants.
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class BERTVariant(Enum):
    """Supported BERT model variants"""

    BERT_BASE = "bert-base-uncased"
    BERT_LARGE = "bert-large-uncased"
    BERT_BASE_CASED = "bert-base-cased"
    DISTILBERT = "distilbert-base-uncased"
    ROBERTA_BASE = "roberta-base"
    ROBERTA_LARGE = "roberta-large"
    LEGAL_BERT = "nlpaueb/legal-bert-base-uncased"
    FINANCIAL_BERT = "ProsusAI/finbert"
    CLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"


class PoolingStrategy(Enum):
    """Token pooling strategies for sentence embeddings"""

    CLS_TOKEN = "CLS_TOKEN"  # Use [CLS] token
    MEAN_POOLING = "MEAN_POOLING"  # Mean of all tokens
    MAX_POOLING = "MAX_POOLING"  # Max pooling
    MEAN_MAX = "MEAN_MAX"  # Concatenate mean and max
    ATTENTION_POOLING = "ATTENTION_POOLING"  # Attention-weighted pooling


class EncodingMode(Enum):
    """Encoding operation modes"""

    INFERENCE = "INFERENCE"  # Inference only
    FINE_TUNING = "FINE_TUNING"  # Fine-tuning mode
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"  # Feature extraction


class SharedBERTEncoder(nn.Module):
    """
    Shared BERT encoder with configurable architecture and fine-tuning capabilities
    """

    def __init__(
        self,
        model_variant: BERTVariant = BERTVariant.BERT_BASE,
        pooling_strategy: PoolingStrategy = PoolingStrategy.CLS_TOKEN,
        max_sequence_length: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_dropout_prob: float = 0.1,
        num_hidden_layers: Optional[int] = None,
        use_pretrained: bool = True,
        cache_dir: str = "./model_cache",
    ):
        """
        Initialize shared BERT encoder

        Args:
            model_variant: BERT model variant to use
            pooling_strategy: Strategy for pooling token embeddings
            max_sequence_length: Maximum sequence length
            hidden_dropout_prob: Hidden layer dropout probability
            attention_dropout_prob: Attention dropout probability
            num_hidden_layers: Number of hidden layers (None for default)
            use_pretrained: Whether to use pretrained weights
            cache_dir: Directory for model caching
        """
        super().__init__()

        self.model_variant = model_variant
        self.pooling_strategy = pooling_strategy
        self.max_sequence_length = max_sequence_length
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model configuration
        self.config = {
            "model_variant": model_variant.value,
            "pooling_strategy": pooling_strategy.value,
            "max_sequence_length": max_sequence_length,
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_dropout_prob": attention_dropout_prob,
            "num_hidden_layers": num_hidden_layers,
            "use_pretrained": use_pretrained,
        }

        # Initialize tokenizer and model
        self.tokenizer = None
        self.bert_model = None
        self.pooler = None

        # Model state
        self.is_initialized = False
        self.is_fine_tuned = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Performance metrics
        self.metrics = {
            "total_encodings": 0,
            "average_encoding_time": 0.0,
            "cache_hits": 0,
            "fine_tuning_epochs": 0,
        }

        # Initialize model components
        self._initialize_model()

        logger.info(f"Initialized BERT encoder with variant: {model_variant.value}")

    def _initialize_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            # Mock initialization - in real implementation would load actual models
            # from transformers import AutoTokenizer, AutoModel
            # self.tokenizer = AutoTokenizer.from_pretrained(self.model_variant.value)
            # self.bert_model = AutoModel.from_pretrained(self.model_variant.value)

            # Mock tokenizer and model for demonstration
            self.tokenizer = MockTokenizer(self.model_variant.value)
            self.bert_model = MockBERTModel(
                variant=self.model_variant.value,
                hidden_size=768 if "base" in self.model_variant.value else 1024,
                num_hidden_layers=self.config.get("num_hidden_layers", 12),
                max_position_embeddings=self.max_sequence_length,
            )

            # Initialize pooling layer
            self.pooler = self._create_pooler()

            # Move to device
            self.bert_model = self.bert_model.to(self.device)
            if self.pooler:
                self.pooler = self.pooler.to(self.device)

            self.is_initialized = True
            logger.info(f"BERT model initialized on device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize BERT model: {e}")
            raise

    def _create_pooler(self) -> Optional[nn.Module]:
        """Create pooling layer based on strategy"""
        try:
            if self.pooling_strategy == PoolingStrategy.CLS_TOKEN:
                return None  # Use CLS token directly

            elif self.pooling_strategy == PoolingStrategy.ATTENTION_POOLING:
                hidden_size = 768 if "base" in self.model_variant.value else 1024
                return AttentionPooler(hidden_size)

            else:
                return None  # Use functional pooling

        except Exception as e:
            logger.error(f"Failed to create pooler: {e}")
            return None

    def encode_text(
        self,
        texts: Union[str, List[str]],
        mode: EncodingMode = EncodingMode.INFERENCE,
        return_attention_mask: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Encode text using BERT encoder

        Args:
            texts: Text or list of texts to encode
            mode: Encoding mode
            return_attention_mask: Whether to return attention masks
            batch_size: Batch size for processing

        Returns:
            Encoding results with embeddings and metadata
        """
        try:
            start_time = datetime.now()

            # Ensure input is list
            if isinstance(texts, str):
                texts = [texts]

            if not texts:
                return {
                    "embeddings": [],
                    "attention_masks": [],
                    "error": "No texts provided",
                }

            # Tokenize texts
            tokenized = self._tokenize_texts(texts)

            # Process in batches
            all_embeddings = []
            all_attention_masks = []

            for i in range(0, len(texts), batch_size):
                # batch_texts = texts[i:i+batch_size]  # Reserved for future use
                batch_tokenized = {
                    key: value[i : i + batch_size] for key, value in tokenized.items()
                }

                # Encode batch
                batch_result = self._encode_batch(batch_tokenized, mode)
                all_embeddings.extend(batch_result["embeddings"])

                if return_attention_mask:
                    all_attention_masks.extend(batch_result["attention_masks"])

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_metrics(len(texts), processing_time)

            result = {
                "embeddings": all_embeddings,
                "num_texts": len(texts),
                "embedding_dim": len(all_embeddings[0]) if all_embeddings else 0,
                "processing_time_ms": processing_time * 1000,
                "model_variant": self.model_variant.value,
                "pooling_strategy": self.pooling_strategy.value,
            }

            if return_attention_mask:
                result["attention_masks"] = all_attention_masks

            logger.debug(f"Encoded {len(texts)} texts in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            return {"error": str(e), "embeddings": [], "num_texts": 0}

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenize texts for BERT processing"""
        try:
            tokenized_texts = []
            attention_masks = []

            for text in texts:
                # Clean and truncate text
                cleaned_text = self._preprocess_text(text)

                # Tokenize (mock implementation)
                tokens = self.tokenizer.tokenize(
                    cleaned_text, max_length=self.max_sequence_length
                )
                tokenized_texts.append(tokens["input_ids"])
                attention_masks.append(tokens["attention_mask"])

            return {"input_ids": tokenized_texts, "attention_mask": attention_masks}

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return {"input_ids": [], "attention_mask": []}

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for BERT encoding"""
        try:
            # Basic preprocessing
            text = text.strip()

            # Remove excessive whitespace
            text = " ".join(text.split())

            # Truncate if too long (rough character-based truncation)
            max_chars = self.max_sequence_length * 4  # Rough estimate
            if len(text) > max_chars:
                text = text[:max_chars]

            return text

        except Exception:
            return ""

    def _encode_batch(
        self, batch_tokenized: Dict[str, List[List[int]]], mode: EncodingMode
    ) -> Dict[str, Any]:
        """Encode a batch of tokenized texts"""
        try:
            # Convert to tensors
            input_ids = torch.tensor(batch_tokenized["input_ids"]).to(self.device)
            attention_mask = torch.tensor(batch_tokenized["attention_mask"]).to(
                self.device
            )

            # Set model mode
            if mode == EncodingMode.INFERENCE:
                self.bert_model.eval()
                torch.set_grad_enabled(False)
            else:
                self.bert_model.train()
                torch.set_grad_enabled(True)

            # Forward pass
            with (
                torch.no_grad()
                if mode == EncodingMode.INFERENCE
                else torch.enable_grad()
            ):
                outputs = self.bert_model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                last_hidden_state = outputs[
                    "last_hidden_state"
                ]  # [batch_size, seq_len, hidden_size]

                # Apply pooling
                embeddings = self._apply_pooling(last_hidden_state, attention_mask)

            # Convert to numpy and return
            embeddings_np = embeddings.cpu().numpy().tolist()
            attention_masks_np = attention_mask.cpu().numpy().tolist()

            return {"embeddings": embeddings_np, "attention_masks": attention_masks_np}

        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return {"embeddings": [], "attention_masks": []}

    def _apply_pooling(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply pooling strategy to hidden states"""
        try:
            if self.pooling_strategy == PoolingStrategy.CLS_TOKEN:
                return hidden_states[:, 0, :]  # Use [CLS] token

            elif self.pooling_strategy == PoolingStrategy.MEAN_POOLING:
                # Mask and mean pool
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                sum_hidden = masked_hidden.sum(dim=1)
                lengths = attention_mask.sum(dim=1, keepdim=True).float()
                return sum_hidden / lengths

            elif self.pooling_strategy == PoolingStrategy.MAX_POOLING:
                # Max pooling
                masked_hidden = hidden_states.clone()
                masked_hidden[attention_mask == 0] = float("-inf")
                return masked_hidden.max(dim=1)[0]

            elif self.pooling_strategy == PoolingStrategy.MEAN_MAX:
                # Concatenate mean and max pooling
                mean_pooled = self._apply_pooling(
                    hidden_states, attention_mask
                )  # This will recursively call mean pooling
                max_pooled = self._apply_pooling(
                    hidden_states, attention_mask
                )  # This will recursively call max pooling
                return torch.cat([mean_pooled, max_pooled], dim=-1)

            elif self.pooling_strategy == PoolingStrategy.ATTENTION_POOLING:
                if self.pooler:
                    return self.pooler(hidden_states, attention_mask)
                else:
                    # Fallback to CLS token
                    return hidden_states[:, 0, :]

            else:
                # Default to CLS token
                return hidden_states[:, 0, :]

        except Exception as e:
            logger.error(f"Pooling failed: {e}")
            return hidden_states[:, 0, :]  # Fallback to CLS

    def fine_tune(
        self,
        train_texts: List[str],
        train_labels: List[Any],
        validation_texts: Optional[List[str]] = None,
        validation_labels: Optional[List[Any]] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        warmup_steps: int = 100,
    ) -> Dict[str, Any]:
        """
        Fine-tune BERT encoder on task-specific data

        Args:
            train_texts: Training texts
            train_labels: Training labels
            validation_texts: Validation texts
            validation_labels: Validation labels
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            warmup_steps: Warmup steps

        Returns:
            Fine-tuning results and metrics
        """
        try:
            logger.info(f"Starting fine-tuning for {num_epochs} epochs")

            # Prepare data loaders (mock implementation)
            train_loader = self._create_data_loader(
                train_texts, train_labels, batch_size, shuffle=True
            )
            val_loader = None
            if validation_texts:
                val_loader = self._create_data_loader(
                    validation_texts, validation_labels, batch_size, shuffle=False
                )

            # Setup optimizer and scheduler
            optimizer = self._create_optimizer(learning_rate)
            scheduler = self._create_scheduler(
                optimizer, len(train_loader) * num_epochs, warmup_steps
            )

            # Training loop
            training_history = {
                "train_losses": [],
                "val_losses": [],
                "learning_rates": [],
                "epochs_completed": 0,
            }

            for epoch in range(num_epochs):
                # Training phase
                train_loss = self._train_epoch(train_loader, optimizer, scheduler)
                training_history["train_losses"].append(train_loss)
                training_history["learning_rates"].append(
                    scheduler.get_last_lr()[0] if scheduler else learning_rate
                )

                # Validation phase
                if val_loader:
                    val_loss = self._validate_epoch(val_loader)
                    training_history["val_losses"].append(val_loss)

                training_history["epochs_completed"] = epoch + 1
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}"
                )

            self.is_fine_tuned = True
            self.metrics["fine_tuning_epochs"] += num_epochs

            logger.info("Fine-tuning completed successfully")

            return {
                "status": "success",
                "training_history": training_history,
                "final_train_loss": training_history["train_losses"][-1],
                "final_val_loss": training_history["val_losses"][-1]
                if training_history["val_losses"]
                else None,
                "epochs_completed": num_epochs,
                "model_size": self._get_model_size(),
            }

        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _create_data_loader(
        self, texts: List[str], labels: List[Any], batch_size: int, shuffle: bool
    ) -> DataLoader:
        """Create PyTorch data loader (mock implementation)"""
        # In real implementation, would create proper Dataset and DataLoader
        return MockDataLoader(texts, labels, batch_size, shuffle)

    def _create_optimizer(self, learning_rate: float):
        """Create optimizer for fine-tuning"""
        # In real implementation: return torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return MockOptimizer(learning_rate)

    def _create_scheduler(self, optimizer, total_steps: int, warmup_steps: int):
        """Create learning rate scheduler"""
        # In real implementation: return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return MockScheduler(optimizer, warmup_steps, total_steps)

    def _train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """Train for one epoch"""
        self.bert_model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch in train_loader:
            # Mock training step
            loss = torch.tensor(0.5 + torch.randn(1).item() * 0.1)  # Simulated loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch"""
        self.bert_model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for batch in val_loader:
                # Mock validation step
                loss = torch.tensor(
                    0.4 + torch.randn(1).item() * 0.08
                )  # Simulated loss
                total_loss += loss.item()

        return total_loss / num_batches

    def save_model(self, save_path: str, include_tokenizer: bool = True) -> bool:
        """Save model and tokenizer to disk"""
        try:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save model configuration
            config_path = save_path / "config.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)

            # Save model state (mock implementation)
            # In real implementation: torch.save(self.state_dict(), save_path / "pytorch_model.bin")
            model_path = save_path / "pytorch_model.bin"
            torch.save({"mock_state": "saved"}, model_path)

            # Save tokenizer (mock implementation)
            if include_tokenizer:
                # In real implementation: self.tokenizer.save_pretrained(save_path)
                tokenizer_path = save_path / "tokenizer_config.json"
                with open(tokenizer_path, "w") as f:
                    json.dump({"tokenizer_type": self.model_variant.value}, f)

            # Save metrics
            metrics_path = save_path / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=2)

            logger.info(f"Model saved successfully to {save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, load_path: str) -> bool:
        """Load model and tokenizer from disk"""
        try:
            load_path = Path(load_path)

            # Load configuration
            config_path = load_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                self.config.update(loaded_config)

            # Load model state (mock implementation)
            model_path = load_path / "pytorch_model.bin"
            if model_path.exists():
                # In real implementation: self.load_state_dict(torch.load(model_path))
                # loaded_state = torch.load(model_path)  # Reserved for actual implementation
                logger.info("Model state loaded")

            # Load metrics
            metrics_path = load_path / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    loaded_metrics = json.load(f)
                self.metrics.update(loaded_metrics)

            logger.info(f"Model loaded successfully from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        try:
            # In real implementation: calculate actual parameter counts
            total_params = (
                110_000_000 if "base" in self.model_variant.value else 340_000_000
            )
            trainable_params = total_params if self.is_fine_tuned else 0

            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "memory_mb": total_params * 4 // (1024 * 1024),  # Rough estimate
            }
        except Exception:
            return {"total_parameters": 0, "trainable_parameters": 0, "memory_mb": 0}

    def _update_metrics(self, num_texts: int, processing_time: float):
        """Update performance metrics"""
        try:
            self.metrics["total_encodings"] += num_texts

            # Update average encoding time
            current_avg = self.metrics["average_encoding_time"]
            total_encodings = self.metrics["total_encodings"]
            new_avg = (
                (current_avg * (total_encodings - num_texts)) + processing_time
            ) / total_encodings
            self.metrics["average_encoding_time"] = new_avg

        except Exception:
            pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "model_variant": self.model_variant.value,
            "pooling_strategy": self.pooling_strategy.value,
            "configuration": self.config,
            "model_size": self._get_model_size(),
            "performance_metrics": self.metrics,
            "status": {
                "is_initialized": self.is_initialized,
                "is_fine_tuned": self.is_fine_tuned,
                "device": str(self.device),
            },
            "capabilities": {
                "max_sequence_length": self.max_sequence_length,
                "embedding_dimension": 768
                if "base" in self.model_variant.value
                else 1024,
                "supports_fine_tuning": True,
                "supports_feature_extraction": True,
            },
        }


class AttentionPooler(nn.Module):
    """Attention-based pooling layer"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Calculate attention weights
        attention_weights = self.attention(hidden_states).squeeze(-1)
        attention_weights = attention_weights.masked_fill(
            attention_mask == 0, float("-inf")
        )
        attention_weights = torch.softmax(attention_weights, dim=-1)

        # Apply attention weights
        pooled_output = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1), dim=1
        )
        return self.dropout(pooled_output)


# Mock classes for demonstration (replace with actual implementations)
class MockTokenizer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def tokenize(self, text: str, max_length: int = 512) -> Dict[str, List[int]]:
        # Mock tokenization - in real implementation use actual tokenizer
        words = text.split()[: max_length - 2]  # Account for [CLS] and [SEP]
        input_ids = (
            [101] + list(range(2000, 2000 + len(words))) + [102]
        )  # Mock token IDs
        attention_mask = [1] * len(input_ids)

        # Pad to max_length
        while len(input_ids) < max_length:
            input_ids.append(0)
            attention_mask.append(0)

        return {
            "input_ids": input_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
        }


class MockBERTModel(nn.Module):
    def __init__(
        self,
        variant: str,
        hidden_size: int,
        num_hidden_layers: int,
        max_position_embeddings: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings

        # Mock embedding and transformer layers
        self.embeddings = nn.Embedding(30522, hidden_size)  # BERT vocab size
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=hidden_size // 64,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                )
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Mock forward pass
        batch_size, seq_len = input_ids.shape

        # Embedding
        embeddings = self.embeddings(input_ids)

        # Transformer layers (simplified)
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states)

        return {"last_hidden_state": hidden_states}


class MockDataLoader:
    def __init__(
        self, texts: List[str], labels: List[Any], batch_size: int, shuffle: bool
    ):
        self.texts = texts
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.texts) // self.batch_size + (
            1 if len(self.texts) % self.batch_size else 0
        )

    def __iter__(self):
        indices = list(range(len(self.texts)))
        if self.shuffle:
            import random

            random.shuffle(indices)

        for i in range(0, len(self.texts), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch_texts = [self.texts[j] for j in batch_indices]
            batch_labels = [self.labels[j] for j in batch_indices]
            yield {"texts": batch_texts, "labels": batch_labels}


class MockOptimizer:
    def __init__(self, learning_rate: float):
        self.lr = learning_rate

    def zero_grad(self):
        pass

    def step(self):
        pass


class MockScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1

    def get_last_lr(self):
        return [self.optimizer.lr]
