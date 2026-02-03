"""
Real-Time Document Intelligence & Compliance Engine
Clause-Level Embeddings Module
Version: 1.0
Date: 2026-02-03

This module generates clause-level embeddings for semantic search and similarity matching.
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class EmbeddingStrategy(Enum):
    """Strategies for generating clause embeddings"""

    CLAUSE_LEVEL = "CLAUSE_LEVEL"  # Individual clause embeddings
    SENTENCE_LEVEL = "SENTENCE_LEVEL"  # Sentence-level embeddings
    TOKEN_LEVEL = "TOKEN_LEVEL"  # Token-level embeddings
    HIERARCHICAL = "HIERARCHICAL"  # Multi-level embeddings


class PoolingStrategy(Enum):
    """Pooling strategies for embedding generation"""

    CLS_TOKEN = "CLS_TOKEN"  # Use CLS token
    MEAN_POOLING = "MEAN_POOLING"  # Average token embeddings
    MAX_POOLING = "MAX_POOLING"  # Max token embeddings
    ATTENTION_WEIGHTED = "ATTENTION_WEIGHTED"  # Attention-weighted pooling
    WEIGHTED_MEAN = "WEIGHTED_MEAN"  # TF-IDF weighted mean


class SimilarityMetric(Enum):
    """Similarity metrics for embedding comparison"""

    COSINE = "COSINE"  # Cosine similarity
    EUCLIDEAN = "EUCLIDEAN"  # Euclidean distance
    DOT_PRODUCT = "DOT_PRODUCT"  # Dot product
    JACCARD = "JACCARD"  # Jaccard similarity


class ClauseEmbeddingGenerator(nn.Module):
    """
    Generate clause-level embeddings for semantic search and similarity matching
    """

    def __init__(
        self,
        bert_encoder,  # SharedBERTEncoder instance
        embedding_dim: int = 768,
        embedding_strategy: EmbeddingStrategy = EmbeddingStrategy.CLAUSE_LEVEL,
        pooling_strategy: PoolingStrategy = PoolingStrategy.ATTENTION_WEIGHTED,
        similarity_metric: SimilarityMetric = SimilarityMetric.COSINE,
        enable_dimensionality_reduction: bool = False,
        reduced_dim: int = 256,
        cache_embeddings: bool = True,
    ):
        """
        Initialize clause embedding generator

        Args:
            bert_encoder: Shared BERT encoder instance
            embedding_dim: Dimension of embeddings
            embedding_strategy: Strategy for embedding generation
            pooling_strategy: Pooling strategy for token aggregation
            similarity_metric: Metric for similarity computation
            enable_dimensionality_reduction: Whether to reduce embedding dimensions
            reduced_dim: Target dimension for dimensionality reduction
            cache_embeddings: Whether to cache generated embeddings
        """
        super().__init__()

        self.bert_encoder = bert_encoder
        self.embedding_dim = embedding_dim
        self.embedding_strategy = embedding_strategy
        self.pooling_strategy = pooling_strategy
        self.similarity_metric = similarity_metric
        self.enable_dimensionality_reduction = enable_dimensionality_reduction
        self.reduced_dim = (
            reduced_dim if enable_dimensionality_reduction else embedding_dim
        )
        self.cache_embeddings = cache_embeddings

        # Embedding cache
        self.embedding_cache = {} if cache_embeddings else None

        # Dimensionality reduction
        if enable_dimensionality_reduction:
            self.dimension_reducer = nn.Linear(embedding_dim, reduced_dim)
        else:
            self.dimension_reducer = None

        # Attention mechanism for weighted pooling
        if pooling_strategy == PoolingStrategy.ATTENTION_WEIGHTED:
            self.attention_head = nn.MultiheadAttention(
                embed_dim=embedding_dim, num_heads=8, batch_first=True
            )

        # TF-IDF weights for weighted mean pooling
        self.tfidf_weights = {}

        # Performance metrics
        self.metrics = {
            "embeddings_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "average_generation_time": 0.0,
            "similarity_computations": 0,
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        logger.info(
            f"Clause embedding generator initialized with strategy: {embedding_strategy.value}"
        )

    def generate_clause_embeddings(
        self,
        clauses: List[Dict[str, Any]],
        return_attention_weights: bool = False,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a list of clauses

        Args:
            clauses: List of clause dictionaries
            return_attention_weights: Whether to return attention weights
            batch_size: Batch size for processing

        Returns:
            Dictionary containing embeddings and metadata
        """
        try:
            generation_start = datetime.now()

            result = {
                "embeddings": [],
                "clause_metadata": [],
                "embedding_metadata": {
                    "total_clauses": len(clauses),
                    "embedding_dim": self.reduced_dim,
                    "strategy": self.embedding_strategy.value,
                    "pooling": self.pooling_strategy.value,
                    "processing_time_ms": 0,
                },
            }

            if not clauses:
                return result

            # Process clauses in batches
            for i in range(0, len(clauses), batch_size):
                batch_clauses = clauses[i : i + batch_size]

                # Generate batch embeddings
                batch_embeddings = self._generate_batch_embeddings(
                    batch_clauses, return_attention_weights
                )

                result["embeddings"].extend(batch_embeddings["embeddings"])
                result["clause_metadata"].extend(batch_embeddings["metadata"])

                if return_attention_weights and "attention_weights" in batch_embeddings:
                    if "attention_weights" not in result:
                        result["attention_weights"] = []
                    result["attention_weights"].extend(
                        batch_embeddings["attention_weights"]
                    )

            # Calculate processing time
            processing_time = (datetime.now() - generation_start).total_seconds() * 1000
            result["embedding_metadata"]["processing_time_ms"] = processing_time

            # Update metrics
            self._update_metrics(len(clauses), processing_time)

            logger.info(
                f"Generated embeddings for {len(clauses)} clauses in {processing_time:.0f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Clause embedding generation failed: {e}")
            return {"embeddings": [], "clause_metadata": [], "error": str(e)}

    def _generate_batch_embeddings(
        self, clauses: List[Dict[str, Any]], return_attention_weights: bool = False
    ) -> Dict[str, Any]:
        """Generate embeddings for a batch of clauses"""
        try:
            batch_result = {
                "embeddings": [],
                "metadata": [],
                "attention_weights": [] if return_attention_weights else None,
            }

            # Extract clause texts
            clause_texts = []
            for clause in clauses:
                text = clause.get("generated_text") or clause.get("extracted_text", "")
                clause_texts.append(text)

            if not clause_texts:
                return batch_result

            # Check cache first
            if self.cache_embeddings:
                cached_embeddings, uncached_indices = self._check_embedding_cache(
                    clause_texts
                )
                for idx, embedding in cached_embeddings.items():
                    batch_result["embeddings"].append(embedding)
                    batch_result["metadata"].append(
                        self._create_clause_metadata(clauses[idx])
                    )
            else:
                uncached_indices = list(range(len(clause_texts)))

            # Generate embeddings for uncached clauses
            if uncached_indices:
                uncached_texts = [clause_texts[i] for i in uncached_indices]

                # Get BERT encodings
                bert_outputs = self._encode_clauses_with_bert(uncached_texts)

                # Generate embeddings based on strategy
                embeddings = self._apply_embedding_strategy(
                    bert_outputs, uncached_texts, return_attention_weights
                )

                # Apply dimensionality reduction if enabled
                if self.enable_dimensionality_reduction:
                    embeddings["embeddings"] = self._apply_dimensionality_reduction(
                        embeddings["embeddings"]
                    )

                # Cache embeddings
                if self.cache_embeddings:
                    for i, embedding in enumerate(embeddings["embeddings"]):
                        cache_key = self._get_cache_key(uncached_texts[i])
                        self.embedding_cache[cache_key] = embedding

                # Add to batch result
                for i, idx in enumerate(uncached_indices):
                    batch_result["embeddings"].append(embeddings["embeddings"][i])
                    batch_result["metadata"].append(
                        self._create_clause_metadata(clauses[idx])
                    )

                    if return_attention_weights and embeddings.get("attention_weights"):
                        batch_result["attention_weights"].append(
                            embeddings["attention_weights"][i]
                        )

            return batch_result

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return {"embeddings": [], "metadata": []}

    def _encode_clauses_with_bert(self, clause_texts: List[str]) -> Dict[str, Any]:
        """Encode clauses using BERT encoder"""
        try:
            # Use shared BERT encoder
            bert_output = self.bert_encoder.encode_batch(
                texts=clause_texts,
                return_attention_masks=True,
                return_token_type_ids=False,
            )

            return bert_output

        except Exception as e:
            logger.error(f"BERT encoding failed: {e}")
            # Return mock encoding for fallback
            return {
                "embeddings": torch.randn(len(clause_texts), 512, self.embedding_dim),
                "attention_masks": torch.ones(len(clause_texts), 512),
            }

    def _apply_embedding_strategy(
        self,
        bert_outputs: Dict[str, Any],
        clause_texts: List[str],
        return_attention_weights: bool = False,
    ) -> Dict[str, Any]:
        """Apply embedding strategy to BERT outputs"""
        try:
            embeddings = bert_outputs["embeddings"]  # [batch_size, seq_len, hidden_dim]
            attention_masks = bert_outputs.get("attention_masks")

            result = {"embeddings": []}
            if return_attention_weights:
                result["attention_weights"] = []

            if self.embedding_strategy == EmbeddingStrategy.CLAUSE_LEVEL:
                # Generate clause-level embeddings
                clause_embeddings = self._generate_clause_level_embeddings(
                    embeddings, attention_masks, return_attention_weights
                )
                result.update(clause_embeddings)

            elif self.embedding_strategy == EmbeddingStrategy.SENTENCE_LEVEL:
                # Generate sentence-level embeddings
                sentence_embeddings = self._generate_sentence_level_embeddings(
                    embeddings, clause_texts, attention_masks
                )
                result.update(sentence_embeddings)

            elif self.embedding_strategy == EmbeddingStrategy.TOKEN_LEVEL:
                # Keep token-level embeddings
                result["embeddings"] = embeddings.tolist()

            elif self.embedding_strategy == EmbeddingStrategy.HIERARCHICAL:
                # Generate hierarchical embeddings
                hierarchical_embeddings = self._generate_hierarchical_embeddings(
                    embeddings, clause_texts, attention_masks
                )
                result.update(hierarchical_embeddings)

            return result

        except Exception as e:
            logger.error(f"Embedding strategy application failed: {e}")
            return {"embeddings": []}

    def _generate_clause_level_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Dict[str, Any]:
        """Generate clause-level embeddings using pooling strategy"""
        try:
            batch_size, seq_len, hidden_dim = embeddings.shape
            result = {"embeddings": []}

            if return_attention_weights:
                result["attention_weights"] = []

            for i in range(batch_size):
                clause_embedding = embeddings[i]  # [seq_len, hidden_dim]
                mask = attention_masks[i] if attention_masks is not None else None

                if self.pooling_strategy == PoolingStrategy.CLS_TOKEN:
                    # Use CLS token (first token)
                    pooled_embedding = clause_embedding[0]

                elif self.pooling_strategy == PoolingStrategy.MEAN_POOLING:
                    # Average pooling with attention mask
                    if mask is not None:
                        masked_embeddings = clause_embedding * mask.unsqueeze(-1)
                        pooled_embedding = masked_embeddings.sum(dim=0) / mask.sum()
                    else:
                        pooled_embedding = clause_embedding.mean(dim=0)

                elif self.pooling_strategy == PoolingStrategy.MAX_POOLING:
                    # Max pooling
                    pooled_embedding, _ = clause_embedding.max(dim=0)

                elif self.pooling_strategy == PoolingStrategy.ATTENTION_WEIGHTED:
                    # Attention-weighted pooling
                    pooled_embedding, attention_weights = (
                        self._attention_weighted_pooling(
                            clause_embedding.unsqueeze(0), mask
                        )
                    )
                    pooled_embedding = pooled_embedding.squeeze(0)

                    if return_attention_weights:
                        result["attention_weights"].append(
                            attention_weights.squeeze(0).tolist()
                        )

                elif self.pooling_strategy == PoolingStrategy.WEIGHTED_MEAN:
                    # TF-IDF weighted mean
                    pooled_embedding = self._tfidf_weighted_pooling(clause_embedding, i)

                result["embeddings"].append(pooled_embedding.tolist())

            return result

        except Exception as e:
            logger.error(f"Clause-level embedding generation failed: {e}")
            return {"embeddings": []}

    def _generate_sentence_level_embeddings(
        self,
        embeddings: torch.Tensor,
        clause_texts: List[str],
        attention_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Generate sentence-level embeddings"""
        try:
            result = {"embeddings": []}

            for i, text in enumerate(clause_texts):
                # Split clause into sentences
                sentences = self._split_into_sentences(text)

                if not sentences:
                    # Fallback to clause-level embedding
                    clause_embedding = embeddings[i].mean(dim=0)
                    result["embeddings"].append([clause_embedding.tolist()])
                    continue

                sentence_embeddings = []
                clause_embedding = embeddings[i]  # [seq_len, hidden_dim]

                # Approximate sentence boundaries in token space
                tokens_per_sentence = len(clause_embedding) // len(sentences)

                for j, sentence in enumerate(sentences):
                    start_idx = j * tokens_per_sentence
                    end_idx = min((j + 1) * tokens_per_sentence, len(clause_embedding))

                    if start_idx < len(clause_embedding):
                        sentence_tokens = clause_embedding[start_idx:end_idx]
                        sentence_emb = sentence_tokens.mean(dim=0)
                        sentence_embeddings.append(sentence_emb.tolist())

                result["embeddings"].append(sentence_embeddings)

            return result

        except Exception as e:
            logger.error(f"Sentence-level embedding generation failed: {e}")
            return {"embeddings": []}

    def _generate_hierarchical_embeddings(
        self,
        embeddings: torch.Tensor,
        clause_texts: List[str],
        attention_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Generate hierarchical embeddings (token -> sentence -> clause)"""
        try:
            result = {"embeddings": []}

            for i, text in enumerate(clause_texts):
                clause_embedding = embeddings[i]

                # Generate multi-level embeddings
                hierarchical_embedding = {
                    "token_level": clause_embedding.tolist(),
                    "sentence_level": [],
                    "clause_level": None,
                }

                # Sentence-level aggregation
                sentences = self._split_into_sentences(text)
                if sentences:
                    tokens_per_sentence = len(clause_embedding) // len(sentences)

                    for j in range(len(sentences)):
                        start_idx = j * tokens_per_sentence
                        end_idx = min(
                            (j + 1) * tokens_per_sentence, len(clause_embedding)
                        )

                        if start_idx < len(clause_embedding):
                            sentence_tokens = clause_embedding[start_idx:end_idx]
                            sentence_emb = sentence_tokens.mean(dim=0)
                            hierarchical_embedding["sentence_level"].append(
                                sentence_emb.tolist()
                            )

                # Clause-level aggregation
                clause_emb = clause_embedding.mean(dim=0)
                hierarchical_embedding["clause_level"] = clause_emb.tolist()

                result["embeddings"].append(hierarchical_embedding)

            return result

        except Exception as e:
            logger.error(f"Hierarchical embedding generation failed: {e}")
            return {"embeddings": []}

    def _attention_weighted_pooling(
        self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention-weighted pooling"""
        try:
            if mask is not None:
                # Apply mask to embeddings
                embeddings = embeddings * mask.unsqueeze(-1)

            # Self-attention for weighting
            attended_output, attention_weights = self.attention_head(
                embeddings,
                embeddings,
                embeddings,
                key_padding_mask=mask is None if mask is not None else None,
            )

            # Weighted average
            if mask is not None:
                attention_weights = attention_weights * mask.unsqueeze(1)
                weight_sum = attention_weights.sum(dim=-1, keepdim=True)
                weight_sum = (
                    weight_sum + (weight_sum == 0).float() * 1e-9
                )  # Avoid division by zero
                attention_weights = attention_weights / weight_sum

            weighted_embedding = (
                attended_output * attention_weights.unsqueeze(-1)
            ).sum(dim=1)

            return weighted_embedding, attention_weights.mean(dim=1)

        except Exception as e:
            logger.error(f"Attention-weighted pooling failed: {e}")
            # Fallback to mean pooling
            if mask is not None:
                masked_embeddings = embeddings * mask.unsqueeze(-1)
                pooled = masked_embeddings.sum(dim=1) / mask.sum(dim=1, keepdim=True)
            else:
                pooled = embeddings.mean(dim=1)

            return pooled, torch.ones(embeddings.shape[:2])

    def _tfidf_weighted_pooling(
        self, embedding: torch.Tensor, clause_index: int
    ) -> torch.Tensor:
        """Apply TF-IDF weighted pooling (simplified implementation)"""
        try:
            # Simplified TF-IDF weighting
            # In a real implementation, you would compute actual TF-IDF weights

            weights = torch.ones(len(embedding))

            # Apply decreasing weights from beginning to end (approximate importance)
            for i in range(len(embedding)):
                weights[i] = 1.0 / (1.0 + i * 0.1)

            # Normalize weights
            weights = weights / weights.sum()

            # Apply weights
            weighted_embedding = (embedding * weights.unsqueeze(-1)).sum(dim=0)

            return weighted_embedding

        except Exception:
            # Fallback to mean pooling
            return embedding.mean(dim=0)

    def _apply_dimensionality_reduction(
        self, embeddings: List[List[float]]
    ) -> List[List[float]]:
        """Apply dimensionality reduction to embeddings"""
        try:
            if self.dimension_reducer is None:
                return embeddings

            # Convert to tensor
            embeddings_tensor = torch.tensor(embeddings).to(self.device)

            # Apply dimensionality reduction
            with torch.no_grad():
                reduced_embeddings = self.dimension_reducer(embeddings_tensor)

            return reduced_embeddings.cpu().tolist()

        except Exception as e:
            logger.warning(f"Dimensionality reduction failed: {e}")
            return embeddings

    def compute_clause_similarities(
        self,
        embeddings1: List[List[float]],
        embeddings2: Optional[List[List[float]]] = None,
        similarity_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Compute similarities between clause embeddings

        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings (if None, compute pairwise for embeddings1)
            similarity_threshold: Threshold for similarity matches

        Returns:
            Similarity matrix and matches
        """
        try:
            if not embeddings1:
                return {"similarity_matrix": [], "matches": []}

            if embeddings2 is None:
                embeddings2 = embeddings1

            # Convert to tensors
            emb1_tensor = torch.tensor(embeddings1)
            emb2_tensor = torch.tensor(embeddings2)

            # Compute similarity matrix
            if self.similarity_metric == SimilarityMetric.COSINE:
                similarity_matrix = self._cosine_similarity_matrix(
                    emb1_tensor, emb2_tensor
                )
            elif self.similarity_metric == SimilarityMetric.EUCLIDEAN:
                similarity_matrix = self._euclidean_similarity_matrix(
                    emb1_tensor, emb2_tensor
                )
            elif self.similarity_metric == SimilarityMetric.DOT_PRODUCT:
                similarity_matrix = torch.matmul(emb1_tensor, emb2_tensor.t())
            else:
                similarity_matrix = self._cosine_similarity_matrix(
                    emb1_tensor, emb2_tensor
                )

            # Find matches above threshold
            matches = []
            for i in range(similarity_matrix.shape[0]):
                for j in range(similarity_matrix.shape[1]):
                    if i != j or embeddings2 is not embeddings1:  # Avoid self-matches
                        similarity_score = similarity_matrix[i, j].item()
                        if similarity_score >= similarity_threshold:
                            matches.append(
                                {
                                    "index1": i,
                                    "index2": j,
                                    "similarity_score": similarity_score,
                                }
                            )

            # Sort matches by similarity score
            matches.sort(key=lambda x: x["similarity_score"], reverse=True)

            # Update metrics
            self.metrics["similarity_computations"] += len(embeddings1) * len(
                embeddings2
            )

            return {
                "similarity_matrix": similarity_matrix.tolist(),
                "matches": matches,
                "similarity_metric": self.similarity_metric.value,
                "threshold": similarity_threshold,
            }

        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return {"similarity_matrix": [], "matches": [], "error": str(e)}

    def _cosine_similarity_matrix(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity matrix"""
        # Normalize embeddings
        emb1_norm = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        emb2_norm = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarity
        return torch.matmul(emb1_norm, emb2_norm.t())

    def _euclidean_similarity_matrix(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Euclidean distance-based similarity matrix"""
        # Compute pairwise distances
        distances = torch.cdist(embeddings1, embeddings2, p=2)

        # Convert to similarities (higher is more similar)
        max_distance = distances.max()
        similarities = 1.0 - (distances / (max_distance + 1e-8))

        return similarities

    def find_similar_clauses(
        self,
        query_clause: str,
        candidate_clauses: List[Dict[str, Any]],
        top_k: int = 5,
        similarity_threshold: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Find clauses similar to a query clause

        Args:
            query_clause: Query clause text
            candidate_clauses: List of candidate clauses
            top_k: Number of top matches to return
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar clauses with similarity scores
        """
        try:
            if not candidate_clauses:
                return []

            # Generate embedding for query clause
            query_embedding = self.generate_clause_embeddings(
                [{"generated_text": query_clause}]
            )

            if not query_embedding["embeddings"]:
                return []

            # Generate embeddings for candidate clauses
            candidate_embeddings = self.generate_clause_embeddings(candidate_clauses)

            if not candidate_embeddings["embeddings"]:
                return []

            # Compute similarities
            similarities = self.compute_clause_similarities(
                query_embedding["embeddings"],
                candidate_embeddings["embeddings"],
                similarity_threshold,
            )

            # Format results
            similar_clauses = []
            for match in similarities["matches"]:
                candidate_idx = match["index2"]
                if candidate_idx < len(candidate_clauses):
                    similar_clause = candidate_clauses[candidate_idx].copy()
                    similar_clause["similarity_score"] = match["similarity_score"]
                    similar_clauses.append(similar_clause)

            # Return top-k matches
            return similar_clauses[:top_k]

        except Exception as e:
            logger.error(f"Similar clause search failed: {e}")
            return []

    def _check_embedding_cache(
        self, texts: List[str]
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """Check embedding cache for cached embeddings"""
        cached_embeddings = {}
        uncached_indices = []

        if not self.cache_embeddings:
            return cached_embeddings, list(range(len(texts)))

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.embedding_cache:
                cached_embeddings[i] = self.embedding_cache[cache_key]
                self.metrics["cache_hits"] += 1
            else:
                uncached_indices.append(i)
                self.metrics["cache_misses"] += 1

        return cached_embeddings, uncached_indices

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Simple hash-based key
        return f"{hash(text.strip().lower())}_{len(text)}"

    def _create_clause_metadata(self, clause: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for clause embedding"""
        return {
            "clause_type": clause.get("clause_type", "UNKNOWN"),
            "text_length": len(
                clause.get("generated_text", clause.get("extracted_text", ""))
            ),
            "relevance_level": clause.get("relevance_level", "MEDIUM"),
            "embedding_strategy": self.embedding_strategy.value,
            "pooling_strategy": self.pooling_strategy.value,
            "embedding_dimension": self.reduced_dim,
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _update_metrics(self, num_clauses: int, processing_time: float):
        """Update performance metrics"""
        try:
            self.metrics["embeddings_generated"] += num_clauses

            # Update average processing time
            current_avg = self.metrics["average_generation_time"]
            total_embeddings = self.metrics["embeddings_generated"]
            new_avg = (
                (current_avg * (total_embeddings - num_clauses)) + processing_time
            ) / total_embeddings
            self.metrics["average_generation_time"] = new_avg

        except Exception:
            pass

    def save_embeddings(self, embeddings_data: Dict[str, Any], file_path: str):
        """Save embeddings to file"""
        try:
            # Convert tensors to lists for serialization
            serializable_data = {}
            for key, value in embeddings_data.items():
                if isinstance(value, torch.Tensor):
                    serializable_data[key] = value.tolist()
                else:
                    serializable_data[key] = value

            with open(file_path, "w") as f:
                json.dump(serializable_data, f, indent=2)

            logger.info(f"Embeddings saved to {file_path}")

        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")

    def load_embeddings(self, file_path: str) -> Dict[str, Any]:
        """Load embeddings from file"""
        try:
            with open(file_path, "r") as f:
                embeddings_data = json.load(f)

            logger.info(f"Embeddings loaded from {file_path}")
            return embeddings_data

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return {}

    def get_embedding_info(self) -> Dict[str, Any]:
        """Get comprehensive embedding generator information"""
        return {
            "configuration": {
                "embedding_dim": self.embedding_dim,
                "reduced_dim": self.reduced_dim,
                "embedding_strategy": self.embedding_strategy.value,
                "pooling_strategy": self.pooling_strategy.value,
                "similarity_metric": self.similarity_metric.value,
                "dimensionality_reduction": self.enable_dimensionality_reduction,
                "caching_enabled": self.cache_embeddings,
            },
            "cache_status": {
                "cache_size": len(self.embedding_cache) if self.embedding_cache else 0,
                "cache_hits": self.metrics["cache_hits"],
                "cache_misses": self.metrics["cache_misses"],
                "hit_rate": (
                    self.metrics["cache_hits"]
                    / max(1, self.metrics["cache_hits"] + self.metrics["cache_misses"])
                ),
            },
            "performance_metrics": self.metrics,
            "device": str(self.device),
            "model_components": {
                "bert_encoder": self.bert_encoder.__class__.__name__,
                "dimension_reducer": self.dimension_reducer is not None,
                "attention_head": self.pooling_strategy
                == PoolingStrategy.ATTENTION_WEIGHTED,
            },
        }
