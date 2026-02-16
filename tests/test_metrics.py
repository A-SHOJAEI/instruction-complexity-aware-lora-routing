"""Tests for evaluation metrics module.

This module provides comprehensive tests for all metrics computation functions
that weren't covered in the existing test suite.
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

from src.instruction_complexity_aware_lora_routing.evaluation.metrics import (
    GenerationMetrics, ComplexityMetrics, RoutingMetrics, MetricsEvaluator
)
from src.instruction_complexity_aware_lora_routing.utils.config import EvaluationConfig


class TestGenerationMetrics:
    """Test generation quality metrics computation."""

    def test_bleu_score_computation(self):
        """Test BLEU score calculation with various inputs."""
        metrics = GenerationMetrics()

        # Perfect match case
        references = ["the quick brown fox jumps over the lazy dog"]
        predictions = ["the quick brown fox jumps over the lazy dog"]

        with patch('nltk.translate.bleu_score.sentence_bleu') as mock_bleu:
            mock_bleu.return_value = 1.0
            score = metrics.compute_bleu(predictions, references)
            assert score == 1.0
            mock_bleu.assert_called_once()

        # Partial match case
        references = ["the quick brown fox"]
        predictions = ["the quick brown cat"]

        with patch('nltk.translate.bleu_score.sentence_bleu') as mock_bleu:
            mock_bleu.return_value = 0.75
            score = metrics.compute_bleu(predictions, references)
            assert score == 0.75

        # No match case
        references = ["hello world"]
        predictions = ["goodbye universe"]

        with patch('nltk.translate.bleu_score.sentence_bleu') as mock_bleu:
            mock_bleu.return_value = 0.0
            score = metrics.compute_bleu(predictions, references)
            assert score == 0.0

    def test_rouge_score_computation(self):
        """Test ROUGE score calculation."""
        metrics = GenerationMetrics()

        references = ["the quick brown fox jumps over the lazy dog"]
        predictions = ["the quick brown fox leaps over the lazy dog"]

        # Mock ROUGE computation
        with patch('rouge_score.rouge_scorer.RougeScorer') as mock_scorer_class:
            mock_scorer = MagicMock()
            mock_score = MagicMock()
            mock_score.fmeasure = 0.8
            mock_scorer.score.return_value = {'rouge1': mock_score, 'rougeL': mock_score}
            mock_scorer_class.return_value = mock_scorer

            scores = metrics.compute_rouge(predictions, references)

            assert 'rouge1' in scores
            assert 'rougeL' in scores
            assert scores['rouge1'] == 0.8
            assert scores['rougeL'] == 0.8

    def test_perplexity_computation(self):
        """Test perplexity calculation from model outputs."""
        metrics = GenerationMetrics()

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer behavior
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.pad_token_id = 0

        # Mock model output with logits
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4], [0.2, 0.6, 0.2]]])
        mock_model.return_value = mock_outputs

        texts = ["sample text for perplexity"]

        perplexity = metrics.compute_perplexity(mock_model, mock_tokenizer, texts)

        assert isinstance(perplexity, float)
        assert perplexity > 0

    def test_generation_metrics_with_empty_inputs(self):
        """Test metrics computation handles empty inputs gracefully."""
        metrics = GenerationMetrics()

        # Empty lists should return zero
        assert metrics.compute_bleu([], []) == 0.0

        # Mismatched lengths should raise error
        with pytest.raises((ValueError, IndexError)):
            metrics.compute_bleu(["test"], [])


class TestComplexityMetrics:
    """Test complexity prediction accuracy metrics."""

    def test_complexity_mae_computation(self):
        """Test Mean Absolute Error for complexity predictions."""
        metrics = ComplexityMetrics()

        predictions = torch.tensor([0.1, 0.5, 0.9])
        targets = torch.tensor([0.2, 0.6, 0.8])

        mae = metrics.compute_complexity_mae(predictions, targets)
        expected_mae = torch.mean(torch.abs(predictions - targets))

        assert torch.isclose(mae, expected_mae)

    def test_complexity_mse_computation(self):
        """Test Mean Squared Error for complexity predictions."""
        metrics = ComplexityMetrics()

        predictions = torch.tensor([0.3, 0.7, 0.5])
        targets = torch.tensor([0.4, 0.6, 0.5])

        mse = metrics.compute_complexity_mse(predictions, targets)
        expected_mse = torch.mean((predictions - targets) ** 2)

        assert torch.isclose(mse, expected_mse)

    def test_complexity_accuracy_computation(self):
        """Test binary classification accuracy for complexity."""
        metrics = ComplexityMetrics()

        # Test with threshold 0.5
        predictions = torch.tensor([0.1, 0.6, 0.8, 0.3])
        targets = torch.tensor([0.2, 0.7, 0.9, 0.4])

        accuracy = metrics.compute_complexity_accuracy(predictions, targets, threshold=0.5)

        # Convert to binary predictions: [0, 1, 1, 0]
        # Convert to binary targets: [0, 1, 1, 0]
        # Accuracy should be 1.0 (all correct)
        assert accuracy == 1.0

    def test_complexity_correlation_computation(self):
        """Test Pearson correlation between predictions and targets."""
        metrics = ComplexityMetrics()

        # Perfect positive correlation
        predictions = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        targets = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])

        correlation = metrics.compute_complexity_correlation(predictions, targets)
        assert correlation > 0.95  # Should be close to 1.0

        # Perfect negative correlation
        predictions = torch.tensor([0.9, 0.7, 0.5, 0.3, 0.1])
        targets = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])

        correlation = metrics.compute_complexity_correlation(predictions, targets)
        assert correlation < -0.95  # Should be close to -1.0


class TestRoutingMetrics:
    """Test routing behavior analysis metrics."""

    def test_routing_entropy_computation(self):
        """Test entropy calculation for routing decisions."""
        metrics = RoutingMetrics()

        # Uniform distribution (maximum entropy)
        routing_probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        entropy = metrics.compute_routing_entropy(routing_probs)
        # log2(4) ≈ 2.0 for uniform distribution over 4 experts
        assert entropy > 1.8  # Close to maximum entropy

        # Deterministic routing (minimum entropy)
        routing_probs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        entropy = metrics.compute_routing_entropy(routing_probs)
        assert entropy < 0.1  # Should be close to 0

    def test_expert_utilization_computation(self):
        """Test expert load balancing metrics."""
        metrics = RoutingMetrics()

        # Perfectly balanced utilization
        routing_probs = torch.tensor([
            [0.5, 0.5],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.5, 0.5]
        ])

        utilization = metrics.compute_expert_utilization(routing_probs)
        assert len(utilization) == 2  # Number of experts
        assert abs(utilization[0] - utilization[1]) < 0.1  # Should be balanced

        # Unbalanced utilization
        routing_probs = torch.tensor([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.9, 0.1],
            [0.8, 0.2]
        ])

        utilization = metrics.compute_expert_utilization(routing_probs)
        assert utilization[0] > utilization[1]  # First expert should be used more

    def test_routing_consistency_computation(self):
        """Test consistency of routing decisions across similar inputs."""
        metrics = RoutingMetrics()

        # Create similar complexity scores
        complexity_scores = torch.tensor([0.5, 0.51, 0.49, 0.52, 0.48])
        routing_probs = torch.tensor([
            [0.6, 0.4],
            [0.65, 0.35],
            [0.55, 0.45],
            [0.7, 0.3],
            [0.5, 0.5]
        ])

        consistency = metrics.compute_routing_consistency(
            complexity_scores, routing_probs, complexity_threshold=0.05
        )
        assert 0.0 <= consistency <= 1.0

    def test_load_balance_loss_computation(self):
        """Test load balancing loss calculation."""
        metrics = RoutingMetrics()

        # Balanced routing should have low loss
        routing_probs = torch.tensor([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5]
        ])

        loss = metrics.compute_load_balance_loss(routing_probs)
        assert loss < 0.1  # Should be low for balanced routing

        # Unbalanced routing should have high loss
        routing_probs = torch.tensor([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ])

        loss = metrics.compute_load_balance_loss(routing_probs)
        assert loss > 0.1  # Should be high for unbalanced routing


class TestMetricsEvaluator:
    """Test the main metrics evaluation coordinator."""

    def test_evaluator_initialization(self):
        """Test proper initialization of MetricsEvaluator."""
        config = EvaluationConfig(
            batch_size=4,
            compute_generation_metrics=True,
            max_eval_samples=100
        )

        evaluator = MetricsEvaluator(config)
        assert evaluator.config == config
        assert isinstance(evaluator.generation_metrics, GenerationMetrics)
        assert isinstance(evaluator.complexity_metrics, ComplexityMetrics)
        assert isinstance(evaluator.routing_metrics, RoutingMetrics)

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation with mocked model and data."""
        config = EvaluationConfig(
            batch_size=2,
            compute_generation_metrics=False,  # Skip expensive generation metrics
            max_eval_samples=4
        )

        evaluator = MetricsEvaluator(config)

        # Mock model
        mock_model = MagicMock()
        mock_outputs = {
            'routing_probs': torch.tensor([[0.6, 0.4], [0.3, 0.7]]),
            'complexity_pred': torch.tensor([0.4, 0.8])
        }
        mock_model.return_value = mock_outputs

        # Mock evaluation data
        eval_data = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([4, 5, 6]),
                'complexity_score': 0.3
            },
            {
                'input_ids': torch.tensor([7, 8, 9]),
                'labels': torch.tensor([10, 11, 12]),
                'complexity_score': 0.7
            }
        ]

        # Run evaluation
        metrics = evaluator.evaluate(mock_model, eval_data)

        # Verify expected metrics are computed
        assert 'routing_entropy' in metrics
        assert 'expert_utilization' in metrics
        assert 'complexity_accuracy' in metrics
        assert 'complexity_mae' in metrics

        # Verify metrics have reasonable values
        assert 0.0 <= metrics['routing_entropy'] <= 2.0  # Entropy bounds
        assert isinstance(metrics['expert_utilization'], (list, np.ndarray))
        assert 0.0 <= metrics['complexity_accuracy'] <= 1.0

    def test_evaluation_with_generation_metrics(self):
        """Test evaluation including expensive generation metrics."""
        config = EvaluationConfig(
            batch_size=2,
            compute_generation_metrics=True,
            max_eval_samples=2
        )

        evaluator = MetricsEvaluator(config)

        # Mock model and tokenizer for generation
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "generated text"

        mock_outputs = {
            'routing_probs': torch.tensor([[0.7, 0.3]]),
            'complexity_pred': torch.tensor([0.5]),
            'logits': torch.tensor([[[0.1, 0.2, 0.7]]])
        }
        mock_model.return_value = mock_outputs

        eval_data = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([4, 5, 6]),
                'complexity_score': 0.4,
                'target_text': "reference text"
            }
        ]

        # Mock generation metrics computation
        with patch.object(evaluator.generation_metrics, 'compute_bleu', return_value=0.5):
            with patch.object(evaluator.generation_metrics, 'compute_rouge', return_value={'rougeL': 0.6}):
                with patch.object(evaluator.generation_metrics, 'compute_perplexity', return_value=3.2):

                    metrics = evaluator.evaluate(mock_model, eval_data, tokenizer=mock_tokenizer)

                    # Verify generation metrics are included
                    assert 'bleu' in metrics
                    assert 'rouge_l' in metrics
                    assert 'perplexity' in metrics

    def test_metrics_aggregation(self):
        """Test proper aggregation of metrics across batches."""
        config = EvaluationConfig(batch_size=1)
        evaluator = MetricsEvaluator(config)

        # Test complexity metrics aggregation
        batch_complexities = [
            torch.tensor([0.3, 0.7]),
            torch.tensor([0.4, 0.6]),
            torch.tensor([0.2, 0.8])
        ]

        batch_predictions = [
            torch.tensor([0.35, 0.65]),
            torch.tensor([0.45, 0.55]),
            torch.tensor([0.25, 0.75])
        ]

        # Aggregate metrics manually for verification
        all_targets = torch.cat(batch_complexities)
        all_preds = torch.cat(batch_predictions)

        mae = evaluator.complexity_metrics.compute_complexity_mae(all_preds, all_targets)
        accuracy = evaluator.complexity_metrics.compute_complexity_accuracy(all_preds, all_targets)

        assert isinstance(mae.item(), float)
        assert 0.0 <= accuracy <= 1.0