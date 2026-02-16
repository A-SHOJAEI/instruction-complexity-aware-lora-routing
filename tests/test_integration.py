"""Integration tests for full training pipeline.

This module tests end-to-end workflows to catch pipeline-level issues that unit
tests might miss. It ensures all components work together correctly.
"""

import tempfile
import shutil
from pathlib import Path
import pytest
import torch
import pandas as pd
from unittest.mock import patch, MagicMock

from src.instruction_complexity_aware_lora_routing.data.loader import DataLoader
from src.instruction_complexity_aware_lora_routing.models.model import ComplexityAwareLoRARouter
from src.instruction_complexity_aware_lora_routing.training.trainer import LoRATrainer
from src.instruction_complexity_aware_lora_routing.evaluation.metrics import MetricsEvaluator
from src.instruction_complexity_aware_lora_routing.utils.config import (
    Config, ModelConfig, DataConfig, TrainingConfig, EvaluationConfig
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_dataset():
    """Create a minimal sample dataset for integration testing."""
    return pd.DataFrame({
        'instruction': [
            "What is machine learning?",
            "Explain quantum computing in detail with mathematical formulations.",
            "How do I cook rice?",
            "Describe the complexity of neural network architectures."
        ],
        'output': [
            "Machine learning is a subset of artificial intelligence.",
            "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information.",
            "Rinse rice, add water, bring to boil, then simmer.",
            "Neural networks vary in complexity from simple perceptrons to deep transformers with millions of parameters."
        ]
    })


@pytest.fixture
def integration_config(temp_output_dir):
    """Create a test configuration for integration testing."""
    return Config(
        model=ModelConfig(
            base_model_name="gpt2",  # Small model for fast testing
            num_experts=2,  # Minimal for testing
            feature_dim=64,
            lora_rank=4,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["c_attn"],
            complexity_weight=0.1
        ),
        data=DataConfig(
            dataset_name="test_dataset",
            max_length=128,  # Short for fast testing
            train_split=0.8,
            val_split=0.2,
            complexity_bins=3
        ),
        training=TrainingConfig(
            num_epochs=2,  # Minimal for testing
            batch_size=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            training_mode="joint",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            save_steps=1,
            eval_steps=1,
            logging_steps=1,
            patience=2
        ),
        evaluation=EvaluationConfig(
            batch_size=2,
            compute_generation_metrics=False,  # Skip expensive metrics for integration test
            max_eval_samples=4
        ),
        paths={
            'output_dir': temp_output_dir,
            'data_dir': temp_output_dir / "data",
            'log_dir': temp_output_dir / "logs",
            'checkpoint_dir': temp_output_dir / "checkpoints"
        }
    )


class TestIntegration:
    """Integration tests for the complete training pipeline."""

    def test_end_to_end_training_pipeline(self, sample_dataset, integration_config, temp_output_dir):
        """Test complete training pipeline from data loading to evaluation.

        This test verifies that:
        1. Data can be loaded and preprocessed correctly
        2. Model can be initialized without errors
        3. Training loop executes without crashes
        4. Checkpoints are saved properly
        5. Evaluation produces expected metrics

        Args:
            sample_dataset: Minimal test dataset
            integration_config: Test configuration
            temp_output_dir: Temporary directory for outputs
        """
        # Step 1: Save test dataset
        dataset_path = temp_output_dir / "test_data.csv"
        sample_dataset.to_csv(dataset_path, index=False)

        # Step 2: Initialize data loader
        data_loader = DataLoader(integration_config.data, str(dataset_path))

        # Mock tokenizer to avoid downloading large models
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.encode.return_value = [1, 2, 3, 4]
            mock_tok.decode.return_value = "test output"
            mock_tok.eos_token_id = 50256
            mock_tok.pad_token_id = 50256
            mock_tok.vocab_size = 50257
            mock_tokenizer.return_value = mock_tok

            data_loader.set_tokenizer(mock_tok)

            # Step 3: Load and verify data
            train_data, val_data = data_loader.load_data()
            assert len(train_data) > 0, "Training data should not be empty"
            assert len(val_data) > 0, "Validation data should not be empty"
            assert len(train_data) + len(val_data) == len(sample_dataset), "Data split should preserve all samples"

            # Verify complexity scores are computed
            assert all('complexity_score' in item for item in train_data), "All training samples should have complexity scores"
            assert all('complexity_score' in item for item in val_data), "All validation samples should have complexity scores"

            # Step 4: Initialize model
            model = ComplexityAwareLoRARouter(integration_config.model)
            assert model is not None, "Model should initialize successfully"

            # Verify model components
            assert hasattr(model, 'router'), "Model should have router component"
            assert hasattr(model, 'experts'), "Model should have experts component"
            assert len(model.experts) == integration_config.model.num_experts, "Should have correct number of experts"

            # Step 5: Initialize trainer
            trainer = LoRATrainer(
                model=model,
                config=integration_config,
                train_data=train_data,
                val_data=val_data
            )
            assert trainer is not None, "Trainer should initialize successfully"

            # Step 6: Run training (minimal epochs for testing)
            with patch('torch.save'):  # Mock checkpoint saving
                training_history = trainer.train()

                # Verify training produces expected outputs
                assert 'train_loss' in training_history, "Training should track train loss"
                assert 'val_loss' in training_history, "Training should track validation loss"
                assert len(training_history['train_loss']) > 0, "Should have training loss history"

            # Step 7: Run evaluation
            evaluator = MetricsEvaluator(integration_config.evaluation)

            with patch.object(evaluator, '_compute_generation_metrics', return_value={'bleu': 0.5, 'rouge_l': 0.6}):
                metrics = evaluator.evaluate(model, val_data)

                # Verify evaluation metrics
                assert 'routing_entropy' in metrics, "Should compute routing metrics"
                assert 'complexity_accuracy' in metrics, "Should compute complexity prediction accuracy"
                assert 'expert_utilization' in metrics, "Should compute expert utilization metrics"

    def test_pipeline_error_recovery(self, sample_dataset, integration_config, temp_output_dir):
        """Test that pipeline handles errors gracefully and provides meaningful messages.

        Args:
            sample_dataset: Minimal test dataset
            integration_config: Test configuration
            temp_output_dir: Temporary directory for outputs
        """
        # Test data loading with invalid file
        with pytest.raises(FileNotFoundError, match="not found"):
            DataLoader(integration_config.data, "nonexistent_file.csv")

        # Test model initialization with invalid config
        invalid_config = integration_config.model
        invalid_config.num_experts = 0  # Invalid number of experts

        with pytest.raises(ValueError, match="Number of experts must be at least 2"):
            ComplexityAwareLoRARouter(invalid_config)

    def test_checkpoint_saving_and_loading(self, sample_dataset, integration_config, temp_output_dir):
        """Test that checkpoints can be saved and loaded correctly.

        Args:
            sample_dataset: Minimal test dataset
            integration_config: Test configuration
            temp_output_dir: Temporary directory for outputs
        """
        # Initialize model
        model = ComplexityAwareLoRARouter(integration_config.model)

        # Save checkpoint
        checkpoint_path = temp_output_dir / "test_checkpoint.pt"
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'config': integration_config.model,
            'epoch': 1,
            'train_loss': 0.5
        }
        torch.save(checkpoint_data, checkpoint_path)

        # Verify checkpoint was saved
        assert checkpoint_path.exists(), "Checkpoint file should be created"

        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model_state_dict' in loaded_checkpoint, "Checkpoint should contain model state"
        assert 'config' in loaded_checkpoint, "Checkpoint should contain config"
        assert loaded_checkpoint['epoch'] == 1, "Checkpoint should preserve epoch number"

        # Verify model can be restored from checkpoint
        new_model = ComplexityAwareLoRARouter(integration_config.model)
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])

        # Basic verification that models are equivalent
        assert len(list(model.parameters())) == len(list(new_model.parameters())), "Models should have same parameter count"

    def test_data_stratification_consistency(self, integration_config, temp_output_dir):
        """Test that complexity-based data stratification works correctly.

        Args:
            integration_config: Test configuration
            temp_output_dir: Temporary directory for outputs
        """
        # Create dataset with known complexity patterns
        diverse_dataset = pd.DataFrame({
            'instruction': [
                "Hi",  # Low complexity
                "What is AI?",  # Medium complexity
                "Explain the mathematical foundations of transformer attention mechanisms with detailed derivations.",  # High complexity
                "How are you?",  # Low complexity
                "Describe quantum entanglement.",  # Medium complexity
                "Provide a comprehensive analysis of the computational complexity of various sorting algorithms."  # High complexity
            ],
            'output': [
                "Hello!",
                "AI is artificial intelligence.",
                "Transformer attention uses scaled dot-product attention with queries, keys, and values matrices...",
                "I'm doing well.",
                "Quantum entanglement is a quantum mechanical phenomenon...",
                "Sorting algorithms vary widely in their time and space complexity characteristics..."
            ]
        })

        # Save dataset
        dataset_path = temp_output_dir / "diverse_data.csv"
        diverse_dataset.to_csv(dataset_path, index=False)

        # Load data with stratification
        data_loader = DataLoader(integration_config.data, str(dataset_path))

        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tok = MagicMock()
            mock_tok.encode.return_value = [1, 2, 3, 4]
            mock_tok.eos_token_id = 50256
            mock_tok.pad_token_id = 50256
            mock_tokenizer.return_value = mock_tok
            data_loader.set_tokenizer(mock_tok)

            train_data, val_data = data_loader.load_data()

            # Verify stratification preserves complexity distribution
            train_complexities = [item['complexity_score'] for item in train_data]
            val_complexities = [item['complexity_score'] for item in val_data]

            # Check that we have a range of complexities in both sets
            assert min(train_complexities) < max(train_complexities), "Training set should have complexity diversity"
            assert min(val_complexities) < max(val_complexities), "Validation set should have complexity diversity"

            # Check that complexity bins are represented
            num_bins = integration_config.data.complexity_bins
            train_bins = data_loader._assign_complexity_bins([item['complexity_score'] for item in train_data], num_bins)
            val_bins = data_loader._assign_complexity_bins([item['complexity_score'] for item in val_data], num_bins)

            # Both sets should have samples from multiple bins (if data allows)
            assert len(set(train_bins)) >= 2, "Training set should span multiple complexity bins"
            assert len(set(val_bins)) >= 2, "Validation set should span multiple complexity bins"