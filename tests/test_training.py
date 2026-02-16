"""Tests for training utilities and components."""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.instruction_complexity_aware_lora_routing.training.trainer import LoRATrainer, EarlyStopping
from src.instruction_complexity_aware_lora_routing.utils.config import Config


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_init(self):
        """Test early stopping initialization."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01, restore_best_weights=True)

        assert early_stop.patience == 3
        assert early_stop.min_delta == 0.01
        assert early_stop.restore_best_weights is True
        assert early_stop.best_loss == float('inf')
        assert early_stop.counter == 0
        assert early_stop.early_stop is False

    def test_improvement(self):
        """Test behavior when validation improves."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01)
        model = nn.Linear(10, 1)

        # First call with good loss
        result = early_stop(0.5, model)
        assert not result
        assert early_stop.best_loss == 0.5
        assert early_stop.counter == 0

        # Second call with better loss
        result = early_stop(0.4, model)
        assert not result
        assert early_stop.best_loss == 0.4
        assert early_stop.counter == 0

    def test_no_improvement(self):
        """Test behavior when validation doesn't improve."""
        early_stop = EarlyStopping(patience=2, min_delta=0.01)
        model = nn.Linear(10, 1)

        # Set initial best loss
        early_stop(0.5, model)

        # Worse loss
        result = early_stop(0.6, model)
        assert not result
        assert early_stop.counter == 1

        # Still worse loss - should trigger stopping
        result = early_stop(0.7, model)
        assert result
        assert early_stop.early_stop is True

    def test_restore_best_weights(self):
        """Test weight restoration on early stopping."""
        early_stop = EarlyStopping(patience=1, min_delta=0.0, restore_best_weights=True)
        model = nn.Linear(10, 1)

        # Get initial weights
        initial_weights = model.state_dict().copy()

        # First call saves weights
        early_stop(0.5, model)

        # Modify model weights
        with torch.no_grad():
            model.weight.fill_(999.0)

        # Trigger early stopping - should restore weights
        early_stop(0.6, model)

        # Weights should be restored (not all 999s)
        current_weights = model.state_dict()
        assert not torch.allclose(current_weights['weight'], torch.full_like(current_weights['weight'], 999.0))


class TestLoRATrainer:
    """Tests for LoRATrainer class."""

    @patch('instruction_complexity_aware_lora_routing.training.trainer.Accelerator')
    @patch('instruction_complexity_aware_lora_routing.training.trainer.mlflow')
    def test_init(self, mock_mlflow, mock_accelerator, sample_config):
        """Test trainer initialization."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.router.parameters.return_value = [nn.Parameter(torch.randn(10))]
        mock_model.expert_models = [MagicMock()]
        mock_model.expert_models[0].parameters.return_value = [nn.Parameter(torch.randn(10))]

        mock_tokenizer = MagicMock()

        # Create mock dataloaders
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()

        # Setup accelerator mock
        mock_accelerator_instance = MagicMock()
        mock_accelerator_instance.is_main_process = True
        mock_accelerator_instance.prepare.return_value = (
            mock_model, MagicMock(), MagicMock(), mock_train_loader, mock_val_loader
        )
        mock_accelerator.return_value = mock_accelerator_instance

        trainer = LoRATrainer(
            config=sample_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_train_loader,
            val_dataloader=mock_val_loader
        )

        assert trainer.config == sample_config
        assert trainer.optimizer is not None
        assert trainer.router_optimizer is not None
        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.early_stopping is not None

        # Check MLflow was set up
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()

    @patch('instruction_complexity_aware_lora_routing.training.trainer.Accelerator')
    @patch('instruction_complexity_aware_lora_routing.training.trainer.mlflow')
    def test_setup_optimizers(self, mock_mlflow, mock_accelerator, sample_config):
        """Test optimizer setup."""
        # Mock model with router and expert parameters
        mock_model = MagicMock()
        router_param = nn.Parameter(torch.randn(10))
        expert_param = nn.Parameter(torch.randn(10))

        mock_model.router.parameters.return_value = [router_param]
        mock_expert = MagicMock()
        mock_expert.parameters.return_value = [expert_param]
        mock_model.expert_models = [mock_expert]

        mock_tokenizer = MagicMock()
        mock_train_loader = MagicMock()
        mock_val_loader = MagicMock()

        # Setup accelerator mock
        mock_accelerator_instance = MagicMock()
        mock_accelerator_instance.is_main_process = True
        mock_accelerator_instance.prepare.return_value = (
            mock_model, MagicMock(), MagicMock(), mock_train_loader, mock_val_loader
        )
        mock_accelerator.return_value = mock_accelerator_instance

        trainer = LoRATrainer(
            config=sample_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_train_loader,
            val_dataloader=mock_val_loader
        )

        # Check that optimizers were created
        assert trainer.optimizer is not None
        assert trainer.router_optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.router_scheduler is not None

    @patch('instruction_complexity_aware_lora_routing.training.trainer.Accelerator')
    @patch('instruction_complexity_aware_lora_routing.training.trainer.mlflow')
    def test_validate_epoch(self, mock_mlflow, mock_accelerator, sample_config):
        """Test validation epoch."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.router.parameters.return_value = [nn.Parameter(torch.randn(10))]
        mock_model.expert_models = [MagicMock()]
        mock_model.expert_models[0].parameters.return_value = [nn.Parameter(torch.randn(10))]

        # Mock model outputs
        mock_outputs = {
            'loss': torch.tensor(0.5),
            'expert_assignments': torch.tensor([0, 1]),
            'routing_logits': torch.randn(2, 3),
            'complexity_pred': torch.tensor([0.3, 0.7])
        }
        mock_model.return_value = mock_outputs
        mock_model.__call__ = lambda *args, **kwargs: mock_outputs

        mock_tokenizer = MagicMock()

        # Create mock validation batch
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10)),
            'complexity_score': torch.tensor([0.3, 0.7]),
            'expert_assignment': torch.tensor([0, 1])
        }

        mock_val_loader = [mock_batch]  # Single batch

        # Setup accelerator mock
        mock_accelerator_instance = MagicMock()
        mock_accelerator_instance.is_main_process = False  # Disable progress bar
        mock_accelerator_instance.prepare.return_value = (
            mock_model, MagicMock(), MagicMock(), MagicMock(), mock_val_loader
        )
        mock_accelerator.return_value = mock_accelerator_instance

        trainer = LoRATrainer(
            config=sample_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=MagicMock(),
            val_dataloader=mock_val_loader
        )

        # Run validation
        metrics = trainer._validate_epoch()

        # Check metrics
        assert 'loss' in metrics
        assert 'routing_accuracy' in metrics
        assert 'complexity_mse' in metrics
        assert 'expert_usage' in metrics

        # Check value ranges
        assert metrics['loss'] >= 0
        assert 0 <= metrics['routing_accuracy'] <= 1
        assert metrics['complexity_mse'] >= 0

    @patch('instruction_complexity_aware_lora_routing.training.trainer.Accelerator')
    @patch('instruction_complexity_aware_lora_routing.training.trainer.mlflow')
    def test_save_checkpoint(self, mock_mlflow, mock_accelerator, sample_config, temp_dir):
        """Test checkpoint saving."""
        # Mock model
        mock_model = MagicMock()
        mock_model.router.parameters.return_value = [nn.Parameter(torch.randn(10))]
        mock_model.expert_models = [MagicMock()]
        mock_model.expert_models[0].parameters.return_value = [nn.Parameter(torch.randn(10))]

        # Mock unwrapped model
        mock_unwrapped = MagicMock()
        mock_unwrapped.save_pretrained = MagicMock()

        mock_tokenizer = MagicMock()

        # Setup accelerator mock
        mock_accelerator_instance = MagicMock()
        mock_accelerator_instance.is_main_process = True
        mock_accelerator_instance.unwrap_model.return_value = mock_unwrapped
        mock_accelerator_instance.prepare.return_value = (
            mock_model, MagicMock(), MagicMock(), MagicMock(), MagicMock()
        )
        mock_accelerator.return_value = mock_accelerator_instance

        # Update config with temp directory
        sample_config.output_dir = temp_dir

        trainer = LoRATrainer(
            config=sample_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=MagicMock(),
            val_dataloader=MagicMock()
        )

        # Test saving
        trainer._save_checkpoint(epoch=1, is_best=True)

        # Check that save_pretrained was called
        mock_unwrapped.save_pretrained.assert_called_once()

        # Check checkpoint directory was created
        checkpoint_dir = Path(temp_dir) / "checkpoints"
        assert checkpoint_dir.exists()

    @patch('instruction_complexity_aware_lora_routing.training.trainer.Accelerator')
    @patch('instruction_complexity_aware_lora_routing.training.trainer.mlflow')
    def test_evaluate_inference_speed(self, mock_mlflow, mock_accelerator, sample_config):
        """Test inference speed evaluation."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.router.parameters.return_value = [nn.Parameter(torch.randn(10))]
        mock_model.expert_models = [MagicMock()]
        mock_model.expert_models[0].parameters.return_value = [nn.Parameter(torch.randn(10))]

        # Mock model methods
        mock_model._extract_instruction_features.return_value = torch.randn(1, 512)
        mock_model._create_router_input.return_value = torch.randn(1, 514)

        mock_router_outputs = {'expert_assignments': torch.tensor([0])}
        mock_model.router.return_value = mock_router_outputs

        mock_model.generate.return_value = torch.randint(0, 1000, (1, 10))

        mock_tokenizer = MagicMock()

        # Create mock validation batch
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (5, 10)),
            'attention_mask': torch.ones(5, 10),
            'complexity_score': torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        }
        mock_val_loader = [mock_batch]

        # Setup accelerator mock
        mock_accelerator_instance = MagicMock()
        mock_accelerator_instance.is_main_process = True
        mock_accelerator_instance.prepare.return_value = (
            mock_model, MagicMock(), MagicMock(), MagicMock(), mock_val_loader
        )
        mock_accelerator.return_value = mock_accelerator_instance

        trainer = LoRATrainer(
            config=sample_config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=MagicMock(),
            val_dataloader=mock_val_loader
        )

        # Test speed evaluation
        speed_metrics = trainer.evaluate_inference_speed(num_samples=3)

        # Check metrics
        required_metrics = [
            'avg_routing_latency_ms',
            'std_routing_latency_ms',
            'avg_generation_latency_ms',
            'std_generation_latency_ms',
            'total_latency_ms'
        ]

        for metric in required_metrics:
            assert metric in speed_metrics
            assert speed_metrics[metric] >= 0  # Latency should be non-negative


class TestTrainerIntegration:
    """Integration tests for trainer components."""

    @patch('instruction_complexity_aware_lora_routing.training.trainer.Accelerator')
    @patch('instruction_complexity_aware_lora_routing.training.trainer.mlflow')
    def test_full_training_step(self, mock_mlflow, mock_accelerator, small_training_config):
        """Test a complete training step."""
        # Create minimal config for fast testing
        config = Config()
        config.training = small_training_config
        config.model.num_experts = 2

        # Mock model
        mock_model = MagicMock()
        mock_model.router.parameters.return_value = [nn.Parameter(torch.randn(10))]
        mock_expert = MagicMock()
        mock_expert.parameters.return_value = [nn.Parameter(torch.randn(10))]
        mock_model.expert_models = [mock_expert, mock_expert]  # Two experts

        # Mock forward pass outputs
        mock_outputs = {
            'loss': torch.tensor(0.5),
            'routing_loss': torch.tensor(0.1),
            'complexity_loss': torch.tensor(0.05),
            'expert_loss': torch.tensor(0.3)
        }
        mock_model.return_value = mock_outputs
        mock_model.__call__ = lambda *args, **kwargs: mock_outputs

        mock_tokenizer = MagicMock()

        # Create mock batch
        mock_batch = {
            'input_ids': torch.randint(0, 1000, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 1000, (2, 10)),
            'complexity_score': torch.tensor([0.3, 0.7]),
            'expert_assignment': torch.tensor([0, 1])
        }

        mock_train_loader = [mock_batch]
        mock_val_loader = [mock_batch]

        # Setup accelerator mock
        mock_accelerator_instance = MagicMock()
        mock_accelerator_instance.is_main_process = False
        mock_accelerator_instance.accumulate.return_value.__enter__ = MagicMock(return_value=None)
        mock_accelerator_instance.accumulate.return_value.__exit__ = MagicMock(return_value=None)
        mock_accelerator_instance.backward = MagicMock()
        mock_accelerator_instance.clip_grad_norm_ = MagicMock()

        mock_optimizer = MagicMock()
        mock_router_optimizer = MagicMock()

        mock_accelerator_instance.prepare.return_value = (
            mock_model, mock_optimizer, mock_router_optimizer, mock_train_loader, mock_val_loader
        )
        mock_accelerator.return_value = mock_accelerator_instance

        trainer = LoRATrainer(
            config=config,
            model=mock_model,
            tokenizer=mock_tokenizer,
            train_dataloader=mock_train_loader,
            val_dataloader=mock_val_loader
        )

        # Override optimizers with mocks
        trainer.optimizer = mock_optimizer
        trainer.router_optimizer = mock_router_optimizer
        trainer.scheduler = MagicMock()
        trainer.router_scheduler = MagicMock()

        # Run single training step
        metrics = trainer._train_epoch()

        # Check that optimizers were called
        assert mock_optimizer.step.call_count > 0
        assert mock_router_optimizer.step.call_count > 0

        # Check metrics
        assert 'loss' in metrics
        assert metrics['loss'] >= 0