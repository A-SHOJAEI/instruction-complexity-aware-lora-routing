"""Tests for model architecture and components."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

from src.instruction_complexity_aware_lora_routing.models.model import ComplexityRouter, ComplexityAwareLoRARouter
from src.instruction_complexity_aware_lora_routing.utils.config import ModelConfig


class TestComplexityRouter:
    """Tests for ComplexityRouter class."""

    def test_init(self):
        """Test router initialization."""
        router = ComplexityRouter(
            input_dim=128,
            hidden_dim=64,
            num_experts=3,
            dropout=0.1,
            temperature=1.0
        )

        assert router.num_experts == 3
        assert router.temperature == 1.0
        assert isinstance(router.feature_encoder, nn.Sequential)
        assert isinstance(router.router_head, nn.Linear)
        assert isinstance(router.complexity_head, nn.Linear)

    def test_forward(self):
        """Test router forward pass."""
        router = ComplexityRouter(
            input_dim=128,
            hidden_dim=64,
            num_experts=3,
            dropout=0.1
        )

        batch_size = 4
        input_features = torch.randn(batch_size, 128)

        outputs = router(input_features)

        # Check output keys
        required_keys = ['complexity_pred', 'routing_logits', 'routing_weights', 'expert_assignments']
        for key in required_keys:
            assert key in outputs

        # Check shapes
        assert outputs['complexity_pred'].shape == (batch_size,)
        assert outputs['routing_logits'].shape == (batch_size, 3)
        assert outputs['routing_weights'].shape == (batch_size, 3)
        assert outputs['expert_assignments'].shape == (batch_size,)

        # Check value ranges
        assert torch.all(outputs['complexity_pred'] >= 0)
        assert torch.all(outputs['complexity_pred'] <= 1)
        assert torch.allclose(outputs['routing_weights'].sum(dim=1), torch.ones(batch_size))

    def test_forward_no_routing_weights(self):
        """Test forward pass without routing weights."""
        router = ComplexityRouter(
            input_dim=128,
            hidden_dim=64,
            num_experts=3
        )

        input_features = torch.randn(4, 128)
        outputs = router(input_features, return_routing_weights=False)

        assert 'complexity_pred' in outputs
        assert 'routing_logits' in outputs
        assert 'routing_weights' not in outputs
        assert 'expert_assignments' not in outputs

    def test_compute_load_balancing_loss(self):
        """Test load balancing loss computation."""
        router = ComplexityRouter(
            input_dim=128,
            hidden_dim=64,
            num_experts=3
        )

        # Perfect balance
        perfect_weights = torch.ones(4, 3) / 3
        perfect_loss = router.compute_load_balancing_loss(perfect_weights)
        assert perfect_loss.item() == pytest.approx(0.0, abs=1e-6)

        # Imbalanced weights
        imbalanced_weights = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        imbalanced_loss = router.compute_load_balancing_loss(imbalanced_weights)
        assert imbalanced_loss.item() > perfect_loss.item()

    def test_get_routing_entropy(self):
        """Test routing entropy computation."""
        router = ComplexityRouter(
            input_dim=128,
            hidden_dim=64,
            num_experts=3
        )

        # Confident routing (low entropy)
        confident_weights = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        confident_entropy = router.get_routing_entropy(confident_weights)

        # Uncertain routing (high entropy)
        uncertain_weights = torch.ones(2, 3) / 3
        uncertain_entropy = router.get_routing_entropy(uncertain_weights)

        assert uncertain_entropy > confident_entropy


class TestComplexityAwareLoRARouter:
    """Tests for ComplexityAwareLoRARouter class."""

    @patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM')
    @patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model')
    def test_init(self, mock_get_peft_model, mock_auto_model, small_model_config, sample_tokenizer):
        """Test model initialization."""
        # Mock base model
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 512
        mock_auto_model.from_pretrained.return_value = mock_base_model

        # Mock PEFT models
        mock_peft_model = MagicMock()
        mock_get_peft_model.return_value = mock_peft_model

        model = ComplexityAwareLoRARouter(
            config=small_model_config,
            base_model_name="microsoft/DialoGPT-small",
            tokenizer=sample_tokenizer
        )

        assert model.config == small_model_config
        assert model.tokenizer == sample_tokenizer
        assert model.num_experts == small_model_config.num_experts
        assert len(model.expert_models) == small_model_config.num_experts
        assert model.router is not None

        # Check that base model loading was called
        mock_auto_model.from_pretrained.assert_called_once()

    @patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM')
    @patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model')
    def test_extract_instruction_features(self, mock_get_peft_model, mock_auto_model, small_model_config, sample_tokenizer):
        """Test instruction feature extraction."""
        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 512

        # Mock forward pass
        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [torch.randn(2, 10, 512)]  # Last layer
        mock_base_model.forward.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_base_model

        mock_get_peft_model.return_value = MagicMock()

        model = ComplexityAwareLoRARouter(
            config=small_model_config,
            base_model_name="microsoft/DialoGPT-small",
            tokenizer=sample_tokenizer
        )

        input_ids = torch.randint(0, 1000, (2, 10))
        features = model._extract_instruction_features(input_ids)

        assert features.shape == (2, 512)
        mock_base_model.forward.assert_called_once()

    @patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM')
    @patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model')
    def test_create_router_input(self, mock_get_peft_model, mock_auto_model, small_model_config, sample_tokenizer):
        """Test router input creation."""
        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 512
        mock_auto_model.from_pretrained.return_value = mock_base_model
        mock_get_peft_model.return_value = MagicMock()

        model = ComplexityAwareLoRARouter(
            config=small_model_config,
            base_model_name="microsoft/DialoGPT-small",
            tokenizer=sample_tokenizer
        )

        batch_size = 2
        instruction_features = torch.randn(batch_size, 512)
        complexity_scores = torch.tensor([0.3, 0.8])

        router_input = model._create_router_input(instruction_features, complexity_scores)

        # Expected size: hidden_size + 1 (complexity) + 1 (threshold feature)
        expected_size = 512 + 1 + 1  # For 1 threshold in small_model_config
        assert router_input.shape == (batch_size, expected_size)

    @patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM')
    @patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model')
    def test_forward_routing_only(self, mock_get_peft_model, mock_auto_model, small_model_config, sample_tokenizer):
        """Test forward pass in routing-only mode."""
        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 512
        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [torch.randn(2, 10, 512)]
        mock_base_model.forward.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_base_model
        mock_get_peft_model.return_value = MagicMock()

        model = ComplexityAwareLoRARouter(
            config=small_model_config,
            base_model_name="microsoft/DialoGPT-small",
            tokenizer=sample_tokenizer
        )

        # Create sample inputs
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        complexity_scores = torch.tensor([0.3, 0.8])
        expert_assignments = torch.tensor([0, 1])

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            complexity_scores=complexity_scores,
            expert_assignments=expert_assignments,
            training_mode="routing_only"
        )

        # Should have routing outputs
        assert 'routing_logits' in outputs
        assert 'complexity_pred' in outputs
        assert 'routing_loss' in outputs
        assert 'complexity_loss' in outputs

    def test_save_and_load_pretrained(self, small_model_config, sample_tokenizer, temp_dir):
        """Test saving and loading model."""
        with patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM'), \
             patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model'):

            # Mock setup
            mock_base_model = MagicMock()
            mock_base_model.config.hidden_size = 512

            # Create model
            model = ComplexityAwareLoRARouter(
                config=small_model_config,
                base_model_name="microsoft/DialoGPT-small",
                tokenizer=sample_tokenizer
            )

            # Test saving
            save_path = temp_dir + "/test_model"

            # Mock the save_pretrained methods
            for expert in model.expert_models:
                expert.save_pretrained = MagicMock()

            model.save_pretrained(save_path)

            # Check that save was called
            for expert in model.expert_models:
                expert.save_pretrained.assert_called()


class TestModelIntegration:
    """Integration tests for model components."""

    @patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM')
    @patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model')
    def test_full_forward_pass(self, mock_get_peft_model, mock_auto_model, small_model_config, sample_tokenizer):
        """Test full forward pass through the model."""
        # Setup comprehensive mocks
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 512

        # Mock base model forward for feature extraction
        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [torch.randn(2, 10, 512)]
        mock_base_model.forward.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_base_model

        # Mock expert models
        mock_expert = MagicMock()
        mock_expert_outputs = MagicMock()
        mock_expert_outputs.loss = torch.tensor(0.5)
        mock_expert_outputs.logits = torch.randn(1, 10, 1000)
        mock_expert.forward.return_value = mock_expert_outputs
        mock_expert.__call__ = mock_expert.forward
        mock_get_peft_model.return_value = mock_expert

        model = ComplexityAwareLoRARouter(
            config=small_model_config,
            base_model_name="microsoft/DialoGPT-small",
            tokenizer=sample_tokenizer
        )

        # Create sample batch
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        complexity_scores = torch.tensor([0.3, 0.8])
        expert_assignments = torch.tensor([0, 1])

        # Test joint training mode
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            complexity_scores=complexity_scores,
            expert_assignments=expert_assignments,
            training_mode="joint"
        )

        # Check that we get the expected outputs
        assert 'loss' in outputs
        assert 'routing_loss' in outputs
        assert 'complexity_loss' in outputs
        assert 'expert_loss' in outputs

    @patch('instruction_complexity_aware_lora_routing.models.model.AutoModelForCausalLM')
    @patch('instruction_complexity_aware_lora_routing.models.model.get_peft_model')
    def test_generate(self, mock_get_peft_model, mock_auto_model, small_model_config, sample_tokenizer):
        """Test text generation."""
        # Setup mocks
        mock_base_model = MagicMock()
        mock_base_model.config.hidden_size = 512

        mock_outputs = MagicMock()
        mock_outputs.hidden_states = [torch.randn(1, 5, 512)]
        mock_base_model.forward.return_value = mock_outputs
        mock_auto_model.from_pretrained.return_value = mock_base_model

        # Mock expert generation
        mock_expert = MagicMock()
        generated_ids = torch.randint(0, 1000, (1, 10))
        mock_expert.generate.return_value = generated_ids
        mock_get_peft_model.return_value = mock_expert

        model = ComplexityAwareLoRARouter(
            config=small_model_config,
            base_model_name="microsoft/DialoGPT-small",
            tokenizer=sample_tokenizer
        )

        # Test generation
        input_ids = torch.randint(0, 1000, (1, 5))
        attention_mask = torch.ones(1, 5)
        complexity_scores = torch.tensor([0.3])

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            complexity_scores=complexity_scores,
            max_length=10
        )

        assert generated.shape[0] == 1
        mock_expert.generate.assert_called()