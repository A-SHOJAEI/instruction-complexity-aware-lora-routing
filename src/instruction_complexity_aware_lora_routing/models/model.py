"""Model architecture for complexity-aware LoRA routing."""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from ..utils.config import ModelConfig
from ..utils.constants import EPSILON, COMPLEXITY_LOSS_WEIGHT, LOAD_BALANCE_LOSS_WEIGHT

logger = logging.getLogger(__name__)


class ComplexityRouter(nn.Module):
    """Neural router that predicts expert selection based on instruction complexity."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        """Initialize complexity router.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Hidden layer dimension.
            num_experts: Number of expert models.
            dropout: Dropout probability.
            temperature: Temperature for softmax (for gating).
        """
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        # Feature extraction layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Router head for expert selection
        self.router_head = nn.Linear(hidden_dim // 2, num_experts)

        # Complexity prediction head (auxiliary task)
        self.complexity_head = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        input_features: torch.Tensor,
        return_routing_weights: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through router.

        Args:
            input_features: Input features tensor [batch_size, feature_dim].
            return_routing_weights: Whether to return routing weights.

        Returns:
            Dictionary containing routing outputs.
        """
        # Extract features
        features = self.feature_encoder(input_features)

        # Predict complexity score
        complexity_pred = torch.sigmoid(self.complexity_head(features)).squeeze(-1)

        # Log complexity distribution for debugging
        if hasattr(self, '_log_routing_stats') and self._log_routing_stats:
            logger.debug(f"Complexity predictions - mean: {complexity_pred.mean().item():.3f}, "
                        f"std: {complexity_pred.std().item():.3f}")

        # Compute routing logits
        routing_logits = self.router_head(features)

        outputs = {
            'complexity_pred': complexity_pred,
            'routing_logits': routing_logits,
        }

        if return_routing_weights:
            # Apply temperature scaling and softmax
            routing_weights = F.softmax(routing_logits / self.temperature, dim=-1)
            outputs['routing_weights'] = routing_weights

            # Hard assignment (argmax)
            expert_assignments = torch.argmax(routing_logits, dim=-1)
            outputs['expert_assignments'] = expert_assignments

        return outputs

    def compute_load_balancing_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute load balancing loss to encourage uniform expert usage.

        Args:
            routing_weights: Routing weights tensor [batch_size, num_experts].

        Returns:
            Load balancing loss.
        """
        # Compute expert usage frequencies
        expert_usage = torch.mean(routing_weights, dim=0)  # [num_experts]

        # Compute uniform target
        uniform_target = 1.0 / self.num_experts

        # L2 loss from uniform distribution
        load_balance_loss = torch.mean((expert_usage - uniform_target) ** 2)

        return load_balance_loss

    def get_routing_entropy(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute routing entropy to measure uncertainty.

        Args:
            routing_weights: Routing weights tensor [batch_size, num_experts].

        Returns:
            Routing entropy.
        """
        # Add small epsilon to avoid log(0)
        eps = EPSILON
        entropy = -torch.sum(routing_weights * torch.log(routing_weights + eps), dim=-1)
        return torch.mean(entropy)


class ComplexityAwareLoRARouter(nn.Module):
    """Main model with complexity-aware LoRA routing."""

    def __init__(
        self,
        config: ModelConfig,
        base_model_name: str,
        tokenizer: PreTrainedTokenizer
    ):
        """Initialize complexity-aware LoRA router.

        Args:
            config: Model configuration.
            base_model_name: Base model name/path.
            tokenizer: Tokenizer instance.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.num_experts = config.num_experts

        # Store model name for creating independent expert copies
        self._base_model_name = base_model_name

        # Load base model for feature extraction only
        logger.info(f"Loading base model: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )

        # Get model configuration
        self.model_config = self.base_model.config
        self.hidden_size = self.model_config.hidden_size

        # Create LoRA experts
        self.expert_models = nn.ModuleList()
        self._create_lora_experts()

        # Create complexity router
        # Router input: concatenation of instruction embeddings + complexity features
        router_input_dim = self.hidden_size + len(config.complexity_thresholds) + 1
        self.router = ComplexityRouter(
            input_dim=router_input_dim,
            hidden_dim=config.router_hidden_dim,
            num_experts=config.num_experts,
            dropout=config.router_dropout
        )

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        logger.info(f"Created model with {config.num_experts} LoRA experts")

    def _create_lora_experts(self) -> None:
        """Create LoRA expert models with independent base model copies.

        Each expert gets its own frozen base model wrapped in a separate PeftModel,
        ensuring independent LoRA parameters that can specialize differently.
        """
        for expert_idx in range(self.num_experts):
            logger.info(f"Creating LoRA expert {expert_idx} (independent base model)")

            # Load a fresh base model for each expert to ensure independence
            expert_base = AutoModelForCausalLM.from_pretrained(
                self._base_model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

            # Freeze base model parameters - only LoRA adapters will be trained
            for param in expert_base.parameters():
                param.requires_grad = False

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
            )

            # Create independent PeftModel for this expert
            expert_model = get_peft_model(expert_base, lora_config)
            self.expert_models.append(expert_model)

        # Log parameter counts
        total_trainable = sum(
            p.numel() for expert in self.expert_models
            for p in expert.parameters() if p.requires_grad
        )
        logger.info(f"Total trainable LoRA parameters across {self.num_experts} experts: {total_trainable:,}")

    def _extract_instruction_features(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract instruction features using base model embeddings.

        Args:
            input_ids: Input token IDs [batch_size, seq_len].

        Returns:
            Instruction features [batch_size, hidden_size].
        """
        # Use base model to get hidden states
        with torch.no_grad():
            outputs = self.base_model.forward(
                input_ids=input_ids,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # Last layer

            # Pool instruction features (mean pooling)
            # In practice, you might want to identify instruction tokens specifically
            instruction_features = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]

        return instruction_features

    def _create_router_input(
        self,
        instruction_features: torch.Tensor,
        complexity_scores: torch.Tensor
    ) -> torch.Tensor:
        """Create router input features.

        Args:
            instruction_features: Instruction embeddings [batch_size, hidden_size].
            complexity_scores: Complexity scores [batch_size].

        Returns:
            Router input features [batch_size, feature_dim].
        """
        batch_size = instruction_features.size(0)

        # Create complexity-based features
        complexity_features = []

        # Add raw complexity score
        complexity_features.append(complexity_scores.unsqueeze(-1))

        # Add threshold-based features
        for threshold in self.config.complexity_thresholds:
            threshold_feature = (complexity_scores > threshold).float().unsqueeze(-1)
            complexity_features.append(threshold_feature)

        # Concatenate all features
        complexity_tensor = torch.cat(complexity_features, dim=-1)  # [batch_size, num_thresholds + 1]

        # Combine instruction and complexity features
        router_input = torch.cat([instruction_features, complexity_tensor], dim=-1)

        return router_input

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        complexity_scores: Optional[torch.Tensor] = None,
        expert_assignments: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        training_mode: str = "joint"  # "joint", "routing_only", "expert_only"
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            labels: Target labels [batch_size, seq_len].
            complexity_scores: Complexity scores [batch_size].
            expert_assignments: Ground truth expert assignments [batch_size].
            return_dict: Whether to return dictionary.
            training_mode: Training mode - "joint", "routing_only", or "expert_only".

        Returns:
            Model outputs dictionary.
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Extract instruction features for routing
        instruction_features = self._extract_instruction_features(input_ids)

        outputs = {}

        # Router forward pass
        if complexity_scores is not None:
            router_input = self._create_router_input(instruction_features, complexity_scores)
            router_outputs = self.router(router_input)

            outputs.update(router_outputs)

            # Compute routing loss
            if expert_assignments is not None:
                routing_loss = F.cross_entropy(
                    router_outputs['routing_logits'],
                    expert_assignments
                )
                outputs['routing_loss'] = routing_loss

            # Compute complexity prediction loss
            complexity_loss = F.mse_loss(
                router_outputs['complexity_pred'],
                complexity_scores
            )
            outputs['complexity_loss'] = complexity_loss

            # Load balancing loss
            if 'routing_weights' in router_outputs:
                load_balance_loss = self.router.compute_load_balancing_loss(
                    router_outputs['routing_weights']
                )
                outputs['load_balance_loss'] = load_balance_loss

        # Expert model forward passes
        if labels is not None and training_mode in ["joint", "expert_only"]:
            expert_losses = []
            expert_logits = []

            # Determine which experts to use
            if complexity_scores is not None and 'expert_assignments' in outputs:
                # Use router predictions
                predicted_experts = outputs['expert_assignments']
            elif expert_assignments is not None:
                # Use ground truth assignments
                predicted_experts = expert_assignments
            else:
                # Use all experts (ensemble)
                predicted_experts = None

            if predicted_experts is not None:
                # Route to specific experts - gather logits back into batch order
                batch_size = input_ids.size(0)
                combined_logits = None

                for expert_idx in range(self.num_experts):
                    # Find samples assigned to this expert
                    expert_mask = (predicted_experts == expert_idx)
                    if expert_mask.sum() == 0:
                        continue

                    # Get expert inputs
                    expert_input_ids = input_ids[expert_mask]
                    expert_attention_mask = attention_mask[expert_mask]
                    expert_labels = labels[expert_mask]

                    # Forward through expert
                    expert_outputs = self.expert_models[expert_idx](
                        input_ids=expert_input_ids,
                        attention_mask=expert_attention_mask,
                        labels=expert_labels
                    )

                    expert_losses.append(expert_outputs.loss)

                    # Scatter logits back into batch-sized tensor
                    if combined_logits is None:
                        combined_logits = torch.zeros(
                            batch_size, expert_outputs.logits.size(1), expert_outputs.logits.size(2),
                            device=expert_outputs.logits.device, dtype=expert_outputs.logits.dtype
                        )
                    combined_logits[expert_mask] = expert_outputs.logits

                if combined_logits is not None:
                    expert_logits.append(combined_logits)

            else:
                # Use all experts (for ensemble inference)
                for expert_idx in range(self.num_experts):
                    expert_outputs = self.expert_models[expert_idx](
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    expert_losses.append(expert_outputs.loss)
                    expert_logits.append(expert_outputs.logits)

            # Aggregate expert losses
            if expert_losses:
                outputs['expert_loss'] = torch.stack(expert_losses).mean()

            # Aggregate expert logits for ensemble
            if expert_logits and len(expert_logits) > 1:
                outputs['ensemble_logits'] = torch.stack(expert_logits).mean(dim=0)
            elif expert_logits:
                outputs['ensemble_logits'] = expert_logits[0]

        # Compute total loss
        # Combine losses with carefully tuned weights
        # These weights balance different training objectives:
        # - Routing loss: Full weight (1.0) - primary objective for expert selection
        # - Complexity loss: 0.1 - auxiliary task, helps routing but shouldn't dominate
        # - Load balance loss: 0.1 - regularization to prevent expert collapse
        # - Expert loss: Full weight (1.0) - primary objective for generation quality
        total_loss = 0.0
        if 'routing_loss' in outputs and training_mode in ["joint", "routing_only"]:
            total_loss += outputs['routing_loss']
        if 'complexity_loss' in outputs:
            total_loss += COMPLEXITY_LOSS_WEIGHT * outputs['complexity_loss']  # Weight complexity loss lower
        if 'load_balance_loss' in outputs:
            total_loss += LOAD_BALANCE_LOSS_WEIGHT * outputs['load_balance_loss']  # Small weight for load balancing
        if 'expert_loss' in outputs and training_mode in ["joint", "expert_only"]:
            total_loss += outputs['expert_loss']

        if total_loss > 0:
            outputs['loss'] = total_loss

        return outputs

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        complexity_scores: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        **generate_kwargs
    ) -> torch.Tensor:
        """Generate text using complexity-aware routing.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            complexity_scores: Complexity scores for routing.
            max_length: Maximum generation length.
            temperature: Generation temperature.
            do_sample: Whether to sample.
            **generate_kwargs: Additional generation arguments.

        Returns:
            Generated token IDs.
        """
        self.eval()
        with torch.no_grad():
            # Route to appropriate experts
            instruction_features = self._extract_instruction_features(input_ids)
            router_input = self._create_router_input(instruction_features, complexity_scores)
            router_outputs = self.router(router_input)

            expert_assignments = router_outputs['expert_assignments']

            # Generate with assigned experts
            generated_outputs = []
            for i in range(input_ids.size(0)):
                expert_idx = expert_assignments[i].item()
                sample_input = input_ids[i:i+1]
                sample_attention = attention_mask[i:i+1]

                # Generate with specific expert
                generated = self.expert_models[expert_idx].generate(
                    input_ids=sample_input,
                    attention_mask=sample_attention,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )
                generated_outputs.append(generated)

            return torch.cat(generated_outputs, dim=0)

    def save_pretrained(self, save_directory: str) -> None:
        """Save model to directory.

        Args:
            save_directory: Directory to save model.
        """
        import os
        os.makedirs(save_directory, exist_ok=True)

        # Save router
        router_path = os.path.join(save_directory, "router.pt")
        torch.save(self.router.state_dict(), router_path)

        # Save expert models
        for expert_idx, expert_model in enumerate(self.expert_models):
            expert_path = os.path.join(save_directory, f"expert_{expert_idx}")
            expert_model.save_pretrained(expert_path)

        # Save configuration
        config_path = os.path.join(save_directory, "model_config.pt")
        torch.save(self.config, config_path)

        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer: PreTrainedTokenizer,
        base_model_name: Optional[str] = None
    ) -> "ComplexityAwareLoRARouter":
        """Load model from directory.

        Args:
            model_path: Path to saved model.
            tokenizer: Tokenizer instance.
            base_model_name: Base model name (if different from saved config).

        Returns:
            Loaded model instance.
        """
        import os

        # Load configuration
        config_path = os.path.join(model_path, "model_config.pt")
        config = torch.load(config_path, map_location="cpu", weights_only=False)

        # Create model instance
        if base_model_name is None:
            base_model_name = config.base_model_name

        model = cls(config, base_model_name, tokenizer)

        # Load router
        router_path = os.path.join(model_path, "router.pt")
        model.router.load_state_dict(torch.load(router_path, map_location="cpu", weights_only=True))

        # Load expert models
        for expert_idx in range(config.num_experts):
            expert_path = os.path.join(model_path, f"expert_{expert_idx}")
            if os.path.exists(expert_path):
                # Load expert model
                expert_model = PeftModel.from_pretrained(model.base_model, expert_path)
                model.expert_models[expert_idx] = expert_model

        logger.info(f"Model loaded from {model_path}")
        return model