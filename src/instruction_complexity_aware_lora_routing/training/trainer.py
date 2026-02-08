"""Training utilities for complexity-aware LoRA routing."""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple, Any, List
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_scheduler, get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

from ..models.model import ComplexityAwareLoRARouter
from ..evaluation.metrics import RoutingMetrics, ComplexityMetrics
from ..utils.config import Config
from ..utils.constants import CHECKPOINT_SAVE_INTERVAL

logger = logging.getLogger(__name__)


def performance_monitor(func_name: str = None):
    """Decorator to monitor function execution time.

    Args:
        func_name: Optional custom name for the function (defaults to function.__name__).

    Returns:
        Decorated function that logs execution time.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = func_name or func.__name__
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"⏱️  {name} completed in {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ {name} failed after {duration:.3f}s: {e}")
                raise
        return wrapper
    return decorator


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait before stopping.
            min_delta: Minimum change to qualify as improvement.
            restore_best_weights: Whether to restore best weights on early stop.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.

        Args:
            val_loss: Current validation loss.
            model: Model to save best weights from.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)

        return self.early_stop


class LoRATrainer:
    """Trainer class for complexity-aware LoRA routing."""

    def __init__(
        self,
        config: Config,
        model: ComplexityAwareLoRARouter,
        tokenizer: AutoTokenizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: Optional[DataLoader] = None
    ):
        """Initialize trainer.

        Args:
            config: Training configuration.
            model: Model to train.
            tokenizer: Tokenizer instance.
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            test_dataloader: Optional test data loader.
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision='fp16' if config.training.fp16 else 'no'
        )

        # Set up optimizers
        self.optimizer = None
        self.router_optimizer = None
        self.scheduler = None
        self.router_scheduler = None
        self._setup_optimizers()

        # Prepare model and dataloaders with accelerator
        (
            self.model,
            self.optimizer,
            self.router_optimizer,
            self.train_dataloader,
            self.val_dataloader
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.router_optimizer,
            self.train_dataloader,
            self.val_dataloader
        )

        if self.test_dataloader:
            self.test_dataloader = self.accelerator.prepare(self.test_dataloader)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # Early stopping
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        # Evaluation metrics
        self.routing_metrics = RoutingMetrics()
        self.complexity_metrics = ComplexityMetrics()

        # MLflow setup
        self._setup_mlflow()

    def _setup_optimizers(self) -> None:
        """Set up optimizers and schedulers."""
        # Separate parameters for router and experts
        router_params = list(self.model.router.parameters())
        expert_params = []
        for expert in self.model.expert_models:
            expert_params.extend([p for p in expert.parameters() if p.requires_grad])

        # Expert optimizer
        self.optimizer = AdamW(
            expert_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        # Router optimizer (typically higher learning rate)
        self.router_optimizer = AdamW(
            router_params,
            lr=self.config.training.router_learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        # Calculate total training steps
        num_training_steps = (
            len(self.train_dataloader) * self.config.training.num_epochs //
            self.config.training.gradient_accumulation_steps
        )

        # Schedulers
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )

        self.router_scheduler = get_linear_schedule_with_warmup(
            self.router_optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        self.mlflow_active = False
        if self.accelerator.is_main_process:
            try:
                if self.config.mlflow_tracking_uri:
                    mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

                # Set experiment
                mlflow.set_experiment(self.config.mlflow_experiment_name)

                # Start run
                mlflow.start_run()

                # Log configuration
                mlflow.log_params({
                    "model_name": self.config.model.base_model_name,
                    "num_experts": self.config.model.num_experts,
                    "lora_rank": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "batch_size": self.config.training.batch_size,
                    "learning_rate": self.config.training.learning_rate,
                    "router_learning_rate": self.config.training.router_learning_rate,
                    "num_epochs": self.config.training.num_epochs,
                    "max_length": self.config.data.max_length,
                })
                self.mlflow_active = True
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}. Training will continue without MLflow.")

    @performance_monitor("Full training")
    def train(self) -> Dict[str, Any]:
        """Main training loop.

        Returns:
            Training history and final metrics.
        """
        logger.info("Starting training...")

        # Set random seeds
        torch.manual_seed(self.config.training.seed)
        np.random.seed(self.config.training.seed)

        training_history = {
            'train_loss': [],
            'val_loss': [],
            'routing_accuracy': [],
            'complexity_mse': [],
            'expert_usage': []
        }

        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.training.num_epochs}")

            # Train epoch
            train_metrics = self._train_epoch()
            training_history['train_loss'].append(train_metrics['loss'])

            # Validate
            val_metrics = self._validate_epoch()
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['routing_accuracy'].append(val_metrics['routing_accuracy'])
            training_history['complexity_mse'].append(val_metrics['complexity_mse'])
            training_history['expert_usage'].append(val_metrics['expert_usage'])

            # Log to MLflow
            if self.accelerator.is_main_process and self.mlflow_active:
                try:
                    mlflow.log_metrics({
                        'train_loss': train_metrics['loss'],
                        'val_loss': val_metrics['loss'],
                        'routing_accuracy': val_metrics['routing_accuracy'],
                        'complexity_mse': val_metrics['complexity_mse'],
                        'epoch': epoch + 1
                    }, step=self.global_step)
                except Exception:
                    pass

            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self._save_checkpoint(epoch, is_best=True)

            # Early stopping check
            if self.early_stopping(val_metrics['loss'], self.model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            # Regular checkpoint
            if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
                self._save_checkpoint(epoch, is_best=False)

        # Final evaluation
        if self.test_dataloader:
            test_metrics = self._evaluate_test()
            training_history['test_metrics'] = test_metrics

            if self.accelerator.is_main_process and self.mlflow_active:
                try:
                    mlflow.log_metrics({
                        'test_loss': test_metrics['loss'],
                        'test_routing_accuracy': test_metrics['routing_accuracy'],
                        'test_complexity_mse': test_metrics['complexity_mse'],
                    })
                except Exception:
                    pass

        # End MLflow run
        if self.accelerator.is_main_process and self.mlflow_active:
            try:
                mlflow.end_run()
            except Exception:
                pass

        logger.info("Training completed!")
        return training_history

    @performance_monitor("Training epoch")
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Training metrics for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Create progress bar
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch + 1}",
            disable=not self.accelerator.is_main_process
        )

        for batch in progress_bar:
            # Three-phase training: router, experts, joint
            phase_losses = []

            # Phase 1: Train router only
            self.router_optimizer.zero_grad()
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    complexity_scores=batch['complexity_score'],
                    expert_assignments=batch['expert_assignment'],
                    training_mode="routing_only"
                )

                loss = outputs.get('loss', 0.0)
                if loss > 0:
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(
                        self.model.router.parameters(),
                        self.config.training.max_grad_norm
                    )
                    self.router_optimizer.step()
                    self.router_scheduler.step()
                    phase_losses.append(loss.item())

            # Phase 2: Train experts only
            self.optimizer.zero_grad()
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    complexity_scores=batch['complexity_score'],
                    expert_assignments=batch['expert_assignment'],
                    training_mode="expert_only"
                )

                loss = outputs.get('loss', 0.0)
                if loss > 0:
                    self.accelerator.backward(loss)
                    # Clip gradients for trainable expert parameters (LoRA only)
                    for expert in self.model.expert_models:
                        trainable_params = [p for p in expert.parameters() if p.requires_grad]
                        if trainable_params:
                            self.accelerator.clip_grad_norm_(
                                trainable_params,
                                self.config.training.max_grad_norm
                            )
                    self.optimizer.step()
                    self.scheduler.step()
                    phase_losses.append(loss.item())

            # Phase 3: Joint training (optional, for fine-tuning)
            if self.epoch >= 1:  # Start joint training after first epoch
                self.optimizer.zero_grad()
                self.router_optimizer.zero_grad()
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        complexity_scores=batch['complexity_score'],
                        expert_assignments=batch['expert_assignment'],
                        training_mode="joint"
                    )

                    loss = outputs.get('loss', 0.0)
                    if loss > 0:
                        self.accelerator.backward(loss)
                        # Clip gradients
                        self.accelerator.clip_grad_norm_(
                            self.model.router.parameters(),
                            self.config.training.max_grad_norm
                        )
                        for expert in self.model.expert_models:
                            trainable_params = [p for p in expert.parameters() if p.requires_grad]
                            if trainable_params:
                                self.accelerator.clip_grad_norm_(
                                    trainable_params,
                                    self.config.training.max_grad_norm
                                )
                        self.optimizer.step()
                        self.router_optimizer.step()
                        self.scheduler.step()
                        self.router_scheduler.step()
                        phase_losses.append(loss.item())

            # Update metrics
            batch_loss = np.mean(phase_losses) if phase_losses else 0.0
            total_loss += batch_loss
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': batch_loss})

            # Log periodically
            if self.global_step % self.config.training.logging_steps == 0:
                if self.accelerator.is_main_process and self.mlflow_active:
                    try:
                        mlflow.log_metric('train_step_loss', batch_loss, step=self.global_step)
                    except Exception:
                        pass

        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}

    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch.

        Returns:
            Validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_routing_preds = []
        all_routing_targets = []
        all_complexity_preds = []
        all_complexity_targets = []
        expert_usage = {i: 0 for i in range(self.config.model.num_experts)}

        with torch.no_grad():
            for batch in tqdm(
                self.val_dataloader,
                desc="Validation",
                disable=not self.accelerator.is_main_process
            ):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    complexity_scores=batch['complexity_score'],
                    expert_assignments=batch['expert_assignment'],
                    training_mode="joint"
                )

                # Accumulate loss
                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()

                # Collect routing predictions
                if 'expert_assignments' in outputs and 'routing_logits' in outputs:
                    routing_preds = outputs['expert_assignments'].cpu().numpy()
                    routing_targets = batch['expert_assignment'].cpu().numpy()

                    all_routing_preds.extend(routing_preds)
                    all_routing_targets.extend(routing_targets)

                    # Update expert usage
                    for pred in routing_preds:
                        expert_usage[pred] += 1

                # Collect complexity predictions
                if 'complexity_pred' in outputs:
                    complexity_preds = outputs['complexity_pred'].cpu().numpy()
                    complexity_targets = batch['complexity_score'].cpu().numpy()

                    all_complexity_preds.extend(complexity_preds)
                    all_complexity_targets.extend(complexity_targets)

        # Calculate metrics
        avg_loss = total_loss / len(self.val_dataloader)

        # Routing accuracy
        routing_accuracy = 0.0
        if all_routing_preds and all_routing_targets:
            routing_accuracy = np.mean(
                np.array(all_routing_preds) == np.array(all_routing_targets)
            )

        # Complexity MSE
        complexity_mse = 0.0
        if all_complexity_preds and all_complexity_targets:
            complexity_mse = np.mean(
                (np.array(all_complexity_preds) - np.array(all_complexity_targets)) ** 2
            )

        # Normalize expert usage
        total_samples = sum(expert_usage.values())
        if total_samples > 0:
            expert_usage = {k: v / total_samples for k, v in expert_usage.items()}

        return {
            'loss': avg_loss,
            'routing_accuracy': routing_accuracy,
            'complexity_mse': complexity_mse,
            'expert_usage': expert_usage
        }

    def _evaluate_test(self) -> Dict[str, float]:
        """Evaluate on test set.

        Returns:
            Test metrics.
        """
        if not self.test_dataloader:
            return {}

        logger.info("Evaluating on test set...")
        self.model.eval()

        # Use the routing and complexity metrics classes
        all_routing_preds = []
        all_routing_targets = []
        all_complexity_preds = []
        all_complexity_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(
                self.test_dataloader,
                desc="Test Evaluation",
                disable=not self.accelerator.is_main_process
            ):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                    complexity_scores=batch['complexity_score'],
                    expert_assignments=batch['expert_assignment'],
                    training_mode="joint"
                )

                if 'loss' in outputs:
                    total_loss += outputs['loss'].item()

                if 'expert_assignments' in outputs:
                    all_routing_preds.extend(outputs['expert_assignments'].cpu().numpy())
                    all_routing_targets.extend(batch['expert_assignment'].cpu().numpy())

                if 'complexity_pred' in outputs:
                    all_complexity_preds.extend(outputs['complexity_pred'].cpu().numpy())
                    all_complexity_targets.extend(batch['complexity_score'].cpu().numpy())

        # Calculate comprehensive metrics
        test_metrics = {
            'loss': total_loss / len(self.test_dataloader)
        }

        if all_routing_preds and all_routing_targets:
            routing_results = self.routing_metrics.compute_metrics(
                np.array(all_routing_targets),
                np.array(all_routing_preds)
            )
            test_metrics.update(routing_results)

        if all_complexity_preds and all_complexity_targets:
            complexity_results = self.complexity_metrics.compute_metrics(
                np.array(all_complexity_targets),
                np.array(all_complexity_preds)
            )
            test_metrics.update({f"complexity_{k}": v for k, v in complexity_results.items()})

        return test_metrics

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best checkpoint.
        """
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if is_best:
            model_path = checkpoint_dir / "best_model"
        else:
            model_path = checkpoint_dir / f"checkpoint_epoch_{epoch}"

        # Unwrap model if using accelerator
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(str(model_path))

        # Save training state
        state_dict = {
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'router_optimizer': self.router_optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'router_scheduler': self.router_scheduler.state_dict(),
        }

        state_path = model_path / "training_state.pt"
        torch.save(state_dict, state_path)

        logger.info(f"Checkpoint saved to {model_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.
        """
        checkpoint_path = Path(checkpoint_path)

        # Load model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model = ComplexityAwareLoRARouter.from_pretrained(
            str(checkpoint_path),
            self.tokenizer
        )

        # Load training state
        state_path = checkpoint_path / "training_state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location=self.accelerator.device)

            self.epoch = state_dict['epoch']
            self.global_step = state_dict['global_step']
            self.best_val_loss = state_dict['best_val_loss']

            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.router_optimizer.load_state_dict(state_dict['router_optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.router_scheduler.load_state_dict(state_dict['router_scheduler'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def evaluate_inference_speed(self, num_samples: int = 100) -> Dict[str, float]:
        """Evaluate inference speed and latency.

        Args:
            num_samples: Number of samples to test.

        Returns:
            Inference speed metrics.
        """
        self.model.eval()

        # Prepare test data
        test_batch = next(iter(self.val_dataloader))
        input_ids = test_batch['input_ids'][:num_samples]
        attention_mask = test_batch['attention_mask'][:num_samples]
        complexity_scores = test_batch['complexity_score'][:num_samples]

        # Measure routing time
        routing_times = []
        generation_times = []

        with torch.no_grad():
            for i in range(len(input_ids)):
                sample_input = input_ids[i:i+1]
                sample_attention = attention_mask[i:i+1]
                sample_complexity = complexity_scores[i:i+1]

                # Time routing
                start_time = time.time()
                instruction_features = self.model._extract_instruction_features(sample_input)
                router_input = self.model._create_router_input(instruction_features, sample_complexity)
                router_outputs = self.model.router(router_input)
                routing_time = (time.time() - start_time) * 1000  # ms

                # Time generation
                start_time = time.time()
                generated = self.model.generate(
                    sample_input,
                    sample_attention,
                    sample_complexity,
                    max_new_tokens=50,
                    do_sample=False
                )
                generation_time = (time.time() - start_time) * 1000  # ms

                routing_times.append(routing_time)
                generation_times.append(generation_time)

        return {
            'avg_routing_latency_ms': np.mean(routing_times),
            'std_routing_latency_ms': np.std(routing_times),
            'avg_generation_latency_ms': np.mean(generation_times),
            'std_generation_latency_ms': np.std(generation_times),
            'total_latency_ms': np.mean(routing_times) + np.mean(generation_times)
        }