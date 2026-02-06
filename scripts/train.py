#!/usr/bin/env python3
"""Training script for complexity-aware LoRA routing."""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoTokenizer
import mlflow

from instruction_complexity_aware_lora_routing.utils.config import Config, load_config
from instruction_complexity_aware_lora_routing.data.loader import AlpacaDataLoader
from instruction_complexity_aware_lora_routing.models.model import ComplexityAwareLoRARouter
from instruction_complexity_aware_lora_routing.training.trainer import LoRATrainer

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train complexity-aware LoRA routing model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Base model name (overrides config)"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Training batch size (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        help="Number of expert models (overrides config)"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run (no actual training)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    return parser.parse_args()


def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration.

    Args:
        debug: Whether to enable debug logging.
    """
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


def override_config(config: Config, args: argparse.Namespace) -> Config:
    """Override configuration with command line arguments.

    Args:
        config: Base configuration.
        args: Command line arguments.

    Returns:
        Updated configuration.
    """
    if args.output_dir:
        config.output_dir = args.output_dir

    if args.model_name:
        config.model.base_model_name = args.model_name

    if args.num_epochs:
        config.training.num_epochs = args.num_epochs

    if args.batch_size:
        config.training.batch_size = args.batch_size

    if args.learning_rate:
        config.training.learning_rate = args.learning_rate

    if args.num_experts:
        config.model.num_experts = args.num_experts
        # Adjust thresholds for new number of experts
        config.model.complexity_thresholds = [
            (i + 1) / config.model.num_experts
            for i in range(config.model.num_experts - 1)
        ]

    return config


def validate_environment() -> None:
    """Validate training environment and dependencies."""
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        logger.warning("CUDA not available, using CPU")

    # Check MLflow
    try:
        mlflow.get_tracking_uri()
        logger.info("MLflow tracking available")
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")


def create_model_and_data(config: Config) -> tuple:
    """Create model and data loaders.

    Args:
        config: Configuration object.

    Returns:
        Tuple of (model, tokenizer, train_loader, val_loader, test_loader).
    """
    # Initialize tokenizer
    logger.info(f"Loading tokenizer: {config.model.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name,
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize data loader
    logger.info("Setting up data loader...")
    data_loader = AlpacaDataLoader(config.data)
    data_loader.set_tokenizer(tokenizer)
    data_loader.set_data_stratifier(
        config.model.num_experts,
        config.model.complexity_thresholds
    )

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config.training.batch_size,
        config.training.dataloader_num_workers
    )

    # Initialize model
    logger.info("Creating model...")
    model = ComplexityAwareLoRARouter(
        config.model,
        config.model.base_model_name,
        tokenizer
    )

    logger.info(f"Model created with {config.model.num_experts} experts")
    logger.info(f"Training data: {len(train_dataset)} samples")
    logger.info(f"Validation data: {len(val_dataset)} samples")
    logger.info(f"Test data: {len(test_dataset)} samples")

    return model, tokenizer, train_loader, val_loader, test_loader


def main() -> None:
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.debug)

    logger.info("Starting complexity-aware LoRA routing training...")
    logger.info(f"Arguments: {vars(args)}")

    # Validate environment
    validate_environment()

    try:
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            config = Config()
            logger.info("Using default configuration")

        # Override with command line arguments
        config = override_config(config, args)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Save final configuration
        config_save_path = Path(config.output_dir) / "training_config.yaml"
        config.save(config_save_path)
        logger.info(f"Training configuration saved to {config_save_path}")

        if args.dry_run:
            logger.info("Dry run mode - exiting without training")
            return

        # Create model and data
        model, tokenizer, train_loader, val_loader, test_loader = create_model_and_data(config)

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = LoRATrainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader
        )

        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming training from {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)

        # Start training
        logger.info("Starting training...")
        training_history = trainer.train()

        # Log final results
        logger.info("Training completed successfully!")
        logger.info(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
        logger.info(f"Best routing accuracy: {max(training_history['routing_accuracy']):.4f}")

        # Evaluate inference speed
        logger.info("Evaluating inference speed...")
        speed_metrics = trainer.evaluate_inference_speed()
        logger.info(f"Average routing latency: {speed_metrics['avg_routing_latency_ms']:.2f}ms")
        logger.info(f"Average generation latency: {speed_metrics['avg_generation_latency_ms']:.2f}ms")

        # Save final model
        final_model_path = Path(config.output_dir) / "final_model"
        model.save_pretrained(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")

        # Check target metrics
        final_routing_accuracy = training_history['routing_accuracy'][-1]
        target_routing_accuracy = config.evaluation.target_routing_accuracy

        if final_routing_accuracy >= target_routing_accuracy:
            logger.info(f"✓ Target routing accuracy achieved: "
                       f"{final_routing_accuracy:.4f} >= {target_routing_accuracy:.4f}")
        else:
            logger.warning(f"✗ Target routing accuracy not achieved: "
                          f"{final_routing_accuracy:.4f} < {target_routing_accuracy:.4f}")

        # Check latency target
        total_latency = speed_metrics['total_latency_ms']
        target_latency = config.evaluation.max_inference_latency_ms

        if total_latency <= target_latency:
            logger.info(f"✓ Target inference latency achieved: "
                       f"{total_latency:.2f}ms <= {target_latency:.2f}ms")
        else:
            logger.warning(f"✗ Target inference latency not achieved: "
                          f"{total_latency:.2f}ms > {target_latency:.2f}ms")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()