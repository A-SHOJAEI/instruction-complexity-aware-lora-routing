"""Configuration management utilities."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration parameters."""

    base_model_name: str = "microsoft/DialoGPT-small"
    num_experts: int = 3
    complexity_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.7])
    router_hidden_dim: int = 256
    router_dropout: float = 0.1
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    batch_size: int = 8
    learning_rate: float = 5e-5
    router_learning_rate: float = 1e-3
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4


@dataclass
class DataConfig:
    """Data configuration parameters."""

    dataset_name: str = "tatsu-lab/alpaca"
    max_length: int = 512
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1
    complexity_features: List[str] = field(default_factory=lambda: [
        "instruction_length", "response_length", "syntactic_complexity",
        "semantic_diversity", "dependency_depth"
    ])
    min_instruction_length: int = 10
    max_instruction_length: int = 1000


@dataclass
class EvaluationConfig:
    """Evaluation configuration parameters."""

    metrics: List[str] = field(default_factory=lambda: [
        "routing_accuracy", "perplexity", "bleu", "rouge_l"
    ])
    target_routing_accuracy: float = 0.85
    target_perplexity_improvement: float = 0.15
    max_inference_latency_ms: float = 5.0
    num_eval_samples: int = 1000


@dataclass
class Config:
    """Main configuration container."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Paths
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"

    # MLflow
    mlflow_experiment_name: str = "complexity-aware-lora-routing"
    mlflow_tracking_uri: Optional[str] = None

    # Hardware
    device: str = "auto"
    num_gpus: int = 1

    def __post_init__(self) -> None:
        """Post-initialization setup."""
        # Create directories
        for dir_path in [self.output_dir, self.cache_dir, self.log_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.log_dir) / "training.log"),
                logging.StreamHandler()
            ]
        )

        # Validate configuration
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.model.num_experts < 2:
            raise ValueError("Number of experts must be at least 2")

        if len(self.model.complexity_thresholds) != self.model.num_experts - 1:
            raise ValueError(
                f"Number of thresholds ({len(self.model.complexity_thresholds)}) "
                f"must be one less than number of experts ({self.model.num_experts})"
            )

        if not (0 < self.data.train_split_ratio < 1):
            raise ValueError("Train split ratio must be between 0 and 1")

        total_split = (
            self.data.train_split_ratio +
            self.data.val_split_ratio +
            self.data.test_split_ratio
        )
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError("Data split ratios must sum to 1.0")

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save configuration file.
        """
        config_dict = {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'evaluation': self.evaluation.__dict__,
            'output_dir': self.output_dir,
            'cache_dir': self.cache_dir,
            'log_dir': self.log_dir,
            'mlflow_experiment_name': self.mlflow_experiment_name,
            'mlflow_tracking_uri': self.mlflow_tracking_uri,
            'device': self.device,
            'num_gpus': self.num_gpus,
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {path}")


def load_config(path: Union[str, Path]) -> Config:
    """Load configuration from YAML file.

    Args:
        path: Path to configuration file.

    Returns:
        Loaded configuration object.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Create config with loaded values
        config = Config(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            data=DataConfig(**config_dict.get('data', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            output_dir=config_dict.get('output_dir', './outputs'),
            cache_dir=config_dict.get('cache_dir', './cache'),
            log_dir=config_dict.get('log_dir', './logs'),
            mlflow_experiment_name=config_dict.get('mlflow_experiment_name', 'complexity-aware-lora-routing'),
            mlflow_tracking_uri=config_dict.get('mlflow_tracking_uri'),
            device=config_dict.get('device', 'auto'),
            num_gpus=config_dict.get('num_gpus', 1),
        )

        logger.info(f"Configuration loaded from {path}")
        return config

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")


def get_default_config() -> Config:
    """Get default configuration.

    Returns:
        Default configuration object.
    """
    return Config()