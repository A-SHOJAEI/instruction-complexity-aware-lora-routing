"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

from src.instruction_complexity_aware_lora_routing.utils.config import Config, ModelConfig, TrainingConfig, DataConfig
from src.instruction_complexity_aware_lora_routing.data.loader import AlpacaDataset
from src.instruction_complexity_aware_lora_routing.data.preprocessing import ComplexityAnalyzer, DataStratifier


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    config = Config()
    config.model.base_model_name = "microsoft/DialoGPT-small"
    config.model.num_experts = 3
    config.model.complexity_thresholds = [0.3, 0.7]
    config.training.batch_size = 2
    config.training.num_epochs = 1
    config.data.max_length = 128
    return config


@pytest.fixture
def sample_tokenizer():
    """Create a sample tokenizer for testing."""
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_data():
    """Create sample instruction-response data."""
    return [
        {
            "instruction": "Write a simple hello world program",
            "input": "",
            "output": "print('Hello, World!')"
        },
        {
            "instruction": "Explain the concept of machine learning",
            "input": "for beginners",
            "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "What is 2+2?",
            "input": "",
            "output": "4"
        },
        {
            "instruction": "Implement a binary search algorithm",
            "input": "in Python",
            "output": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1"
        }
    ]


@pytest.fixture
def sample_dataframe(sample_data):
    """Create sample DataFrame from sample data."""
    return pd.DataFrame(sample_data)


@pytest.fixture
def sample_complexity_scores():
    """Create sample complexity scores."""
    return [0.2, 0.8, 0.1, 0.9]


@pytest.fixture
def sample_expert_assignments():
    """Create sample expert assignments."""
    return [0, 2, 0, 2]


@pytest.fixture
def sample_dataset(sample_data, sample_tokenizer, sample_complexity_scores, sample_expert_assignments):
    """Create sample AlpacaDataset."""
    return AlpacaDataset(
        data=sample_data,
        tokenizer=sample_tokenizer,
        max_length=128,
        complexity_scores=sample_complexity_scores,
        expert_assignments=sample_expert_assignments
    )


@pytest.fixture
def sample_complexity_analyzer():
    """Create sample ComplexityAnalyzer."""
    data_config = DataConfig()
    return ComplexityAnalyzer(data_config)


@pytest.fixture
def sample_data_stratifier():
    """Create sample DataStratifier."""
    return DataStratifier(num_experts=3, complexity_thresholds=[0.3, 0.7])


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")  # Use CPU for testing to avoid GPU memory issues


@pytest.fixture
def random_seed():
    """Set random seed for reproducible testing."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def small_model_config():
    """Create a small model configuration for fast testing."""
    config = ModelConfig()
    config.base_model_name = "microsoft/DialoGPT-small"
    config.num_experts = 2
    config.complexity_thresholds = [0.5]
    config.router_hidden_dim = 32
    config.lora_rank = 4
    config.lora_alpha = 8
    return config


@pytest.fixture
def small_training_config():
    """Create a small training configuration for fast testing."""
    config = TrainingConfig()
    config.batch_size = 2
    config.learning_rate = 1e-3
    config.router_learning_rate = 1e-3
    config.num_epochs = 1
    config.warmup_steps = 2
    config.gradient_accumulation_steps = 1
    config.save_steps = 10
    config.eval_steps = 5
    config.logging_steps = 2
    return config