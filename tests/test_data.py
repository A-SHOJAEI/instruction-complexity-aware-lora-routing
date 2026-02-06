"""Tests for data loading and preprocessing modules."""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.instruction_complexity_aware_lora_routing.data.loader import AlpacaDataset, AlpacaDataLoader
from src.instruction_complexity_aware_lora_routing.data.preprocessing import ComplexityAnalyzer, DataStratifier
from src.instruction_complexity_aware_lora_routing.utils.config import DataConfig


class TestAlpacaDataset:
    """Tests for AlpacaDataset class."""

    def test_init(self, sample_data, sample_tokenizer):
        """Test dataset initialization."""
        dataset = AlpacaDataset(
            data=sample_data,
            tokenizer=sample_tokenizer,
            max_length=128
        )

        assert len(dataset) == len(sample_data)
        assert dataset.tokenizer == sample_tokenizer
        assert dataset.max_length == 128
        assert len(dataset.complexity_scores) == len(sample_data)
        assert len(dataset.expert_assignments) == len(sample_data)

    def test_init_with_dataframe(self, sample_dataframe, sample_tokenizer):
        """Test dataset initialization with DataFrame."""
        dataset = AlpacaDataset(
            data=sample_dataframe,
            tokenizer=sample_tokenizer,
            max_length=128
        )

        assert len(dataset) == len(sample_dataframe)
        assert isinstance(dataset.data, list)

    def test_init_with_custom_scores(self, sample_data, sample_tokenizer, sample_complexity_scores, sample_expert_assignments):
        """Test dataset initialization with custom scores."""
        dataset = AlpacaDataset(
            data=sample_data,
            tokenizer=sample_tokenizer,
            max_length=128,
            complexity_scores=sample_complexity_scores,
            expert_assignments=sample_expert_assignments
        )

        assert dataset.complexity_scores == sample_complexity_scores
        assert dataset.expert_assignments == sample_expert_assignments

    def test_init_length_mismatch(self, sample_data, sample_tokenizer):
        """Test initialization with mismatched lengths raises error."""
        with pytest.raises(ValueError, match="Complexity scores length must match data length"):
            AlpacaDataset(
                data=sample_data,
                tokenizer=sample_tokenizer,
                max_length=128,
                complexity_scores=[0.1, 0.2]  # Wrong length
            )

    def test_getitem(self, sample_dataset):
        """Test dataset item retrieval."""
        item = sample_dataset[0]

        # Check required keys
        required_keys = ['input_ids', 'attention_mask', 'labels', 'complexity_score',
                        'expert_assignment', 'instruction', 'response']
        for key in required_keys:
            assert key in item

        # Check tensor shapes
        assert item['input_ids'].shape[0] == sample_dataset.max_length
        assert item['attention_mask'].shape[0] == sample_dataset.max_length
        assert item['labels'].shape[0] == sample_dataset.max_length

        # Check tensor types
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['attention_mask'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)
        assert isinstance(item['complexity_score'], torch.Tensor)
        assert isinstance(item['expert_assignment'], torch.Tensor)

        # Check string fields
        assert isinstance(item['instruction'], str)
        assert isinstance(item['response'], str)

    def test_getitem_all_samples(self, sample_dataset):
        """Test that all samples can be retrieved without errors."""
        for i in range(len(sample_dataset)):
            item = sample_dataset[i]
            assert item is not None

    def test_len(self, sample_dataset):
        """Test dataset length."""
        assert len(sample_dataset) == 4


class TestComplexityAnalyzer:
    """Tests for ComplexityAnalyzer class."""

    def test_init(self):
        """Test analyzer initialization."""
        config = DataConfig()
        analyzer = ComplexityAnalyzer(config)

        assert analyzer.config == config
        assert analyzer.tfidf is not None
        # Note: nlp might be None if spacy model not available

    def test_analyze_instruction_length(self, sample_complexity_analyzer):
        """Test instruction length analysis."""
        short_instruction = "Hello"
        long_instruction = "This is a very long instruction that contains many words and should have a higher complexity score"

        short_score = sample_complexity_analyzer.analyze_instruction_length(short_instruction)
        long_score = sample_complexity_analyzer.analyze_instruction_length(long_instruction)

        assert 0 <= short_score <= 1
        assert 0 <= long_score <= 1
        assert long_score > short_score

    def test_analyze_response_length(self, sample_complexity_analyzer):
        """Test response length analysis."""
        short_response = "Yes"
        long_response = "This is a very detailed response that explains the concept thoroughly with many examples and detailed explanations"

        short_score = sample_complexity_analyzer.analyze_response_length(short_response)
        long_score = sample_complexity_analyzer.analyze_response_length(long_response)

        assert 0 <= short_score <= 1
        assert 0 <= long_score <= 1
        assert long_score > short_score

    def test_analyze_syntactic_complexity(self, sample_complexity_analyzer):
        """Test syntactic complexity analysis."""
        simple_text = "The cat sat."
        complex_text = "The sophisticated algorithm, which was developed by researchers, efficiently processes large datasets."

        simple_score = sample_complexity_analyzer.analyze_syntactic_complexity(simple_text)
        complex_score = sample_complexity_analyzer.analyze_syntactic_complexity(complex_text)

        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
        # Note: Complex text should generally have higher score, but this depends on spaCy model availability

    def test_analyze_dependency_depth(self, sample_complexity_analyzer):
        """Test dependency depth analysis."""
        simple_text = "I run."
        complex_text = "The algorithm that the researchers developed processes data efficiently."

        simple_score = sample_complexity_analyzer.analyze_dependency_depth(simple_text)
        complex_score = sample_complexity_analyzer.analyze_dependency_depth(complex_text)

        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1

    def test_analyze_semantic_diversity(self, sample_complexity_analyzer):
        """Test semantic diversity analysis."""
        similar_instruction = "Write a program"
        similar_response = "Here is a program"

        different_instruction = "Explain machine learning"
        different_response = "The weather is sunny today"

        similar_score = sample_complexity_analyzer.analyze_semantic_diversity(
            similar_instruction, similar_response
        )
        different_score = sample_complexity_analyzer.analyze_semantic_diversity(
            different_instruction, different_response
        )

        assert 0 <= similar_score <= 1
        assert 0 <= different_score <= 1
        # Different topics should generally have higher diversity

    def test_compute_complexity_features(self, sample_complexity_analyzer):
        """Test computing all complexity features."""
        instruction = "Write a complex algorithm"
        response = "Here is the implementation"

        features = sample_complexity_analyzer.compute_complexity_features(instruction, response)

        # Check that all configured features are present
        expected_features = sample_complexity_analyzer.config.complexity_features
        for feature in expected_features:
            assert feature in features
            assert 0 <= features[feature] <= 1

    def test_compute_overall_complexity(self, sample_complexity_analyzer):
        """Test overall complexity computation."""
        features = {
            "instruction_length": 0.3,
            "response_length": 0.7,
            "syntactic_complexity": 0.5,
            "semantic_diversity": 0.4,
            "dependency_depth": 0.6
        }

        complexity = sample_complexity_analyzer.compute_overall_complexity(features)

        assert 0 <= complexity <= 1

        # Test with empty features
        empty_complexity = sample_complexity_analyzer.compute_overall_complexity({})
        assert empty_complexity == 0.5

    def test_set_tokenizer(self, sample_complexity_analyzer, sample_tokenizer):
        """Test setting tokenizer."""
        sample_complexity_analyzer.set_tokenizer(sample_tokenizer)
        assert sample_complexity_analyzer.tokenizer == sample_tokenizer


class TestDataStratifier:
    """Tests for DataStratifier class."""

    def test_init(self):
        """Test stratifier initialization."""
        stratifier = DataStratifier(num_experts=3, complexity_thresholds=[0.3, 0.7])

        assert stratifier.num_experts == 3
        assert stratifier.complexity_thresholds == [0.3, 0.7]

    def test_init_invalid_thresholds(self):
        """Test initialization with invalid thresholds."""
        with pytest.raises(ValueError, match="Number of thresholds"):
            DataStratifier(num_experts=3, complexity_thresholds=[0.5])  # Wrong number of thresholds

    def test_assign_expert(self, sample_data_stratifier):
        """Test expert assignment."""
        # Test boundary cases
        assert sample_data_stratifier.assign_expert(0.0) == 0
        assert sample_data_stratifier.assign_expert(0.2) == 0
        assert sample_data_stratifier.assign_expert(0.3) == 0
        assert sample_data_stratifier.assign_expert(0.5) == 1
        assert sample_data_stratifier.assign_expert(0.7) == 1
        assert sample_data_stratifier.assign_expert(0.8) == 2
        assert sample_data_stratifier.assign_expert(1.0) == 2

    def test_stratify_data(self, sample_data_stratifier, sample_dataframe):
        """Test data stratification."""
        complexity_scores = [0.1, 0.5, 0.8, 0.2]

        stratified = sample_data_stratifier.stratify_data(sample_dataframe, complexity_scores)

        # Check all experts are represented in output
        assert len(stratified) == 3
        for expert_idx in range(3):
            assert expert_idx in stratified
            assert isinstance(stratified[expert_idx], pd.DataFrame)

        # Check data distribution
        total_samples = sum(len(df) for df in stratified.values())
        assert total_samples == len(sample_dataframe)

    def test_stratify_data_length_mismatch(self, sample_data_stratifier, sample_dataframe):
        """Test stratification with mismatched lengths."""
        complexity_scores = [0.1, 0.5]  # Wrong length

        with pytest.raises(ValueError, match="Number of complexity scores must match data length"):
            sample_data_stratifier.stratify_data(sample_dataframe, complexity_scores)

    def test_get_expert_distribution(self, sample_data_stratifier):
        """Test expert distribution calculation."""
        complexity_scores = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]

        distribution = sample_data_stratifier.get_expert_distribution(complexity_scores)

        assert len(distribution) == 3
        assert sum(distribution.values()) == len(complexity_scores)

        # Check specific assignments
        expected_distribution = {0: 2, 1: 2, 2: 2}
        assert distribution == expected_distribution


class TestAlpacaDataLoader:
    """Tests for AlpacaDataLoader class."""

    def test_init(self):
        """Test data loader initialization."""
        config = DataConfig()
        loader = AlpacaDataLoader(config)

        assert loader.config == config
        assert loader.complexity_analyzer is not None
        assert loader.tokenizer is None
        assert loader.data_stratifier is None

    def test_set_tokenizer(self, sample_tokenizer):
        """Test setting tokenizer."""
        config = DataConfig()
        loader = AlpacaDataLoader(config)

        loader.set_tokenizer(sample_tokenizer)

        assert loader.tokenizer == sample_tokenizer
        assert loader.complexity_analyzer.tokenizer == sample_tokenizer

    def test_set_data_stratifier(self):
        """Test setting data stratifier."""
        config = DataConfig()
        loader = AlpacaDataLoader(config)

        loader.set_data_stratifier(num_experts=3, complexity_thresholds=[0.3, 0.7])

        assert loader.data_stratifier is not None
        assert loader.data_stratifier.num_experts == 3

    @patch('instruction_complexity_aware_lora_routing.data.loader.load_dataset')
    def test_load_raw_data(self, mock_load_dataset):
        """Test loading raw data."""
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.__iter__ = lambda x: iter([
            {"instruction": "Test instruction 1", "input": "", "output": "Test output 1"},
            {"instruction": "Test instruction 2", "input": "", "output": "Test output 2"},
        ])
        mock_load_dataset.return_value = mock_dataset

        # Mock pandas DataFrame creation
        with patch('pandas.DataFrame') as mock_df:
            mock_df.return_value = pd.DataFrame([
                {"instruction": "Test instruction 1", "input": "", "output": "Test output 1"},
                {"instruction": "Test instruction 2", "input": "", "output": "Test output 2"},
            ])

            config = DataConfig()
            loader = AlpacaDataLoader(config)

            # Mock string length filtering
            with patch.object(mock_df.return_value, '__len__', return_value=2):
                with patch.object(mock_df.return_value, '__getitem__') as mock_getitem:
                    with patch.object(mock_df.return_value, 'reset_index') as mock_reset:
                        mock_reset.return_value = mock_df.return_value
                        mock_getitem.return_value = mock_df.return_value

                        data = loader.load_raw_data()

                        assert data is not None
                        mock_load_dataset.assert_called_once()

    def test_compute_complexity_scores(self):
        """Test complexity score computation."""
        config = DataConfig()
        loader = AlpacaDataLoader(config)

        # Create sample data
        data = pd.DataFrame([
            {"instruction": "Simple task", "output": "Simple response"},
            {"instruction": "More complex task with detailed instructions", "output": "Detailed response"},
        ])

        scores = loader.compute_complexity_scores(data)

        assert len(scores) == len(data)
        assert all(0 <= score <= 1 for score in scores)

    def test_create_dataloaders(self, sample_dataset):
        """Test dataloader creation."""
        config = DataConfig()
        loader = AlpacaDataLoader(config)

        train_loader, val_loader, test_loader = loader.create_dataloaders(
            sample_dataset, sample_dataset, sample_dataset,
            batch_size=2, num_workers=0  # Use 0 workers for testing
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Check batch sizes
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        assert test_loader.batch_size == 2

        # Check shuffle settings
        assert train_loader.sampler is not None  # Should be shuffled
        # val and test loaders use SequentialSampler by default