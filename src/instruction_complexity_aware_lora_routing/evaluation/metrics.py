"""Evaluation metrics for complexity-aware LoRA routing."""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

logger = logging.getLogger(__name__)


class RoutingMetrics:
    """Metrics for evaluating routing accuracy and expert utilization."""

    def __init__(self, num_experts: Optional[int] = None):
        """Initialize routing metrics.

        Args:
            num_experts: Number of expert models.
        """
        self.num_experts = num_experts

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute routing accuracy.

        Args:
            y_true: True expert assignments.
            y_pred: Predicted expert assignments.

        Returns:
            Routing accuracy.
        """
        return accuracy_score(y_true, y_pred)

    def compute_precision_recall_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute precision, recall, and F1 scores.

        Args:
            y_true: True expert assignments.
            y_pred: Predicted expert assignments.

        Returns:
            Dictionary with precision, recall, and F1 scores.
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Also compute macro and micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )

        return {
            'precision_per_expert': precision,
            'recall_per_expert': recall,
            'f1_per_expert': f1,
            'support_per_expert': support,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
        }

    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """Compute confusion matrix.

        Args:
            y_true: True expert assignments.
            y_pred: Predicted expert assignments.
            normalize: Normalization method ('true', 'pred', 'all', or None).

        Returns:
            Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred, normalize=normalize)

    def compute_expert_utilization(self, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute expert utilization statistics.

        Args:
            y_pred: Predicted expert assignments.

        Returns:
            Expert utilization statistics.
        """
        unique, counts = np.unique(y_pred, return_counts=True)
        total_samples = len(y_pred)

        utilization = np.zeros(self.num_experts or max(unique) + 1)
        for expert, count in zip(unique, counts):
            utilization[expert] = count / total_samples

        # Compute load balancing metrics
        ideal_utilization = 1.0 / len(utilization)
        load_balance_variance = np.var(utilization)
        load_balance_gini = self._gini_coefficient(utilization)

        return {
            'utilization_per_expert': utilization,
            'ideal_utilization': ideal_utilization,
            'load_balance_variance': load_balance_variance,
            'load_balance_gini': load_balance_gini,
            'most_used_expert': np.argmax(utilization),
            'least_used_expert': np.argmin(utilization),
            'utilization_entropy': self._entropy(utilization),
        }

    def _gini_coefficient(self, x: np.ndarray) -> float:
        """Compute Gini coefficient for load balancing.

        Args:
            x: Utilization array.

        Returns:
            Gini coefficient.
        """
        # Sort array
        sorted_x = np.sort(x)
        n = len(x)

        # Compute Gini
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_x)) / (n * np.sum(sorted_x))

        return gini

    def _entropy(self, x: np.ndarray) -> float:
        """Compute entropy of utilization distribution.

        Args:
            x: Utilization array.

        Returns:
            Entropy value.
        """
        # Add small epsilon to avoid log(0)
        x_norm = x + 1e-8
        x_norm = x_norm / np.sum(x_norm)
        return -np.sum(x_norm * np.log(x_norm))

    def compute_routing_confidence(self, routing_probs: np.ndarray) -> Dict[str, float]:
        """Compute routing confidence metrics.

        Args:
            routing_probs: Routing probability distributions [n_samples, n_experts].

        Returns:
            Confidence metrics.
        """
        # Max probability (confidence)
        max_probs = np.max(routing_probs, axis=1)
        avg_confidence = np.mean(max_probs)

        # Entropy (uncertainty)
        entropy = -np.sum(routing_probs * np.log(routing_probs + 1e-8), axis=1)
        avg_entropy = np.mean(entropy)

        # Margin (difference between top 2 probabilities)
        sorted_probs = np.sort(routing_probs, axis=1)
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
        avg_margin = np.mean(margins)

        return {
            'avg_confidence': avg_confidence,
            'avg_entropy': avg_entropy,
            'avg_margin': avg_margin,
            'confidence_std': np.std(max_probs),
            'entropy_std': np.std(entropy),
            'margin_std': np.std(margins),
        }

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        routing_probs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive routing metrics.

        Args:
            y_true: True expert assignments.
            y_pred: Predicted expert assignments.
            routing_probs: Optional routing probabilities.

        Returns:
            Dictionary of all routing metrics.
        """
        metrics = {}

        # Basic accuracy
        metrics['routing_accuracy'] = self.compute_accuracy(y_true, y_pred)

        # Precision, recall, F1
        prf_metrics = self.compute_precision_recall_f1(y_true, y_pred)
        metrics.update(prf_metrics)

        # Confusion matrix
        metrics['confusion_matrix'] = self.compute_confusion_matrix(y_true, y_pred)

        # Expert utilization
        util_metrics = self.compute_expert_utilization(y_pred)
        metrics.update(util_metrics)

        # Routing confidence (if probabilities provided)
        if routing_probs is not None:
            conf_metrics = self.compute_routing_confidence(routing_probs)
            metrics.update(conf_metrics)

        return metrics

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 6)
    ) -> None:
        """Plot confusion matrix.

        Args:
            y_true: True expert assignments.
            y_pred: Predicted expert assignments.
            save_path: Optional path to save the plot.
            figsize: Figure size.
        """
        cm = self.compute_confusion_matrix(y_true, y_pred, normalize='true')

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[f'Expert {i}' for i in range(cm.shape[1])],
            yticklabels=[f'Expert {i}' for i in range(cm.shape[0])]
        )
        plt.title('Routing Confusion Matrix')
        plt.xlabel('Predicted Expert')
        plt.ylabel('True Expert')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_expert_utilization(
        self,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot expert utilization statistics.

        Args:
            y_pred: Predicted expert assignments.
            save_path: Optional path to save the plot.
            figsize: Figure size.
        """
        util_metrics = self.compute_expert_utilization(y_pred)
        utilization = util_metrics['utilization_per_expert']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Bar plot of utilization
        experts = [f'Expert {i}' for i in range(len(utilization))]
        ax1.bar(experts, utilization)
        ax1.axhline(y=util_metrics['ideal_utilization'], color='r', linestyle='--',
                   label=f'Ideal ({util_metrics["ideal_utilization"]:.3f})')
        ax1.set_title('Expert Utilization')
        ax1.set_ylabel('Utilization Rate')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)

        # Pie chart
        ax2.pie(utilization, labels=experts, autopct='%1.1f%%')
        ax2.set_title('Expert Usage Distribution')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class ComplexityMetrics:
    """Metrics for evaluating complexity prediction accuracy."""

    def __init__(self):
        """Initialize complexity metrics."""
        pass

    def compute_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute regression metrics for complexity prediction.

        Args:
            y_true: True complexity scores.
            y_pred: Predicted complexity scores.

        Returns:
            Dictionary of regression metrics.
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1],
        }

    def compute_binned_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_bins: int = 5
    ) -> Dict[str, Any]:
        """Compute accuracy within complexity bins.

        Args:
            y_true: True complexity scores.
            y_pred: Predicted complexity scores.
            num_bins: Number of complexity bins.

        Returns:
            Binned accuracy metrics.
        """
        # Create bins
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Assign samples to bins
        true_bins = np.digitize(y_true, bin_edges) - 1
        pred_bins = np.digitize(y_pred, bin_edges) - 1

        # Clip to valid range
        true_bins = np.clip(true_bins, 0, num_bins - 1)
        pred_bins = np.clip(pred_bins, 0, num_bins - 1)

        # Compute accuracy per bin
        bin_accuracies = []
        bin_counts = []

        for bin_idx in range(num_bins):
            bin_mask = true_bins == bin_idx
            if bin_mask.sum() > 0:
                bin_accuracy = np.mean(true_bins[bin_mask] == pred_bins[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(bin_mask.sum())
            else:
                bin_accuracies.append(0.0)
                bin_counts.append(0)

        return {
            'bin_edges': bin_edges,
            'bin_centers': bin_centers,
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts),
            'overall_bin_accuracy': np.mean(true_bins == pred_bins),
        }

    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        num_bins: int = 10
    ) -> Dict[str, Any]:
        """Compute calibration metrics for complexity predictions.

        Args:
            y_true: True complexity scores.
            y_pred: Predicted complexity scores.
            num_bins: Number of calibration bins.

        Returns:
            Calibration metrics.
        """
        # Sort predictions
        sorted_indices = np.argsort(y_pred)
        sorted_true = y_true[sorted_indices]
        sorted_pred = y_pred[sorted_indices]

        # Create bins of equal size
        bin_size = len(y_pred) // num_bins
        bin_boundaries = []
        bin_lowers = []
        bin_uppers = []
        bin_accuracies = []
        bin_confidences = []

        for i in range(num_bins):
            start_idx = i * bin_size
            if i == num_bins - 1:  # Last bin gets remaining samples
                end_idx = len(y_pred)
            else:
                end_idx = (i + 1) * bin_size

            bin_true = sorted_true[start_idx:end_idx]
            bin_pred = sorted_pred[start_idx:end_idx]

            if len(bin_true) > 0:
                bin_lower = np.min(bin_pred)
                bin_upper = np.max(bin_pred)
                bin_accuracy = np.mean(bin_true)  # Average true complexity in bin
                bin_confidence = np.mean(bin_pred)  # Average predicted complexity in bin

                bin_boundaries.append((bin_lower, bin_upper))
                bin_lowers.append(bin_lower)
                bin_uppers.append(bin_upper)
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)

        # Compute Expected Calibration Error (ECE)
        ece = 0.0
        total_samples = len(y_pred)

        for i in range(len(bin_accuracies)):
            bin_count = bin_size if i < num_bins - 1 else total_samples - i * bin_size
            ece += (bin_count / total_samples) * abs(bin_confidences[i] - bin_accuracies[i])

        return {
            'bin_boundaries': bin_boundaries,
            'bin_lowers': bin_lowers,
            'bin_uppers': bin_uppers,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'expected_calibration_error': ece,
        }

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Compute comprehensive complexity metrics.

        Args:
            y_true: True complexity scores.
            y_pred: Predicted complexity scores.

        Returns:
            Dictionary of all complexity metrics.
        """
        metrics = {}

        # Regression metrics
        reg_metrics = self.compute_regression_metrics(y_true, y_pred)
        metrics.update(reg_metrics)

        # Binned accuracy
        binned_metrics = self.compute_binned_accuracy(y_true, y_pred)
        metrics.update(binned_metrics)

        # Calibration metrics
        calib_metrics = self.compute_calibration_metrics(y_true, y_pred)
        metrics.update(calib_metrics)

        return metrics

    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> None:
        """Plot scatter plot of true vs predicted complexity.

        Args:
            y_true: True complexity scores.
            y_pred: Predicted complexity scores.
            save_path: Optional path to save the plot.
            figsize: Figure size.
        """
        plt.figure(figsize=figsize)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
        plt.xlabel('True Complexity')
        plt.ylabel('Predicted Complexity')
        plt.title('Complexity Prediction Scatter Plot')
        plt.legend()

        # Add R² score to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_calibration(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot calibration curve.

        Args:
            y_true: True complexity scores.
            y_pred: Predicted complexity scores.
            save_path: Optional path to save the plot.
            figsize: Figure size.
        """
        calib_metrics = self.compute_calibration_metrics(y_true, y_pred)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Calibration curve
        ax1.plot(calib_metrics['bin_confidences'], calib_metrics['bin_accuracies'], 'o-', label='Calibration')
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
        ax1.set_xlabel('Mean Predicted Complexity')
        ax1.set_ylabel('Mean True Complexity')
        ax1.set_title(f'Calibration Curve (ECE: {calib_metrics["expected_calibration_error"]:.3f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.hist(y_pred, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Predicted Complexity')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Predictions')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class GenerationMetrics:
    """Metrics for evaluating generation quality."""

    def __init__(self):
        """Initialize generation metrics."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def compute_bleu_score(
        self,
        references: List[str],
        candidates: List[str],
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    ) -> Dict[str, float]:
        """Compute BLEU scores.

        Args:
            references: Reference texts.
            candidates: Generated texts.
            weights: N-gram weights for BLEU.

        Returns:
            BLEU score metrics.
        """
        bleu_scores = []

        for ref, cand in zip(references, candidates):
            # Tokenize
            ref_tokens = nltk.word_tokenize(ref.lower())
            cand_tokens = nltk.word_tokenize(cand.lower())

            # Compute BLEU
            bleu = sentence_bleu([ref_tokens], cand_tokens, weights=weights)
            bleu_scores.append(bleu)

        return {
            'bleu_mean': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores),
            'bleu_scores': bleu_scores,
        }

    def compute_rouge_scores(
        self,
        references: List[str],
        candidates: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores.

        Args:
            references: Reference texts.
            candidates: Generated texts.

        Returns:
            ROUGE score metrics.
        """
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for ref, cand in zip(references, candidates):
            scores = self.rouge_scorer.score(ref, cand)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)

        # Compute averages
        results = {}
        for metric, scores in rouge_scores.items():
            results[f'{metric}_mean'] = np.mean(scores)
            results[f'{metric}_std'] = np.std(scores)

        return results

    def compute_perplexity(
        self,
        model,
        tokenizer: AutoTokenizer,
        texts: List[str],
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """Compute perplexity of generated texts.

        Args:
            model: Language model for perplexity calculation.
            tokenizer: Tokenizer instance.
            texts: List of texts to evaluate.
            device: Device to use for computation.

        Returns:
            Perplexity metrics.
        """
        model.eval()
        perplexities = []

        with torch.no_grad():
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(device)

                # Compute loss
                outputs = model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss.item()

                # Convert to perplexity
                perplexity = torch.exp(torch.tensor(loss)).item()
                perplexities.append(perplexity)

        return {
            'perplexity_mean': np.mean(perplexities),
            'perplexity_std': np.std(perplexities),
            'perplexity_median': np.median(perplexities),
            'perplexities': perplexities,
        }

    def compute_diversity_metrics(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer
    ) -> Dict[str, float]:
        """Compute diversity metrics for generated texts.

        Args:
            texts: List of generated texts.
            tokenizer: Tokenizer instance.

        Returns:
            Diversity metrics.
        """
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = tokenizer.tokenize(text)
            all_tokens.extend(tokens)

        # Compute distinct-n metrics
        def distinct_n(tokens: List[str], n: int) -> float:
            ngrams = set()
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.add(ngram)
            return len(ngrams) / max(len(tokens) - n + 1, 1)

        distinct_1 = distinct_n(all_tokens, 1)
        distinct_2 = distinct_n(all_tokens, 2)
        distinct_3 = distinct_n(all_tokens, 3)

        # Compute self-similarity
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                # Simple Jaccard similarity
                tokens_i = set(tokenizer.tokenize(texts[i]))
                tokens_j = set(tokenizer.tokenize(texts[j]))

                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)

                similarity = intersection / max(union, 1)
                similarities.append(similarity)

        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'distinct_3': distinct_3,
            'avg_self_similarity': np.mean(similarities) if similarities else 0.0,
            'vocabulary_size': len(set(all_tokens)),
            'total_tokens': len(all_tokens),
        }