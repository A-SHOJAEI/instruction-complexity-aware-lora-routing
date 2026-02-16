#!/usr/bin/env python3
"""Evaluation script for complexity-aware LoRA routing."""

import argparse
import logging
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm

from src.instruction_complexity_aware_lora_routing.utils.config import Config, load_config
from src.instruction_complexity_aware_lora_routing.data.loader import AlpacaDataLoader
from src.instruction_complexity_aware_lora_routing.models.model import ComplexityAwareLoRARouter
from src.instruction_complexity_aware_lora_routing.evaluation.metrics import (
    RoutingMetrics,
    ComplexityMetrics,
    GenerationMetrics
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate complexity-aware LoRA routing model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
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
        default="./evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--eval-split",
        type=str,
        choices=["test", "val", "train"],
        default="test",
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to evaluate (None for all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--generate-examples",
        action="store_true",
        help="Generate example outputs for qualitative analysis"
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to generate"
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=100,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--compute-perplexity",
        action="store_true",
        help="Compute perplexity metrics (requires base model)"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save all predictions to file"
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


def load_model_and_data(
    model_path: str,
    config: Config,
    eval_split: str,
    batch_size: int,
    num_samples: int = None
) -> tuple:
    """Load model and evaluation data.

    Args:
        model_path: Path to trained model.
        config: Configuration object.
        eval_split: Dataset split to evaluate on.
        batch_size: Batch size for evaluation.
        num_samples: Number of samples to evaluate.

    Returns:
        Tuple of (model, tokenizer, dataloader, dataset).
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.base_model_name,
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = ComplexityAwareLoRARouter.from_pretrained(model_path, tokenizer)
    model.eval()

    # Load data
    data_loader = AlpacaDataLoader(config.data)
    data_loader.set_tokenizer(tokenizer)
    data_loader.set_data_stratifier(
        config.model.num_experts,
        config.model.complexity_thresholds
    )

    # Create datasets
    train_dataset, val_dataset, test_dataset = data_loader.create_datasets()

    # Select evaluation dataset
    if eval_split == "train":
        eval_dataset = train_dataset
    elif eval_split == "val":
        eval_dataset = val_dataset
    else:  # test
        eval_dataset = test_dataset

    # Subsample if requested
    if num_samples and len(eval_dataset) > num_samples:
        indices = np.random.choice(len(eval_dataset), num_samples, replace=False)
        eval_dataset = torch.utils.data.Subset(eval_dataset, indices)

    # Create dataloader
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Loaded model and {len(eval_dataset)} {eval_split} samples")

    return model, tokenizer, eval_dataloader, eval_dataset


def evaluate_routing_and_complexity(
    model: ComplexityAwareLoRARouter,
    dataloader: torch.utils.data.DataLoader
) -> Dict[str, Any]:
    """Evaluate routing and complexity prediction performance.

    Args:
        model: Trained model.
        dataloader: Evaluation dataloader.

    Returns:
        Dictionary of evaluation metrics.
    """
    model.eval()
    device = next(model.parameters()).device

    all_routing_preds = []
    all_routing_probs = []
    all_routing_targets = []
    all_complexity_preds = []
    all_complexity_targets = []
    all_losses = []

    logger.info("Evaluating routing and complexity prediction...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                complexity_scores=batch['complexity_score'],
                expert_assignments=batch['expert_assignment'],
                training_mode="joint"
            )

            # Collect predictions
            if 'expert_assignments' in outputs:
                all_routing_preds.extend(outputs['expert_assignments'].cpu().numpy())
            if 'routing_weights' in outputs:
                all_routing_probs.extend(outputs['routing_weights'].cpu().numpy())
            if 'complexity_pred' in outputs:
                all_complexity_preds.extend(outputs['complexity_pred'].cpu().numpy())

            # Collect targets
            all_routing_targets.extend(batch['expert_assignment'].cpu().numpy())
            all_complexity_targets.extend(batch['complexity_score'].cpu().numpy())

            # Collect loss
            if 'loss' in outputs:
                all_losses.append(outputs['loss'].item())

    # Compute routing metrics
    routing_metrics = RoutingMetrics(num_experts=model.num_experts)
    routing_results = routing_metrics.compute_metrics(
        np.array(all_routing_targets),
        np.array(all_routing_preds),
        np.array(all_routing_probs) if all_routing_probs else None
    )

    # Compute complexity metrics
    complexity_metrics = ComplexityMetrics()
    complexity_results = complexity_metrics.compute_metrics(
        np.array(all_complexity_targets),
        np.array(all_complexity_preds)
    )

    # Combine results
    results = {
        'average_loss': np.mean(all_losses) if all_losses else 0.0,
        'routing': routing_results,
        'complexity': complexity_results,
    }

    return results


def generate_examples(
    model: ComplexityAwareLoRARouter,
    tokenizer: AutoTokenizer,
    dataset: torch.utils.data.Dataset,
    num_examples: int,
    max_length: int
) -> List[Dict[str, Any]]:
    """Generate example outputs for qualitative analysis.

    Args:
        model: Trained model.
        tokenizer: Tokenizer instance.
        dataset: Evaluation dataset.
        num_examples: Number of examples to generate.
        max_length: Maximum generation length.

    Returns:
        List of example dictionaries.
    """
    model.eval()
    device = next(model.parameters()).device

    # Select random examples
    indices = np.random.choice(len(dataset), min(num_examples, len(dataset)), replace=False)
    examples = []

    logger.info(f"Generating {len(indices)} examples...")

    with torch.no_grad():
        for idx in tqdm(indices, desc="Generating"):
            sample = dataset[idx]

            # Prepare inputs
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            complexity_score = sample['complexity_score'].unsqueeze(0).to(device)

            # Get routing decision
            instruction_features = model._extract_instruction_features(input_ids)
            router_input = model._create_router_input(instruction_features, complexity_score)
            router_outputs = model.router(router_input)

            predicted_expert = router_outputs['expert_assignments'][0].item()
            true_expert = sample['expert_assignment'].item()
            predicted_complexity = router_outputs['complexity_pred'][0].item()
            true_complexity = sample['complexity_score'].item()
            routing_weights = router_outputs['routing_weights'][0].cpu().numpy()

            # Generate response
            generated_ids = model.generate(
                input_ids,
                attention_mask,
                complexity_score,
                max_length=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

            # Decode texts
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Remove input from generated text
            if generated_text.startswith(input_text):
                generated_response = generated_text[len(input_text):].strip()
            else:
                generated_response = generated_text

            # Extract instruction and expected response
            instruction = sample['instruction']
            expected_response = sample['response']

            examples.append({
                'instruction': instruction,
                'expected_response': expected_response,
                'generated_response': generated_response,
                'true_expert': true_expert,
                'predicted_expert': predicted_expert,
                'expert_correct': predicted_expert == true_expert,
                'true_complexity': true_complexity,
                'predicted_complexity': predicted_complexity,
                'complexity_error': abs(predicted_complexity - true_complexity),
                'routing_weights': routing_weights.tolist(),
                'routing_confidence': np.max(routing_weights),
            })

    return examples


def compute_generation_metrics(
    examples: List[Dict[str, Any]],
    model: ComplexityAwareLoRARouter,
    tokenizer: AutoTokenizer,
    compute_perplexity: bool = False
) -> Dict[str, Any]:
    """Compute generation quality metrics.

    Args:
        examples: Generated examples.
        model: Model for perplexity computation.
        tokenizer: Tokenizer instance.
        compute_perplexity: Whether to compute perplexity.

    Returns:
        Generation metrics.
    """
    generation_metrics = GenerationMetrics()

    references = [ex['expected_response'] for ex in examples]
    candidates = [ex['generated_response'] for ex in examples]

    results = {}

    # BLEU scores
    bleu_results = generation_metrics.compute_bleu_score(references, candidates)
    results.update({f'bleu_{k}': v for k, v in bleu_results.items()})

    # ROUGE scores
    rouge_results = generation_metrics.compute_rouge_scores(references, candidates)
    results.update(rouge_results)

    # Diversity metrics
    diversity_results = generation_metrics.compute_diversity_metrics(candidates, tokenizer)
    results.update({f'diversity_{k}': v for k, v in diversity_results.items()})

    # Perplexity (if requested)
    if compute_perplexity:
        try:
            # Use base model for perplexity computation
            perplexity_results = generation_metrics.compute_perplexity(
                model.base_model,
                tokenizer,
                candidates
            )
            results.update({f'perplexity_{k}': v for k, v in perplexity_results.items()})
        except Exception as e:
            logger.warning(f"Could not compute perplexity: {e}")

    return results


def save_results(
    results: Dict[str, Any],
    examples: List[Dict[str, Any]],
    output_dir: str,
    save_predictions: bool = False
) -> None:
    """Save evaluation results to files.

    Args:
        results: Evaluation results.
        examples: Generated examples.
        output_dir: Output directory.
        save_predictions: Whether to save all predictions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save summary results
    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.number):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        json.dump(convert_numpy(results), f, indent=2)

    logger.info(f"Summary results saved to {summary_path}")

    # Save examples
    examples_path = output_path / "generated_examples.json"
    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)

    logger.info(f"Generated examples saved to {examples_path}")

    # Create examples DataFrame for easier viewing
    if examples:
        examples_df = pd.DataFrame(examples)
        examples_csv_path = output_path / "generated_examples.csv"
        examples_df.to_csv(examples_csv_path, index=False)
        logger.info(f"Examples CSV saved to {examples_csv_path}")

    # Save detailed predictions if requested
    if save_predictions and examples:
        predictions_path = output_path / "all_predictions.json"
        detailed_predictions = {
            'routing_predictions': [ex['predicted_expert'] for ex in examples],
            'routing_targets': [ex['true_expert'] for ex in examples],
            'complexity_predictions': [ex['predicted_complexity'] for ex in examples],
            'complexity_targets': [ex['true_complexity'] for ex in examples],
            'routing_weights': [ex['routing_weights'] for ex in examples],
        }

        with open(predictions_path, 'w') as f:
            json.dump(detailed_predictions, f, indent=2)

        logger.info(f"Detailed predictions saved to {predictions_path}")


def print_summary(results: Dict[str, Any], examples: List[Dict[str, Any]]) -> None:
    """Print evaluation summary.

    Args:
        results: Evaluation results.
        examples: Generated examples.
    """
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Routing performance
    if 'routing' in results:
        routing = results['routing']
        print(f"\nROUTING PERFORMANCE:")
        print(f"  Accuracy: {routing['routing_accuracy']:.4f}")
        print(f"  F1 Score (macro): {routing.get('f1_macro', 'N/A'):.4f}")
        print(f"  Load Balance Variance: {routing.get('load_balance_variance', 'N/A'):.4f}")

        if 'utilization_per_expert' in routing:
            print(f"  Expert Utilization:")
            for i, util in enumerate(routing['utilization_per_expert']):
                print(f"    Expert {i}: {util:.3f}")

    # Complexity prediction
    if 'complexity' in results:
        complexity = results['complexity']
        print(f"\nCOMPLEXITY PREDICTION:")
        print(f"  MSE: {complexity['mse']:.4f}")
        print(f"  RMSE: {complexity['rmse']:.4f}")
        print(f"  MAE: {complexity['mae']:.4f}")
        print(f"  R²: {complexity['r2']:.4f}")

    # Generation quality (if available)
    if 'bleu_mean' in results:
        print(f"\nGENERATION QUALITY:")
        print(f"  BLEU: {results['bleu_mean']:.4f}")
        print(f"  ROUGE-L: {results.get('rougeL_mean', 'N/A'):.4f}")
        print(f"  Distinct-1: {results.get('diversity_distinct_1', 'N/A'):.4f}")
        print(f"  Distinct-2: {results.get('diversity_distinct_2', 'N/A'):.4f}")

    # Example analysis
    if examples:
        routing_correct = sum(1 for ex in examples if ex['expert_correct'])
        avg_complexity_error = np.mean([ex['complexity_error'] for ex in examples])
        avg_confidence = np.mean([ex['routing_confidence'] for ex in examples])

        print(f"\nEXAMPLE ANALYSIS ({len(examples)} samples):")
        print(f"  Routing Accuracy: {routing_correct}/{len(examples)} ({routing_correct/len(examples):.3f})")
        print(f"  Avg Complexity Error: {avg_complexity_error:.4f}")
        print(f"  Avg Routing Confidence: {avg_confidence:.4f}")

    print("\n" + "="*80)


def main() -> None:
    """Main evaluation function."""
    args = parse_arguments()
    setup_logging(args.debug)

    logger.info("Starting model evaluation...")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Load configuration
        if os.path.exists(args.config):
            config = load_config(args.config)
        else:
            config = Config()

        # Override batch size
        if args.batch_size:
            config.training.batch_size = args.batch_size

        # Load model and data
        model, tokenizer, dataloader, dataset = load_model_and_data(
            args.model_path,
            config,
            args.eval_split,
            args.batch_size,
            args.num_samples
        )

        # Evaluate routing and complexity prediction
        results = evaluate_routing_and_complexity(model, dataloader)

        # Generate examples if requested
        examples = []
        if args.generate_examples:
            examples = generate_examples(
                model,
                tokenizer,
                dataset,
                args.num_examples,
                args.max_generation_length
            )

            # Compute generation metrics
            generation_results = compute_generation_metrics(
                examples,
                model,
                tokenizer,
                args.compute_perplexity
            )
            results.update(generation_results)

        # Save results
        save_results(results, examples, args.output_dir, args.save_predictions)

        # Print summary
        print_summary(results, examples)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()