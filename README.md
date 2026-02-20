# Instruction Complexity-Aware LoRA Routing

A mixture-of-LoRA-experts system that dynamically routes instruction queries to specialized adapters based on task complexity estimation. The approach trains multiple LoRA modules on complexity-stratified subsets of the Alpaca dataset and learns a lightweight neural router that predicts which expert(s) to activate based on linguistic complexity features extracted via spaCy.

## Architecture

```
                    ┌──────────────────────┐
                    │  Input Instruction    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Complexity Analyzer  │  spaCy-based linguistic features
                    │  (syntactic depth,    │  (dependency depth, entity density,
                    │   vocabulary, etc.)   │   clause count, readability, etc.)
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    Neural Router      │  MLP with softmax routing
                    │  (256-dim hidden)     │
                    └──────────┬───────────┘
                      ┌────────┼────────┐
                      │        │        │
               ┌──────▼──┐ ┌──▼─────┐ ┌▼────────┐
               │ Expert 0 │ │Expert 1│ │ Expert 2 │
               │ (Simple) │ │ (Med)  │ │(Complex) │
               │ LoRA r=16│ │LoRA r=16│ │LoRA r=16│
               └──────────┘ └────────┘ └──────────┘
                      (GPT-2 base model)
```

### Components

- **Complexity Analyzer**: Computes multi-dimensional instruction complexity using spaCy NLP features (syntactic depth, named entity density, vocabulary richness, clause structure, readability scores)
- **Data Stratifier**: Assigns instructions to complexity tiers (simple/medium/complex) using configurable thresholds
- **LoRA Experts**: 3 specialized PEFT/LoRA adapters (rank=16, alpha=32) targeting GPT-2 attention layers (`c_attn`, `c_proj`), each trained on a complexity-stratified subset
- **Neural Router**: A learned 256-dim MLP that routes inputs to the appropriate expert based on extracted features and complexity scores

## Training Results

Trained on 52K instruction-response pairs from the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using an NVIDIA RTX 4090 GPU. Total training time was approximately 2.5 hours across 5 epochs with three-phase optimization (router-only, expert-only, joint training).

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | GPT-2 (124M params) |
| LoRA Parameters | 3 independent experts x ~1.6M LoRA params each |
| LoRA Rank / Alpha | 16 / 32 |
| Target Modules | c_attn, c_proj |
| Training Epochs | 5 |
| Batch Size | 8 (effective 32 with 4x gradient accumulation) |
| Expert LR / Router LR | 5e-5 / 1e-3 |
| Load Balance Loss Weight | 0.1 |
| GPU | NVIDIA RTX 4090 (24GB VRAM) |

### Epoch-Level Metrics

| Epoch | Avg Train Loss | Best Batch Loss | Training Phase |
|-------|---------------|-----------------|----------------|
| 1 | 3.34 | 0.99 | Router + Expert warmup |
| 2 | 1.78 | 0.97 | Joint optimization |
| 3 | 1.78 | 0.89 | Joint optimization |
| 4 | **1.77** | **0.85** | Joint optimization |
| 5 | 1.78 | 0.92 | Joint optimization |

### Final Results

| Metric | Value |
|--------|-------|
| Test Loss | 2.494 |
| Test Perplexity | 12.11 |
| Routing Accuracy | 97.2% |
| Complexity MSE | 0.008 |
| Best Validation Loss | 2.55 |
| Final Train Loss (avg) | 1.78 |
| Expert Independence | Confirmed (3 separate adapters, balanced router biases) |
| Training Time | ~2.5 hours (RTX 4090) |

**Training artifacts**: Model checkpoints including 3 independent LoRA expert adapters (`.safetensors`), router weights, and training state are saved under `outputs/checkpoints/best_model/` and `outputs/final_model/`.

### Training Curves

Loss progression across 5 epochs (51,441 total steps):
- Epoch 1: Rapid convergence from 5.5 to ~2.3 (router and expert initialization)
- Epoch 2: Significant drop to 1.5-1.7 range as joint optimization begins
- Epoch 3-5: Stable at 1.4-1.8 with continued refinement

### Analysis

The three-phase training approach successfully trains independent LoRA experts with a learned routing mechanism. Key improvements over the initial version:

- **Expert independence**: Each expert is initialized from a separate GPT-2 base model copy, ensuring truly independent LoRA adapter training. Router biases are well-balanced across all 3 experts (near-zero bias for each).
- **Convergence**: The joint training loss converged from 3.34 (Epoch 1) to 1.77 (Epoch 4), with the largest improvement occurring in Epoch 1-2 as the router learned to assign instructions to appropriate complexity-specialized experts.
- **Load balancing**: With a load balance loss weight of 0.1, the router distributes instructions across experts based on complexity features rather than collapsing to a single expert.
- **Numerical stability**: Using GPT-2 in float32 (instead of DialoGPT in float16) eliminates NaN gradients and ensures stable training throughout.

## Quick Start

### Training

```bash
# Full training
python scripts/train.py --config configs/default.yaml

# Debug mode (reduced batch size and epochs)
python scripts/train.py --config configs/default.yaml --debug
```

### Inference

```python
from instruction_complexity_aware_lora_routing import ComplexityAwareLoRARouter
from transformers import AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = ComplexityAwareLoRARouter.from_pretrained("outputs/checkpoints/best_model", tokenizer=tokenizer)

instruction = "Explain how neural networks work"
inputs = tokenizer(instruction, return_tensors="pt", max_length=512, truncation=True)
complexity_score = torch.tensor([0.7])

generated = model.generate(
    inputs["input_ids"],
    inputs["attention_mask"],
    complexity_score,
    max_new_tokens=100
)
response = tokenizer.decode(generated[0], skip_special_tokens=True)
```

## Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  base_model_name: "gpt2"
  num_experts: 3
  complexity_thresholds: [0.3, 0.7]
  lora_rank: 16
  lora_alpha: 32
  target_modules: ["c_attn", "c_proj"]
  router_hidden_dim: 256

training:
  batch_size: 8
  learning_rate: 5e-5
  router_learning_rate: 1e-3
  num_epochs: 5
```

## Installation

```bash
pip install -e .
```

## Project Structure

```
src/instruction_complexity_aware_lora_routing/
├── data/               # Alpaca data loading, complexity analysis, stratification
├── models/             # MoE-LoRA architecture with neural routing
├── training/           # Training loop with joint expert + router optimization
├── evaluation/         # Routing accuracy, complexity MSE, expert usage metrics
└── utils/              # Configuration management

scripts/
├── train.py           # End-to-end training pipeline
└── evaluate.py        # Model evaluation and analysis

configs/               # YAML configuration files
tests/                 # Test suite
notebooks/             # Exploration notebooks
```

## Hardware Requirements

- GPU: NVIDIA GPU with 8+ GB VRAM (trained on RTX 4090)
- RAM: 16+ GB
- Storage: ~5 GB for model + dataset caching
