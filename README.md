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
                      (GPT-2 / DialoGPT-small base)
```

### Components

- **Complexity Analyzer**: Computes multi-dimensional instruction complexity using spaCy NLP features (syntactic depth, named entity density, vocabulary richness, clause structure, readability scores)
- **Data Stratifier**: Assigns instructions to complexity tiers (simple/medium/complex) using configurable thresholds
- **LoRA Experts**: 3 specialized PEFT/LoRA adapters (rank=16, alpha=32) targeting GPT-2 attention layers (`c_attn`, `c_proj`), each trained on a complexity-stratified subset
- **Neural Router**: A learned 256-dim MLP that routes inputs to the appropriate expert based on extracted features and complexity scores

## Training Results

Trained on 52K instruction-response pairs from the [Alpaca dataset](https://huggingface.co/datasets/tatsu-lab/alpaca) using an NVIDIA RTX 4090 GPU.

| Metric | Value |
|--------|-------|
| Base Model | DialoGPT-small (117M params) |
| LoRA Parameters | 3 experts x ~590K params each |
| Training Epochs | 3 |
| Best Validation Loss | 7.77 |
| Test Loss | 7.80 |
| Test Perplexity | 2444.50 |
| **Routing Accuracy** | **96.9%** |
| Avg Routing Latency | 3.97ms |
| Training Time | ~40 min (RTX 4090) |

**Training artifacts**: Model checkpoints including 3 LoRA expert adapters (`.safetensors`), router weights, and training state are saved under `outputs/checkpoints/best_model/`.

### Training Curves

Loss progression across 3 epochs (15,600 steps):
- Epoch 1: Loss converged from ~5.9 to ~6.0 (avg)
- Epoch 2: Loss stabilized around 5.5-7.0
- Epoch 3: Loss ranged 4.2-7.5 with continued fluctuation

### Known Limitations

- **Expert collapse**: The router currently routes all test inputs to Expert 1 (medium complexity). This is a known issue with MoE training that can be addressed with load balancing losses, auxiliary routing losses, or longer training with diversity-encouraging regularization.
- **Perplexity**: The language modeling perplexity is high due to the small model size (117M) and short training. Scaling to larger base models (GPT-2 medium/large) would significantly improve generation quality.

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

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
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
  base_model_name: "microsoft/DialoGPT-small"
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
  num_epochs: 3
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
