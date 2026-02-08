"""Constants for the instruction complexity-aware LoRA routing system.

This module centralizes magic numbers and configuration values used throughout
the codebase to improve maintainability and documentation.
"""

# Feature normalization constants
INSTRUCTION_LENGTH_CAP = 50.0  # Maximum normalized instruction length
RESPONSE_LENGTH_CAP = 100.0   # Maximum normalized response length
SYNTAX_TREE_DEPTH_CAP = 10.0  # Maximum normalized syntax tree depth
SENTENCE_COUNT_CAP = 10.0     # Maximum normalized sentence count
AVG_SENT_LENGTH_CAP = 30.0    # Maximum normalized average sentence length

# Complexity computation weights
COMPLEXITY_FEATURE_WEIGHTS = {
    "instruction_length": 0.2,   # Instruction length importance (20%)
    "response_length": 0.2,      # Response length importance (20%)
    "syntactic_complexity": 0.3, # Primary linguistic indicator (30%)
    "semantic_diversity": 0.15,  # Vocabulary richness (15%)
    "dependency_depth": 0.15     # Grammatical complexity (15%)
}

# Loss combination weights
COMPLEXITY_LOSS_WEIGHT = 0.1   # Weight for auxiliary complexity prediction task
LOAD_BALANCE_LOSS_WEIGHT = 0.1   # Weight for expert load balancing regularization (increased to prevent collapse)

# Model architecture constants
EPSILON = 1e-8  # Small constant to avoid log(0) in entropy calculations
MAX_DEPENDENCY_DEPTH = 20  # Cutoff to prevent infinite loops in parsing

# Training constants
CHECKPOINT_SAVE_INTERVAL = 5  # Save checkpoint every N epochs
DEFAULT_COMPLEXITY_SCORE = 0.5  # Fallback complexity when computation fails

# Data processing constants
DEFAULT_MAX_LENGTH = 512  # Default maximum sequence length
PROMPT_RESPONSE_SPLIT_RATIO = 50  # Character ratio for prompt/response splitting