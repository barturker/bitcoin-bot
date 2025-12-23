# Claude Code Rules

## Development Server & Execution
- **NEVER** run training scripts automatically without user confirmation
- Training can be resource-intensive; always ask before starting `python src/train.py`
- Let the user handle long-running processes and server management
- If testing is needed, ask the user about their preferred setup

## Language
- All code, comments, and documentation must be in **English only**
- Variable names, function names in English
- No Turkish or other languages in the codebase

## File Size & Architecture
- **Target max:** ≤ 300 lines per file
- **Hard limit:** ≤ 400 lines (only for complex modules like trading_env)
- **Strict rule:** Utilities and helpers must never exceed 200 lines
- Architecture is module-based with clear separation of concerns
- Every file must have a single, clearly defined responsibility

### File Organization
```
src/
  train.py          # Training orchestration
  trading_env.py    # Gym environment definition
  indicators.py     # Technical indicators & data preprocessing
  dashboard.py      # Visualization & monitoring
  test.py           # Testing utilities

config.yaml         # All hyperparameters and settings
logs/               # Training logs, TensorBoard, trade logs
models/             # Saved model checkpoints
cache/              # Preprocessed data, scalers
data/               # Raw data files (not in git)
```

### Module Responsibilities
- **train.py**: Training loop, callbacks, model creation
- **trading_env.py**: Gym environment, reward calculation, state management
- **indicators.py**: Data loading, technical indicators, feature engineering
- **dashboard.py**: Real-time monitoring, visualization
- **config.yaml**: ALL hyperparameters (no hardcoded values in code)

### Code Organization Rules (IMPORTANT)
- Keep modules focused and single-purpose
- Move shared utilities to separate files only if used by 2+ modules
- **Reuse is earned, not assumed**

### File Size Enforcement
When a file exceeds 300 lines:
1. Extract helper functions to feature-local utilities
2. Split into smaller, intention-revealing sub-modules
3. Move configuration to config.yaml

---

## Configuration Management
- **ALL hyperparameters must be in config.yaml**
- No hardcoded learning rates, batch sizes, or environment parameters
- Use environment variables only for paths and secrets

### Config Organization:
```yaml
data:
  train_path: "data/btc_data.csv"
  timeframe: "1h"

environment:
  window_size: 50
  initial_capital: 10000

model:
  learning_rate: 0.0003
  batch_size: 64

trading:
  spread_pct: 0.001
  commission_pct: 0.001
```

### Never hardcode in Python:
- Model hyperparameters
- Environment settings
- Trading parameters
- File paths

---

## Data & API Layer
- **NEVER** load data directly in training loops
- Data loading lives in `indicators.py`
- Use caching for preprocessed data

### Data Rules:
1. **Return clean DataFrames, not raw files**
   - Raw data shape ≠ Model input shape
   - Apply preprocessing before returning
2. **Handle missing data and edge cases**
3. **Log data statistics for debugging**

```python
# Good: indicators.py
def load_and_preprocess_data(
    csv_path: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    normalize: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Load, clean, and preprocess trading data."""
    ...

# Bad: Direct file read in train.py
df = pd.read_csv("data/btc.csv")  # NO!
```

---

## Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Modules | snake_case | `trading_env.py`, `indicators.py` |
| Classes | PascalCase | `BitcoinTradingEnv`, `RealTimeProgressCallback` |
| Functions | snake_case | `load_config()`, `create_env()` |
| Constants | SCREAMING_SNAKE_CASE | `MAX_POSITION_DURATION`, `DEFAULT_CAPITAL` |
| Variables | snake_case | `train_df`, `episode_rewards` |
| **Booleans** | `is/has/can/should` prefix | `is_done`, `has_position`, `can_trade` |
| Private | Leading underscore | `_calculate_reward()`, `_update_state()` |

---

## Import Order
Always organize imports in this order (with blank lines between groups):

```python
# 1. Standard library
import os
import sys
from datetime import datetime
from typing import Optional, Tuple

# 2. Third-party libraries
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# 3. Local modules
from indicators import load_and_preprocess_data
from trading_env import BitcoinTradingEnv
```

---

## Error Handling
- **Every** data loading and model operation must have try/except
- **NEVER** let training crash without saving state
- Log errors with context for debugging
- Use graceful degradation where possible

### Error Handling Pattern:
```python
def train():
    try:
        model = create_model(config)
        model.learn(total_timesteps=config['total_timesteps'])
    except KeyboardInterrupt:
        print("Training interrupted by user")
        model.save("models/interrupted_checkpoint.zip")
    except Exception as e:
        print(f"Training failed: {e}")
        traceback.print_exc()
        # Save whatever progress was made
        if model is not None:
            model.save("models/error_checkpoint.zip")
    finally:
        cleanup_resources()
```

---

## Performance & ML Best Practices

### Data Handling
- Use numpy vectorized operations, avoid Python loops for data
- Cache preprocessed features to disk
- Use appropriate dtypes (float32 for features, int32 for actions)

### Training
- Always use callbacks for monitoring (TensorBoard, checkpoints)
- Implement early stopping based on validation metrics
- Log all hyperparameters for reproducibility

### Memory Management
- Clear unused DataFrames with `del df; gc.collect()`
- Use generators for large datasets
- Monitor GPU memory if using CUDA

```python
# Good - vectorized
returns = np.diff(prices) / prices[:-1]

# Bad - Python loop
returns = []
for i in range(1, len(prices)):
    returns.append((prices[i] - prices[i-1]) / prices[i-1])
```

---

## No Magic Numbers/Strings
- Extract constants to config.yaml or module-level constants
- Use descriptive names for all numeric values

```python
# Bad - magic numbers
if position_duration > 100:
    reward -= 0.01

# Good - named constants from config
if position_duration > config['environment']['max_position_duration']:
    reward -= config['rewards']['holding_penalty']
```

---

## Type Hints
- **Use type hints** for all function signatures
- **NO `Any` type** - use proper types or `Union`
- Define custom types for complex structures

```python
# Good
def create_env(
    df: pd.DataFrame,
    features: pd.DataFrame,
    config: dict[str, Any]
) -> BitcoinTradingEnv:
    ...

# Bad - no hints
def create_env(df, features, config):
    ...
```

---

## Testing & Validation
- Test environment step/reset functions independently
- Validate config values on load
- Use deterministic seeds for reproducibility

```python
# Always set seeds for reproducibility
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])
```

---

## Git & Version Control
- **NEVER commit:**
  - Large data files (use .gitignore)
  - Model checkpoints (except final releases)
  - Cache files
  - API keys or secrets
- **ALWAYS commit:**
  - Config changes with descriptive messages
  - New features with tests
