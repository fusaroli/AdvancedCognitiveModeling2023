#!/usr/bin/env bash
# uCloud bootstrap for amortized Bayesian inference on the rule-based particle filter.
# Run once at the start of the session:
#     bash install_deps.sh
# Tested on uCloud "VS Code" / Ubuntu 22.04. CUDA wheels for JAX are pulled
# automatically when a GPU is visible; CPU-only nodes silently fall back.

set -euo pipefail

python -m pip install --upgrade pip

# Backend: JAX is preferred per the amortized-workflow guardrails.
# Try CUDA 12 wheels first (for GPU nodes); fall back to CPU-only JAX silently.
if python -c "import subprocess, sys; r = subprocess.run(['nvidia-smi'], capture_output=True); sys.exit(0 if r.returncode == 0 else 1)" 2>/dev/null; then
    echo "GPU detected — installing JAX with CUDA 12 support."
    python -m pip install --upgrade "jax[cuda12]"
else
    echo "No GPU detected — installing CPU-only JAX."
    python -m pip install --upgrade "jax[cpu]"
fi

python -m pip install --upgrade \
    "bayesflow>=2.0" \
    "keras>=3.5" \
    "numpy" \
    "pandas" \
    "matplotlib" \
    "tqdm"

# Sanity check: print versions and detected devices
python - <<'PY'
import os
os.environ.setdefault("KERAS_BACKEND", "jax")
import jax, keras, bayesflow as bf
print("jax       :", jax.__version__, "devices:", jax.devices())
print("keras     :", keras.__version__, "backend:", keras.backend.backend())
print("bayesflow :", bf.__version__)
PY

echo
echo "Install complete. Run training with:"
echo "    KERAS_BACKEND=jax python amortized_particle_filter.py"
