#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ABLATION="${RUN_ABLATION:-0}"
RUN_CIFAR="${RUN_CIFAR:-0}"

echo "[1/9] Checking environment and running unit tests"
"${PYTHON_BIN}" code/check_environment.py
"${PYTHON_BIN}" -m pytest -q

echo "[2/9] Training MNIST evaluation classifier"
"${PYTHON_BIN}" code/train_mnist_classifier.py

echo "[3/9] Training unconditional MNIST baseline"
"${PYTHON_BIN}" code/train_mnist_ddpm.py

echo "[4/9] Generating required MNIST results"
"${PYTHON_BIN}" code/show_mnist_ddpm.py

echo "[5/9] Training cosine class-conditional DDPM"
"${PYTHON_BIN}" code/train_conditional_ddpm.py

echo "[6/9] Evaluating classifier-free guidance"
"${PYTHON_BIN}" code/show_conditional_ddpm.py

echo "[7/9] Training matched linear class-conditional DDPM"
"${PYTHON_BIN}" code/train_conditional_ddpm.py \
  --beta-schedule linear \
  --output-dir outputs/mnist_conditional_linear

echo "[8/9] Comparing schedules and samplers"
"${PYTHON_BIN}" code/compare_schedules.py \
  --linear-checkpoint outputs/mnist_conditional_linear/latest.pt \
  --cosine-checkpoint outputs/mnist_conditional/latest.pt
"${PYTHON_BIN}" code/compare_samplers.py

echo "[9/9] Optional long-running experiments"
if [[ "${RUN_ABLATION}" == "1" ]]; then
  "${PYTHON_BIN}" code/run_ablation.py
  "${PYTHON_BIN}" code/plot_experiments.py
fi
if [[ "${RUN_CIFAR}" == "1" ]]; then
  "${PYTHON_BIN}" code/train_cifar10_ddpm.py
  "${PYTHON_BIN}" code/show_cifar10_ddpm.py
fi

echo "All requested stages completed. Results are under outputs/."
