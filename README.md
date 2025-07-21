# 📘 Symbolic Algorithm Discovery with CGP and LLMs

This project explores the automatic discovery of interpretable algorithms from data using **Cartesian Genetic Programming (CGP)** and **Large Language Model (LLM)-assisted evolutionary search**. We focus on reconstructing and improving the **Kalman filter** using only raw input-output trajectories and black-box loss optimization.

---

## 🧪 Experimental Overview

We organize our experiments into two core evaluations:

| Table | Description |
|-------|-------------|
| **Table 1** | **Kalman Optimal Conditions**  
Evaluates CGP and LLM-assisted approaches on tasks where Kalman filter assumptions hold (e.g., linear dynamics, Gaussian noise). We compare learned algorithms to the classical Kalman filter using MSE metrics. |
| **Table 2** | **Adversarial and Non-Ideal Settings**  
Assesses performance when classical assumptions are violated, including delayed observations, nonlinear systems, and non-Gaussian noise. We report how symbolic programs adapt and generalize in these challenging regimes. |

---

## ✅ Key Findings

- **Near-Optimal Recovery**: Both CGP and LLM-assisted search can rediscover Kalman-like update rules from raw data.
- **Beyond Kalman**: In adversarial settings, both methods evolve **interpretable programs** that outperform the Kalman filter.
- **Interpretability**: Output programs are symbolic, concise, and human-readable — a key advantage over black-box models.

---

## 🔁 Reproducibility

This repository includes:

- Modular implementations of CGP and FunSearch-style mutation
- Configurable environments for ideal and non-ideal conditions
- Automatic logging of symbolic programs, MSE scores, and trajectory plots

To reproduce the results:

```bash
# Clone the repo and set up the environment
git clone https://github.com/yourname/algorithm-discovery-kalman
cd algorithm-discovery-kalman
pip install -r requirements.txt

# Run CGP experiments
python run_cgp.py --config configs/ideal.yaml

# Run LLM-assisted search
python run_funsearch.py --config configs/nonideal.yaml
