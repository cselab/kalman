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


## 💻 Computational Resources

This section describes the hardware and software setup used to run the experiments in this project.

### 🧠 CGP-Based Symbolic Search
- **Compute Node**: 64-core SEAS cluster node  
- **RAM**: 10 GB  
- **Runtime**: Approximately 3 days per experiment  
- **Environment**: Python 3.10  
The CGP pipeline is CPU-efficient and does not require GPU acceleration. Parallel evaluation of mutant programs across multiple CPU cores significantly accelerates the search process.

### 🤖 LLM-Assisted Evolutionary Search
- **GPU Configuration**: 4× NVIDIA H100 (80 GB each)  
- **Use Case**: Large language model inference and parallel fitness evaluation  
- **Runtime**: Approximately 3 days  
- **Model Used**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`  
LLM-assisted search requires GPU execution. We recommend assigning one GPU per island to fully parallelize the symbolic search process.

#### SLURM Configuration Example
```bash
#SBATCH --gres=gpu:4
#SBATCH --constraint=h100
```
#### num_threads coresponds to the number of islands 
```python
parser.add_argument("--num_threads", type=int, default=4)
```

To run the experiments on smaller or fewer GPUs, reduce memory usage with the following modifications:

Lower the batch size:
```python
prompt = mem.sample_batch(30)  # Reduce to 10–15 for lower memory usage
```

Limit the output token length from the LLM (It might affect convergence):

```python
outputs = model.generate(**inputs, max_length=3024)  # Reduce to 2024 
```



To reproduce the results (Will be done soon):

```bash
# Clone the repo and set up the environment
git clone https://github.com/yourname/algorithm-discovery-kalman
cd algorithm-discovery-kalman
pip install -r requirements.txt

# Run CGP experiments
python run_cgp.py --config configs/ideal.yaml

# Run LLM-assisted search
python run_funsearch.py --config configs/nonideal.yaml
