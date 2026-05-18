# 📘 Symbolic Algorithm Discovery with CGP and LLMs

This project explores the automatic discovery of interpretable algorithms from data using **Cartesian Genetic Programming (CGP)** and **Large Language Model (LLM)-assisted evolutionary search**. We focus on reconstructing and improving the **Kalman filter** using only raw input-output trajectories and black-box loss optimization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cselab/kalman/blob/main/demo/demo.ipynb) — a self-contained reproduction of Tables 1 and 2 (see `demo/demo.py` / `demo/demo.ipynb`).

For directory layout, the seed convention, how to run CGP and FunSearch end-to-end, and known reproducibility caveats, see [`DEVELOPER.md`](DEVELOPER.md).

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



To reproduce the results:

1.  **Navigate to the `bipacking` directory:**
    ```bash
    cd bipacking
    ```

2.  **Run the FunSearch experiment:**
    This script will generate candidate programs using the LLM-assisted evolutionary search. The output will be saved to `.out` and `.err` files.

    ```bash
    python funsearch.py --iterations 1000 --num_threads 4 --outer_iterations 100
    ```
    *   `--iterations`: The number of iterations for each of the `num_threads` processes.
    *   `--num_threads`: The number of parallel processes (islands) to run. This should typically match the number of available GPUs.
    *   `--outer_iterations`: The number of times the evolutionary process is repeated, re-initializing the worst-performing islands.

3.  **Validate the results:**
    This script will analyze the `.out` files, extract the best-performing programs, and evaluate them on a validation set.

    ```bash
    python validate.py
    ```

---

### Other Experiments

The `table1` and `table2` directories contain scripts for the CGP and Random Search experiments, as well as the different configurations for the FunSearch experiments. The scripts in these directories are specific to the experiments described in the paper and may require more specific setup. The `bipacking` directory contains the core logic for the FunSearch experiments.

### Generate zip

```bash
git archive --format=zip --prefix=kalman/ HEAD > kalman.zip
```
