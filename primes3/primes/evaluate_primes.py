import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
from time import time
from math import factorial

# ----------------------
# Configuration
# ----------------------
nth_max = 100_000_000  # Adjust if needed
print(f"📂 Loading the first {nth_max:,} primes from disk")

# ----------------------
# Load primes from disk
# ----------------------
prime_list = np.load("prime_list.npy")
assert len(prime_list) >= nth_max, "Not enough primes loaded from file."

# ----------------------
# Heuristic functions
# ----------------------
def basic_heuristic(n):
    return n * ln(n)

def improved_heuristic(n):
    return n * (ln(n) + ln(ln(n)))

def heuristic(n):
    return n * (ln(n) + ln(ln(n)) - 1 + ((ln(ln(n)) - 2) / ln(n)))

def heuristic(x):

    logx = ln(x)
    series_sum = 0
    for k in range(4):  # terms: k = 0 to 3
        series_sum += (logx ** k) / factorial(k)
    
    return (x / logx) * series_sum

# ----------------------
# Heuristic Evaluation
# ----------------------
eval_start = time()

n_vals = np.arange(2, nth_max + 2)
pn_actual = np.array(prime_list[:nth_max], dtype=np.float64)

pn_basic    = basic_heuristic(n_vals)
pn_improved = improved_heuristic(n_vals)
pn_best     = heuristic(n_vals)

eval_duration = time() - eval_start
print(f"\n⏱️ Heuristic evaluation completed in {eval_duration:.2f} seconds")

# ----------------------
# Error Metrics
# ----------------------
def mse(true, pred):
    return np.mean((true - pred)**2)

def mean_relative_error_pct(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100

print("\n📊 Mean Squared Error:")
print(f"  Basic:    {mse(pn_actual, pn_basic):.2e}")
print(f"  Improved: {mse(pn_actual, pn_improved):.2e}")
print(f"  Best:     {mse(pn_actual, pn_best):.2e}")

print("\n📉 Mean Relative Error (%):")
print(f"  Basic:    {mean_relative_error_pct(pn_actual, pn_basic):.4f}%")
print(f"  Improved: {mean_relative_error_pct(pn_actual, pn_improved):.4f}%")
print(f"  Best:     {mean_relative_error_pct(pn_actual, pn_best):.4f}%")

# ----------------------
# Plotting
# ----------------------
plot_start = time()

# Downsample for performance if very large
sample_step = 1000 if nth_max > 1_000_000 else 1
n_sample = n_vals[::sample_step]
actual_sample = pn_actual[::sample_step]
basic_sample = pn_basic[::sample_step]
improved_sample = pn_improved[::sample_step]
best_sample = pn_best[::sample_step]

# Plot 1: Heuristic vs Actual
plt.figure(figsize=(10, 6))
plt.plot(n_sample, actual_sample, label='Actual pₙ', linewidth=2)
plt.plot(n_sample, basic_sample, '--', label='Basic: n·ln(n)')
plt.plot(n_sample, improved_sample, '-.', label='Improved')
plt.plot(n_sample, best_sample, ':', label='Best')
plt.title('Heuristic Approximations vs Actual nth Prime')
plt.xlabel('n')
plt.ylabel('Prime Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plot1_from_loaded.png")
plt.show()

# Plot 2: Relative Error (%) of Best Heuristic
rel_err_pct = ((pn_best - pn_actual))
rel_err_sample = rel_err_pct[::sample_step]

plt.figure(figsize=(10, 6))
plt.plot(n_sample, rel_err_sample, color='purple', alpha=0.7)
plt.title('Real Error (%) of Best Heuristic')
plt.xlabel('n (Prime Index)')
plt.ylabel('Relative Error (%)')
plt.grid()
plt.tight_layout()
plt.savefig("plot2_from_loaded.png")
plt.show()

plot_duration = time() - plot_start
print(f"📈 Plotting completed in {plot_duration:.2f} seconds")
