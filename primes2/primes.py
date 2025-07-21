import numpy as np
import matplotlib.pyplot as plt
from numpy import log as ln
from time import time

# ----------------------
# Configuration
# ----------------------
nth_max = 10_000_000  # Number of primes to generate (e.g., 1 million)
print(f"🌟 Computing the first {nth_max:,} primes")

# ----------------------
# Heuristic functions
# ----------------------
def basic_heuristic(n):
    return n * ln(n)

def improved_heuristic(n):
    return n * (ln(n) + ln(ln(n)))

def best_heuristic(n):
    return n * (ln(n) + ln(ln(n)) - 1 + ((ln(ln(n)) - 2) / ln(n)))

# ----------------------
# Prime generation: Sieve to generate at least nth_max primes
# ----------------------
def generate_n_primes(n):
    # Estimate upper bound for nth prime using best heuristic
    estimate = int(best_heuristic(n) * 1.05)
    sieve = bytearray([True]) * (estimate + 1)
    sieve[0:2] = bytearray([False, False])
    for i in range(2, int(estimate**0.5) + 1):
        if sieve[i]:
            sieve[i*i:estimate+1:i] = bytearray([False]) * len(range(i*i, estimate+1, i))
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    return primes[:n]

# ----------------------
# Generate primes
# ----------------------
start = time()
prime_list = generate_n_primes(nth_max)
np.save("prime_list.npy", np.array(prime_list))
duration = time() - start
print(f"🔍 Generated {len(prime_list):,} primes in {duration:.2f}s")

# ----------------------
# Heuristic evaluation
# ----------------------
# Avoid log(log(1)) by starting from 2
n_vals = np.arange(2, nth_max + 2)
pn_actual = np.array(prime_list[:nth_max], dtype=np.float64)

pn_basic    = basic_heuristic(n_vals)
pn_improved = improved_heuristic(n_vals)
pn_best     = best_heuristic(n_vals)

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
# Plot: nth prime vs heuristic estimates
# ----------------------
plt.figure(figsize=(10, 6))
plt.plot(n_vals, pn_actual, label='Actual pₙ', linewidth=2)
plt.plot(n_vals, pn_basic, '--', label='Basic: n·ln(n)')
plt.plot(n_vals, pn_improved, '-.', label='Improved')
plt.plot(n_vals, pn_best, ':', label='Best')
plt.title('Heuristic Approximations vs Actual nth Prime')
plt.xlabel('n')
plt.ylabel('Prime Value')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plot1.png")
plt.show()

# ----------------------
# Plot: Relative Error (%) of Best Heuristic
# ----------------------
rel_err_pct = np.abs((pn_best - pn_actual) / pn_actual) * 100
plt.figure(figsize=(10, 6))
plt.plot(n_vals, rel_err_pct, color='purple', alpha=0.7)
plt.title('Relative Error (%) of Best Heuristic')
plt.xlabel('n (Prime Index)')
plt.ylabel('Relative Error (%)')
plt.grid()
plt.tight_layout()
plt.savefig("plot2.png")
plt.show()
