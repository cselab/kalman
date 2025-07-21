import os
import re
import pickle
import numpy as np
import tempfile
import subprocess
import sys
import traceback

# === Load validation instances ===
with open("val_instances.pkl", "rb") as f:
    val_instances = pickle.load(f)

def l1_bound(items: np.ndarray, capacity: int) -> float:
    return np.ceil(np.sum(items) / capacity)

def l1_bound_dataset(instances: list) -> float:
    return np.mean([
        l1_bound(instance['Items']['Size'].to_numpy(), instance['Bin Capacity'])
        for instance in instances
    ])

val_l1 = l1_bound_dataset(val_instances)

# === Evaluate function string on val_instances ===
def evaluate_priority_function_on_val(priority_code: str) -> float:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as code_file:
            code_file.write(priority_code)
            code_file.flush()

        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as data_file:
            pickle.dump((val_instances, val_l1), data_file)
            data_file.flush()

        eval_script = """
import sys
import numpy as np
import pickle
import traceback

def create_function(path):
    with open(path, 'r') as f:
        code = f.read()
    local_vars = {}
    global_vars = {"np": __import__("numpy")}
    exec(code, global_vars, local_vars)
    return local_vars['priority']

def evaluate(instances, priority_fn):
    def get_valid_bin_indices(item, bins):
        return np.nonzero((bins - item) >= 0)[0]
    def online_binpack(items, bins, priority_fn):
        packing = [[] for _ in bins]
        for item in items:
            valid = get_valid_bin_indices(item, bins)
            if len(valid) == 0:
                continue
            scores = priority_fn(item, bins[valid])
            best = valid[np.argmax(scores)]
            bins[best] -= item
            packing[best].append(item)
        return packing, bins
    num_bins = []
    for instance in instances:
        try:
            items = instance['Items']['Size'].to_numpy()
            capacity = instance['Bin Capacity']
            bins = np.array([capacity] * len(items))
            _, bins_packed = online_binpack(items, bins, priority_fn)
            used_bins = (bins_packed != capacity).sum()
            num_bins.append(used_bins)
        except:
            pass
    return -np.mean(num_bins)

try:
    code_path, data_path = sys.argv[1], sys.argv[2]
    with open(data_path, 'rb') as f:
        instances, opt_bins = pickle.load(f)

    if opt_bins is None or opt_bins <= 0 or np.isnan(opt_bins) or np.isinf(opt_bins):
        raise ValueError("Invalid opt_bins")

    fn = create_function(code_path)
    avg_bins = -evaluate(instances, fn)

    if np.isnan(avg_bins):
        raise ValueError("avg_bins is NaN")

    excess = (avg_bins - opt_bins) / opt_bins

    if np.isnan(excess):
        raise ValueError("excess is NaN")

    print(100 * excess)

except:
    print(float('inf'))
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as eval_file:
            eval_file.write(eval_script)
            eval_file.flush()

        process = subprocess.run(
            [sys.executable, eval_file.name, code_file.name, data_file.name],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=60
        )

        result_str = process.stdout.decode().strip()
        return float(result_str) if np.isfinite(float(result_str)) else float('inf')

    except:
        traceback.print_exc()
        return float('inf')
    finally:
        for path in [code_file.name, data_file.name, eval_file.name]:
            try:
                os.unlink(path)
            except:
                pass

# === Crawl .out files and evaluate ===
matches = []
for filename in os.listdir("."):
    if filename.endswith(".out"):
        with open(filename, "r") as f:
            content = f.read()
        file_matches = re.findall(
            r"best score so far: ([\d\.eE+-]+), content:\s*(def priority\(.*?)(?=\n\[Process|\Z)",
            content,
            flags=re.DOTALL
        )
        matches.extend(file_matches)

total = len(matches)
print(f"Found {total} candidate functions.\n")

best_score = float('inf')
best_function = ""

for i, (raw_score, func_code) in enumerate(matches, 1):
    func_code_cleaned = func_code.strip()
    if not func_code_cleaned.endswith("\n"):
        func_code_cleaned += "\n"

    validated_score = evaluate_priority_function_on_val(func_code_cleaned)
    print(f"[{i}/{total}] Raw: {float(raw_score):.4f} | Validated: {validated_score:.4f}%")

    if validated_score < best_score:
        best_score = validated_score
        best_function = func_code_cleaned

# === Final Output ===
print("\n=== Best Function on Validation Set ===")
print(f"Score: {best_score:.4f}% overhead\n")
print(best_function)
