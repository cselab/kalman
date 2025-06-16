import os
import pickle
import numpy as np

# === Load Pickled Datasets ===
def load_datasets(directory):
    datasets = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".pkl"):
            path = os.path.join(directory, filename)
            with open(path, "rb") as f:
                datasets[filename.replace(".pkl", "")] = pickle.load(f)
    return datasets['all_datasets']

# === L1 Lower Bound ===
def l1_bound(items: np.ndarray, capacity: int) -> float:
    return np.ceil(np.sum(items) / capacity)

def l1_bound_dataset(instances: list) -> float:
    l1_bounds = []
    for instance in instances:
        items = instance['Items']['Size'].to_numpy()
        capacity = instance['Bin Capacity']
        l1_bounds.append(l1_bound(items, capacity))
    return np.mean(l1_bounds)

# === Online Bin Packing ===
def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    return np.nonzero((bins - item) >= 0)[0]

def default_priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -(bins - item)

def or_priority(item: float, bins: np.ndarray) -> np.ndarray:
    def s(bin, item):
        gap = bin - item
        if gap <= 2:
            return 4
        elif gap <= 3:
            return 3
        elif gap <= 5:
            return 2
        elif gap <= 7:
            return 1
        elif gap <= 9:
            return 0.9
        elif gap <= 12:
            return 0.95
        elif gap <= 15:
            return 0.97
        elif gap <= 18:
            return 0.98
        elif gap <= 20:
            return 0.98
        elif gap <= 21:
            return 0.98
        else:
            return 0.99
    return np.array([s(b, item) for b in bins])

def online_binpack(items: np.ndarray, bins: np.ndarray, priority_fn=default_priority) -> tuple[list[list[float]], np.ndarray]:
    packing = [[] for _ in bins]
    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        if len(valid_bin_indices) == 0:
            continue
        priorities = priority_fn(item, bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins

def evaluate(instances: list, priority_fn=default_priority) -> float:
    num_bins = []
    for instance in instances:
        capacity = instance['Bin Capacity']
        items = instance['Items']['Size'].to_numpy()
        bins = np.array([capacity] * len(items))
        _, bins_packed = online_binpack(items, bins, priority_fn=priority_fn)
        num_bins.append((bins_packed != capacity).sum())
    return -np.mean(num_bins)

# === Evaluation Helper ===
def run_evaluation(name: str, instances: list, priority_fn, opt_bins: float):
    avg_bins = -evaluate(instances, priority_fn=priority_fn)
    excess = (avg_bins - opt_bins) / opt_bins
    print(f"\n📊 {name}")
    print(f"  Average bins used: {avg_bins:.2f}")
    print(f"  L1 lower bound: {opt_bins:.2f}")
    print(f"  Excess: {100 * excess:.2f}%")

# === Main Evaluation ===
if __name__ == "__main__":
    # Optional: load datasets if needed for reference
    # datasets = load_datasets("./")  # Not used now, but available

    # === Load saved train and validation sets ===
    with open("train_instances.pkl", "rb") as f:
        train_instances = pickle.load(f)

    with open("val_instances.pkl", "rb") as f:
        val_instances = pickle.load(f)

    # === Run evaluation ===
    print("\n=== Evaluation on Loaded Saved Sets ===")

    # Training set
    train_l1 = l1_bound_dataset(train_instances)
    run_evaluation("Train Set (Default Priority)", train_instances, default_priority, train_l1)
    run_evaluation("Train Set (OR Priority)", train_instances, or_priority, train_l1)

    # Validation set
    val_l1 = l1_bound_dataset(val_instances)
    run_evaluation("Validation Set (Default Priority)", val_instances, default_priority, val_l1)
    run_evaluation("Validation Set (OR Priority)", val_instances, or_priority, val_l1)
