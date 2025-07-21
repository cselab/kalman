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

def new_priority(item: float, bins: np.ndarray) -> np.ndarray:
    return (item * (bins ** 2)) * np.exp(item - bins) + np.sin(np.pi * (item / bins)) + np.log(1 + np.sin(np.pi * (item / bins))) + np.mean(bins) * np.log(item + 1) + np.mean(bins) * np.sqrt(item) + np.mean(bins) ** 4 + np.mean(bins) * np.log(1 + item)


# === Main Evaluation ===
if __name__ == "__main__":
    datasets = load_datasets("./")  # Folder with .pkl files

    opt_num_bins = {}
    for name, dataset in datasets.items():
        if name == "binpack5.txt":
            continue 

        instances = dataset if isinstance(dataset, list) else list(dataset.values())[0]
        opt_num_bins[name] = l1_bound_dataset(instances)

    # Evaluate default priority
    for name, dataset in datasets.items():
        if name == "binpack5.txt":
            continue 

        instances = dataset if isinstance(dataset, list) else list(dataset.values())[0]
        avg_num_bins = -evaluate(instances)
        print("\n\n\n++++++++++++++++++++++++++++++++++++++++++++++++\n\n\n\n")
        excess = (avg_num_bins - opt_num_bins[name]) / opt_num_bins[name]
        print(f"\n📊 Dataset: {name} (Default Priority)")
        print(f"  Average number of bins: {avg_num_bins:.2f}")
        print(f"  Lower bound on optimum: {opt_num_bins[name]:.2f}")
        print(f"  Excess: {100 * excess:.2f}%")


        instances = dataset if isinstance(dataset, list) else list(dataset.values())[0]
        avg_num_bins = -evaluate(instances, priority_fn=or_priority)
        excess = (avg_num_bins - opt_num_bins[name]) / opt_num_bins[name]
        print(f"\n📊 Dataset: {name} (funsearch found)")
        print(f"  Average number of bins: {avg_num_bins:.2f}")
        print(f"  Lower bound on optimum: {opt_num_bins[name]:.2f}")
        print(f"  Excess: {100 * excess:.2f}%")

            
        instances = dataset if isinstance(dataset, list) else list(dataset.values())[0]
        avg_num_bins = -evaluate(instances, priority_fn=new_priority)
        excess = (avg_num_bins - opt_num_bins[name]) / opt_num_bins[name]
        print(f"\n📊 Dataset: {name} (new_priority found)")
        print(f"  Average number of bins: {avg_num_bins:.2f}")
        print(f"  Lower bound on optimum: {opt_num_bins[name]:.2f}")
        print(f"  Excess: {100 * excess:.2f}%")
        





import matplotlib.pyplot as plt


plt.rcParams.update({
    'font.size': 30,
    'axes.titlesize': 30,
    'axes.labelsize': 35,
    'xtick.labelsize': 35,
    'ytick.labelsize': 35,
    'legend.fontsize': 25,
    'figure.titlesize': 30
})
# === Visualization ===
dataset_names = []
default_excess = []
funsearch_excess = []
newpriority_excess = []

for name, dataset in datasets.items():
    if name == "binpack5.txt":
        continue 

    instances = dataset if isinstance(dataset, list) else list(dataset.values())[0]
    opt = opt_num_bins[name]

    # Default
    avg_default = -evaluate(instances, priority_fn=default_priority)
    dataset_names.append(name)
    default_excess.append((avg_default - opt) / opt * 100)

    # Funsearch
    avg_fun = -evaluate(instances, priority_fn=or_priority)
    funsearch_excess.append((avg_fun - opt) / opt * 100)

    # New Priority
    avg_new = -evaluate(instances, priority_fn=new_priority)
    newpriority_excess.append((avg_new - opt) / opt * 100)

# Plotting
x = np.arange(len(dataset_names))
width = 0.25

plt.figure(figsize=(16, 9), dpi=1200)  # High-resolution and large figure
plt.bar(x - width, default_excess, width=width, label='Best fit')
plt.bar(x, funsearch_excess, width=width, label='Funsearch')
plt.bar(x + width, newpriority_excess, width=width, label='Our Results')

plt.xticks(x, dataset_names, rotation=45, fontweight='normal')
plt.ylabel('Excess over L1 bound (%)', fontweight='normal')
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save to high-resolution PNG
plt.savefig("binpacking_comparison.pdf", format='pdf', dpi=1200)
plt.savefig("binpacking_comparison.png", format='png', dpi=1200)

plt.close()
