import pickle
import numpy as np
import pandas as pd

def generate_instance(num_items, capacity):
    """Generates a single bin packing instance."""
    # Item sizes are drawn from a uniform distribution between 1 and the capacity.
    item_sizes = np.random.uniform(1, capacity, size=num_items)
    items_df = pd.DataFrame({'Size': item_sizes})
    return {'Items': items_df, 'Bin Capacity': capacity}

def generate_dataset(num_instances_per_config, configs):
    """Generates a dataset of bin packing instances."""
    dataset = []
    for config in configs:
        for _ in range(num_instances_per_config):
            dataset.append(generate_instance(**config))
    return dataset

def main():
    """Generates and saves the training and validation datasets."""
    # Set a seed for reproducibility.
    np.random.seed(42)

    # Define the configurations for the instances.
    # Each configuration is a dictionary of parameters for generate_instance.
    # We will vary the number of items.
    train_configs = [{'num_items': i, 'capacity': 100} for i in range(10, 60, 5)]
    val_configs = [{'num_items': i, 'capacity': 100} for i in range(10, 60, 5)]

    # Generate the datasets.
    # We'll generate 10 instances for each configuration.
    train_instances = generate_dataset(10, train_configs)
    val_instances = generate_dataset(10, val_configs)

    # Save the datasets.
    with open('bipacking/train_instances.pkl', 'wb') as f:
        pickle.dump(train_instances, f)

    with open('bipacking/val_instances.pkl', 'wb') as f:
        pickle.dump(val_instances, f)

    print("Successfully generated and saved train_instances.pkl and val_instances.pkl")

if __name__ == '__main__':
    main()
