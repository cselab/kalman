import numpy as np
import pickle
import random
import hashlib
import evaluate
import re
import os
import json
from collections import defaultdict

class Memory:
    def __init__(self, capacity, n = 0):
        self.capacity = capacity
        self.Hash = set()
        self.best_score_history = []
        self.clusters = defaultdict(list)
        self.cluster_ids = []
        self.n = n
        self.N = 4000

        self.pattern = r"(def fun(?:_\d+)?\(.*?\):\n.*?return .*?\n)"

        self.instuction = """
### Task:
make this prompt as small as you can but keep the accuracy : Generate **30 UNIQUE correct implementations** of a Python function named fun.

### Requirements:
1. **Function Signature**:
- The function must have the exact signature:
    
python
    def fun(a, b, c, d):

- It must accept exactly **four input variables**:
    - a: a numpy array of shape (2,)
    - b: a numpy array of shape (2, 2)
    - c: a numpy array of shape (2, 2)
    - d: a numpy array of shape (2, 2)

2. **Allowed Operations**:
- Only the following operations are allowed:
    - Matrix multiplication (@)
    - Addition (+)
    - Transposition (.T)
- **No additional operations** (e.g., arithmetic, conditionals, loops) are allowed.

3. **Uniqueness**:
- Each implementation must use a **unique combination** of the allowed operations.
- You can use the allowed operations multiple times as needed.

4. **Output**:
- The function must return exactly **two outputs** (o1, o2), which are numpy arrays.

5. **Code Format**:
- Provide **runnable Python code** for each implementation.
- Do not include any additional text, explanations, or markdown formatting (e.g.,  
python
).

### Make sure you output runable functions in your thinking 
### Examples of Unique Implementations you should modify and combine:
"""

        self.example1 = """
#score : 6.9
def fun(a, b, c, d):
    temp1 = a 
    temp2 = d.T + c.T
    o1 = temp1
    o2 = temp2
    return o1, o2
"""

        self.example2 = """
#score : 8000
def fun(a, b, c, d):
    temp1 = b
    temp2 = b + c @ d
    o1 = temp1
    o2 = temp2.T
    return o1, o2
"""

    def _hash_item(self, item):
        return hashlib.sha256(item.encode()).hexdigest()

    def _get_cluster_id(self, score):
        return score

    def add(self, score, item):
        new_hash = self._hash_item(item)
        if new_hash in self.Hash:
            print(f"[add] Duplicate detected, skipping. Score: {score}")
            return

        cluster_id = self._get_cluster_id(score)
        program_len = len(item)
        print(f"[add] Adding item. Score: {score}, Length: {program_len}, Cluster: {cluster_id}")
        self.clusters[cluster_id].append((score, item, program_len))
        self.Hash.add(new_hash)

        if cluster_id not in self.cluster_ids:
            self.cluster_ids.append(cluster_id)

        total_items = sum(len(c) for c in self.clusters.values())
        if total_items > self.capacity:
            print(f"[add] Capacity exceeded ({total_items} > {self.capacity}). Triggering eviction.")
            self._evict_worst()

    def _evict_worst(self):
        worst_cluster = max(
            self.clusters.items(),
            key=lambda kv: kv[0] if kv[1] else -np.inf
        )[0]
        if self.clusters[worst_cluster]:
            score, item, _ = self.clusters[worst_cluster].pop()
            print(f"[evict] Removing item from cluster {worst_cluster}, score: {score}")
            self.Hash.discard(self._hash_item(item))
            if not self.clusters[worst_cluster]:
                print(f"[evict] Cluster {worst_cluster} is now empty. Removing cluster.")
                del self.clusters[worst_cluster]
                self.cluster_ids.remove(worst_cluster)

    def sample_softmax_inv(self, num_samples=2, T0=0.5, T_program=1.0):
        if not self.clusters or not self.cluster_ids:
            print("[sample] No clusters available.")
            return []

        T_cluster = 0.8
        cluster_scores = self.cluster_ids
        cluster_scores = np.argsort(np.argsort(np.array(self.cluster_ids)))
        cluster_scores = np.log2(cluster_scores + 2)
        cluster_probs = self.softmax(-np.array(cluster_scores), T_cluster)

        print(f"[sample] Sampling from {len(self.cluster_ids)} clusters.")
        results = []
        for _ in range(num_samples):
            cluster_idx = np.random.choice(len(self.cluster_ids), p=cluster_probs)
            cid = self.cluster_ids[cluster_idx]
            items = self.clusters[cid]
            print(f"[sample] Selected cluster {cid} with {len(items)} items.")

            lengths = [length for _, _, length in items]
            min_len = min(lengths)
            max_len = max(lengths)
            norm_lens = [(l - min_len) / (max_len - min_len + 1e-6) for l in lengths]
            inv_lens = [-x for x in norm_lens]
            prog_probs = self.softmax(np.array(inv_lens), T_program)

            idx = np.random.choice(len(items), p=prog_probs)
            score, item, _ = items[idx]
            print(f"[sample] Sampled item with score: {score}")
            results.append((score, item))

        return results

    def get_sorted(self):
        return sorted([item for cluster in self.clusters.values() for item in cluster], key=lambda x: x[0])

    def get_best(self):
        items = self.get_sorted()
        return (items[0][0], items[0][1]) if items else (None, None)

    def remove_half_worst(self):
        items = self.get_sorted()
        keep = items[:len(items)//2]
        print(f"[prune] Keeping {len(keep)} items, removing {len(items) - len(keep)}")
        self._rebuild(keep)

    def remove_duplicates_by_score(self):
        seen_scores = set()
        unique = []
        for score, item, length in self.get_sorted():
            if score not in seen_scores:
                seen_scores.add(score)
                unique.append((score, item, length))
        print(f"[dedup] Reduced to {len(unique)} unique scores")
        self._rebuild(unique)

    def _rebuild(self, items):
        print(f"[rebuild] Rebuilding memory with {len(items)} items")
        self.clusters.clear()
        self.cluster_ids.clear()
        self.Hash.clear()
        for score, item, _ in items:
            self.add(score, item)

    def create_prompt(self):
        samples = self.sample_softmax_inv(num_samples=2)
        if not samples or len(samples) < 2:
            print("[prompt] Not enough examples to sample, using static examples.")
            return self.instuction + "\n#### Example 1: \n" + self.example1 + "\n#### Example 2: \n" + self.example2 + "<think></think>"
        print(f"[prompt] Using sampled examples with scores {samples[0][0]} and {samples[1][0]}")
        return self.instuction + "\n # Try to minimize the loss in the generated examples\n\n#### Example 1: \n" + samples[0][1] + "\n loss = " + str(samples[0][0]) + "\n#### Example 2: \n" + samples[1][1] + "\n loss = " + str(samples[1][0])  + "<think></think>"

    def add_score(self, score):
        print(f"[score] New score recorded: {score}")
        self.best_score_history.append(score)

    def save_batch(self, llm_outputs):
        total_matches = []
        self.n = self.n + 1
        print(f"[batch] Saving batch. Step: {self.n}, LLM outputs: {len(llm_outputs)}")
        print("llm_outputs : ",llm_outputs)
        for llm_output in llm_outputs:
            matches = re.findall(self.pattern, llm_output, re.DOTALL)
            print(len(matches))
            total_matches.extend(matches)
        print(f"[batch] Extracted {len(total_matches)} function candidates.")

        for func in total_matches:
            if not self.seen(func):
                try:
                    score = evaluate.evaluate_graph(func)
                    if score == np.inf or np.isnan(score):
                        print(f"[batch] Invalid score: {score}, skipping.")
                        continue
                    print(f"[batch] Evaluated score: {score}")
                    self.add(score, func)
                except Exception as e:
                    print(f"[batch] Error evaluating function: {e}")

    def seen(self, a):
        return self._hash_item(a) in self.Hash

    def softmax(self, x, temperature=0.2):
        x = np.array(x)
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()

    def sample_batch(self, batch_size):
        print(f"[sample_batch] Sampling {batch_size} prompts.")
        return [self.create_prompt() for _ in range(batch_size)]

    def save(self, filename="memory_save.txt"):
        print(f"[save] Saving memory to {filename}")
        data = {
            "capacity": self.capacity,
            "clusters": dict(self.clusters),
            "Hash": self.Hash,
            "n": self.n
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def load(self, filename="memory_save.txt"):
        print(f"[load] Loading memory from {filename}")
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.capacity = data["capacity"]
        self.clusters = defaultdict(list, data["clusters"])
        self.Hash = data["Hash"]
        self.n = data.get("n", 0)
        self.cluster_ids = list(self.clusters.keys())

    def save_checkpoint(self, checkpoint_dir, thread_id):
        os.makedirs(checkpoint_dir, exist_ok=True)
        mem_file = os.path.join(checkpoint_dir, f"memory_thread_{thread_id}.txt")
        hist_file = os.path.join(checkpoint_dir, f"history_thread_{thread_id}.json")
        print(f"[checkpoint] Saving to {mem_file} and {hist_file}")
        self.save(mem_file)
        with open(hist_file, 'w') as f:
            json.dump(self.best_score_history, f)

    def load_checkpoint(self, checkpoint_dir, thread_id):
        mem_file = os.path.join(checkpoint_dir, f"memory_thread_{thread_id}.txt")
        hist_file = os.path.join(checkpoint_dir, f"history_thread_{thread_id}.json")
        if os.path.exists(mem_file):
            print(f"[checkpoint] Loading memory from {mem_file}")
            self.load(mem_file)
        if os.path.exists(hist_file):
            print(f"[checkpoint] Loading history from {hist_file}")
            with open(hist_file, 'r') as f:
                self.best_score_history = json.load(f)
