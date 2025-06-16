import heapq
import numpy as np
import pickle
import random
import hashlib
import evaluate
import re
import multiprocessing
import traceback
import sys
from multiprocessing import Process, Queue
from typing import Optional
import subprocess
import tempfile
import json

def regex_worker(pattern: str, text: str, queue: Queue):
    """Worker function to run regex in a subprocess."""
    try:
        matches = re.findall(pattern, text, re.DOTALL)
        queue.put(matches)
    except Exception as e:
        queue.put([])


class Memory:
    def __init__(self, capacity):
        self.Hash = set()
        self.capacity = capacity
        #self.pattern = r"def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:\n(?:[ \t]+.*\n)*?[ \t]+return[^\n]*\n"
        self.pattern = r"^def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:\n(?:[ \t]+[^\n]*\n)+"
        # We store tuples as (-score, item) so that the highest score among the bottom items is at the root.
        self.heap = []
        self.instuction = """
### Task:
Generate **10 UNIQUE correct implementations** of a Python function named `priority`.
These functions a heuristics trying to optimize the binpacking problem
### Requirements:
1. **Function Signature**:
- The function must have the exact signature:
    def priority(item: float, bins: np.ndarray) -> np.ndarray:
- It must accept exactly **two args variables**:
    - bins: np.ndarray
    - item: float, 
2. **Uniqueness**:
- Each implementation must use a **unique combination** of the allowed operations.
- You can use the allowed operations multiple times as needed.

3. **Output**:
- Array of same size as bins with priority score of each bin. type : np.ndarray

4. **Code Format**:
- Provide **runnable Python code** for each implementation.
- Do not include any additional text, comment, explanations, or markdown formatting (e.g., ```python ```).
- I should be able to get this functions with this pattern r"def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:\n(?:[ \t]+.*\n)*?[ \t]+return[^\n]*\n"
### Examples of Unique Implementations you should modify and combine:

        """
        #### Example 1: 
        self.example1 = """
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -(bins - item)
"""

        #### Example 2:
        self.example2 = """ 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return -(bins - item)
"""

    def create_prompt(self):
        if len(self.heap) == 0:
            return self.instuction + "\n#### Example 1: \n" + self.example1 + "\n#### Example 2: \n" + self.example2
        else:
            samples = self.sample_softmax_inv()
            return self.instuction + "\n # Try to minimize the loss in the generated examples\n\n#### Example 1: \n" + samples[0][1] + "\n\nloss = " + str(samples[0][0]) + "\n#### Example 2: \n" + samples[1][1] + "\n\nloss = " + str(samples[1][0]) #+ "\n<think>\n"

    def add(self, score, item):
        """
        Add a new (score, item) entry to the heap.
        Also update self.Hash so it contains only the hashes for items in the heap.
        """
        
        entry = (-score, item)
        new_hash = hashlib.sha256(item.encode()).hexdigest()
        if new_hash in self.Hash:
            return
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
            self.Hash.add(new_hash)
        else:
            # The current best (i.e. the worst among bottom items) is at the root.
            current_best = -self.heap[0][0]
            if score < current_best:
                # heapreplace returns the removed (worst) entry.
                replaced = heapq.heapreplace(self.heap, entry)
                replaced_item = replaced[1]
                replaced_hash = hashlib.sha256(replaced_item.encode()).hexdigest()
                if replaced_hash in self.Hash:
                    self.Hash.remove(replaced_hash)
                self.Hash.add(new_hash)

    def get_sorted(self):
        # Returns items sorted by score (lowest first)
        return sorted([(-s, item) for s, item in self.heap], key=lambda x: x[0])

    def get_best(self):
        """
        Returns the current best (lowest) score and its item
        from the heap without removing it.
        """
        if not self.heap:
            return None, None
        best_score, best_item = self.get_sorted()[0]
        return best_score, best_item

    def remove_half_worst(self):
        """
        Remove the worst 50% of elements from the heap.
        After removal, update self.Hash so it contains only hashes of the remaining items.
        """
        n = len(self.heap)
        if n == 0:
            return
        sorted_entries = self.get_sorted()  # Sorted from best (lowest score) to worst (highest score)
        num_to_keep = n - (n // 2)
        kept_entries = sorted_entries[:num_to_keep]
        for score, item in kept_entries:
            self.add(-score,item)
        # Rebuild self.Hash from the new heap.
        self.Hash = {hashlib.sha256(item.encode()).hexdigest() for _, item in self.heap}
    def remove_duplicates_by_score(self):
        """
        Remove duplicate implementations based solely on score.
        If multiple implementations have the same score, only the first encountered is kept.
        """
        sorted_items = self.get_sorted()  # (score, item) tuples sorted from lowest to highest score
        seen_scores = set()
        unique_items = []
        for score, item in sorted_items:
            if score in seen_scores:
                continue
            seen_scores.add(score)
            unique_items.append((score, item))
        self.heap = [(-score, item) for score, item in unique_items]
        heapq.heapify(self.heap)
        self.Hash = {hashlib.sha256(item.encode()).hexdigest() for _, item in self.heap}

    @staticmethod
    def softmax(x, temperature=0.2):
        """Compute softmax values for a list of numbers with a temperature parameter."""
        x = np.array(x)
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()

    def sample_softmax_inv(self, num_samples=2):
        """
        Sample items using softmax(1/score) probabilities.
        Lower scores will have higher probability.
        Returns:
            List of sampled (score, item) tuples.
        """
        actual_entries = [(-s, item) for s, item in self.heap]
        scores = [score for score, _ in actual_entries]
        inv_scores = [1.0 / score for score in scores]
        probs = Memory.softmax(inv_scores)
        #if len(actual_entries) < num_samples:
        #    print("We catchedd the error")
        #    print("num_samples : ",num_samples)
        #    print("len(actual_entries) : ",len(actual_entries))
        indices = np.random.choice(len(actual_entries), size=num_samples, replace=True, p=probs)
        return [actual_entries[i] for i in indices]

    def save(self, filename="memory_save.txt"):
        """
        Save the current memory state to a file.
        """
        with open(filename, "wb") as f:
            pickle.dump({'capacity': self.capacity, 'heap': self.heap}, f)
        with open("seen_solutions.pkl", "wb") as f:
            pickle.dump(self.Hash, f)

    

    @classmethod
    def load(cls, filename="memory_save.txt"):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        obj = cls(data['capacity'])
        obj.heap = data['heap']
        with open("seen_solutions.pkl", "rb") as f:
            obj.Hash = pickle.load(f)
        return obj

    def seen(self, a):
        """
        Check if the function a (its hash) is already present in the heap.
        """
        a_hash = hashlib.sha256(a.encode()).hexdigest()
        return a_hash in self.Hash

    def sample_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.create_prompt())
        return batch


    


    def safe_extract_matches(self, pattern: str, text: str, timeout: int = 3) -> list[str]:
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as tmp:
            tmp.write(f"""
import re, json
pattern = {repr(pattern)}
text = {repr(text)}
matches = re.findall(pattern, text, re.MULTILINE)
print(json.dumps(matches))
    """)
            tmp.flush()

        try:
            result = subprocess.run(
                [sys.executable, tmp.name],
                capture_output=True,
                timeout=timeout,
                text=True
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                print("⚠️ Subprocess regex error:", result.stderr)
                return []
        except subprocess.TimeoutExpired:
            print("⚠️ Subprocess timed out.")
            return []
    #
    def save_batch(self, llm_outputs, thread_id):
        total_matches = []
        sys.stdout.write("[Process {thread_id}]  in save_batch ")
        sys.stdout.flush()

        try:
            for llm_output in llm_outputs:
                if not isinstance(llm_output, str):
                    continue
                try:
                    total_matches.extend(self.safe_extract_matches(self.pattern, llm_output, timeout=45))
                except re.error as regex_err:
                    sys.stdout.write(f"[Regex Error] Invalid pattern: {regex_err}")
                    sys.stdout.flush()
                    break
        except Exception as e:
            sys.stdout.write(f"[Unexpected Error] During match extraction: {e}")
            sys.stdout.flush()
            total_matches = []

        sys.stdout.write(f"[Process {thread_id}]  total_matches : {len(total_matches)}")
        sys.stdout.flush()

        sys.stdout.write(f"[Process {thread_id}] before evaluate_graphs")
        sys.stdout.flush()
        for func in total_matches:
            try:
                if callable(self.seen) and self.seen(func):
                    continue
            except Exception as e:
                sys.stdout.write(f"[Seen Check Error] Could not check function: {e}")
                sys.stdout.flush()
                continue
            try:
                score = evaluate.evaluate_graph(func)
                sys.stdout.write(f"function:\n{func}")
                sys.stdout.write(f"score::\n{score}")
                sys.stdout.flush()

                if score == np.inf or np.isnan(score):
                    continue
                self.add(score, func)
            except Exception as e:
                sys.stdout.write(f"[Memory] Error evaluating function:\n{func}")
                sys.stdout.flush()
                traceback.print_exc()

        sys.stdout.write(f"[Process {thread_id}] after evaluate_graphs")
        sys.stdout.flush()
