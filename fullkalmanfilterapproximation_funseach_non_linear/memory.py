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


class Memory:
    def __init__(self, capacity):
        self.Hash = set()
        self.capacity = capacity
        self.pattern = r"def aproximate\(x, F, P, Q, z, R\):\n(?:[ \t]+.*\n)*?[ \t]+return[^\n]*"
        self.heap = []
        self.instuction = """
### Task:
# Generate 10 unique implementations of a Python function named aproximate.

# Requirements:
# 1. Function Signature:
#     def aproximate(x, F, P, Q, z, R):
#
# 2. Output:
#     Each function must return a tuple of six NumPy arrays.
#
# 3. Constraints:
# - Use NumPy operations creatively.
# - Do not include comments or markdown formatting inside the functions.
#
# 4. Additional Notes:
# - All returned outputs must be NumPy arrays with matching dimensions where appropriate.
# - You may introduce global variables in the function to track state or trajectory (there is a global list history for that reason), but keep the implementations minimal and elegant.
"""

        self.example1 = """
def aproximate(x, F, P, Q, z, R):
    x = np.array([
        0.05 * x[0]**3-2*x[0],
        0.1 * np.sin(x[1])
    ])
    xp = F @ x
    P = F @ P @ F.T + Q
    y = z - xp
    S = P + R
    K = P @ np.linalg.inv(S)
    x = xp + K @ y
    P = (np.eye(F.shape[0]) - K) @ P
    return xp, P, y, S, K, x
"""

        self.example2 = """
def aproximate(x, F, P, Q, z, R):
    import numpy as np
    x = np.array([
        0.05 * x[0]**3-2*x[0],
        0.1 * np.sin(x[1])
    ])
    c=1.0
    xp = F @ x
    P = F @ P @ F.T + Q
    y = z - xp
    S = P + R
    mahalanobis = y.T @ np.linalg.inv(S) @ y
    scale = 1 + c * np.clip(mahalanobis - 1, 0, None)
    R_adj = R * scale
    S_adj = P + R_adj
    K = P @ np.linalg.inv(S_adj)
    x = xp + K @ y
    P = (np.eye(F.shape[0]) - K) @ P
    return xp, P, y, S_adj, K, x
"""

    def create_prompt(self):
        if len(self.heap) == 0:
            return self.instuction + "\n#### Example 1: \n" + self.example1 + "\n#### Example 2: \n" + self.example2
        else:
            samples = self.sample_softmax_inv()
            return self.instuction + "\n # Try to minimize the loss in the generated examples\n\n#### Example 1: \n" + samples[0][1] + "\n\nloss = " + str(samples[0][0]) + "\n#### Example 2: \n" + samples[1][1] + "\n\nloss = " + str(samples[1][0])

    def add(self, score, item):
        entry = (-score, item)
        new_hash = hashlib.sha256(item.encode()).hexdigest()
        if new_hash in self.Hash:
            return
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, entry)
            self.Hash.add(new_hash)
        else:
            current_best = -self.heap[0][0]
            if score < current_best:
                replaced = heapq.heapreplace(self.heap, entry)
                replaced_hash = hashlib.sha256(replaced[1].encode()).hexdigest()
                if replaced_hash in self.Hash:
                    self.Hash.remove(replaced_hash)
                self.Hash.add(new_hash)

    def get_sorted(self):
        return sorted([(-s, item) for s, item in self.heap], key=lambda x: x[0])

    def get_best(self):
        if not self.heap:
            return None, None
        return self.get_sorted()[0]

    def remove_half_worst(self):
        n = len(self.heap)
        if n == 0:
            return
        sorted_entries = self.get_sorted()
        num_to_keep = n - (n // 2)
        for score, item in sorted_entries[:num_to_keep]:
            self.add(-score, item)
        self.Hash = {hashlib.sha256(item.encode()).hexdigest() for _, item in self.heap}

    def remove_duplicates_by_score(self):
        sorted_items = self.get_sorted()
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
        x = np.array(x)
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum()

    def sample_softmax_inv(self, num_samples=2):
        actual_entries = [(-s, item) for s, item in self.heap]
        scores = [score for score, _ in actual_entries]
        inv_scores = [1.0 / score for score in scores]
        probs = Memory.softmax(inv_scores)
        indices = np.random.choice(len(actual_entries), size=num_samples, replace=True, p=probs)
        return [actual_entries[i] for i in indices]

    def save(self, filename="memory_save.txt"):
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
        return hashlib.sha256(a.encode()).hexdigest() in self.Hash

    def sample_batch(self, batch_size):
        return [self.create_prompt() for _ in range(batch_size)]

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

    def save_batch(self, llm_outputs, thread_id):
        total_matches = []
        sys.stdout.write(f"[Process {thread_id}] in save_batch\n")
        sys.stdout.flush()

        try:
            for llm_output in llm_outputs:
                if not isinstance(llm_output, str):
                    continue
                try:
                    total_matches.extend(self.safe_extract_matches(self.pattern, llm_output, timeout=45))
                except re.error as regex_err:
                    sys.stdout.write(f"[Regex Error] Invalid pattern: {regex_err}\n")
                    sys.stdout.flush()
                    break
        except Exception as e:
            sys.stdout.write(f"[Unexpected Error] During match extraction: {e}\n")
            sys.stdout.flush()
            total_matches = []

        sys.stdout.write(f"[Process {thread_id}] total_matches: {len(total_matches)}\n")
        sys.stdout.flush()

        sys.stdout.write(f"[Process {thread_id}] before evaluate_graphs\n")
        sys.stdout.flush()
        for func in total_matches:
            try:
                if self.seen(func):
                    continue
            except Exception as e:
                sys.stdout.write(f"[Seen Check Error] Could not check function: {e}\n")
                sys.stdout.flush()
                continue
            try:
                score = evaluate.evaluate_graph(func)
                sys.stdout.write(f"function:\n{func}\n")
                sys.stdout.write(f"score: {score}\n")
                sys.stdout.flush()
                if score == np.inf or np.isnan(score):
                    continue
                self.add(score, func)
            except Exception as e:
                sys.stdout.write(f"[Memory] Error {e} evaluating function:\n{func}\n")
                sys.stdout.flush()
                traceback.print_exc()

        sys.stdout.write(f"[Process {thread_id}] after evaluate_graphs\n")
        sys.stdout.flush()

