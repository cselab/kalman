import heapq
import numpy as np
import pickle
import random
import hashlib
import evaluate
import re
import multiprocessing


class Memory:

    def __init__(self, capacity):
        self.Hash = set()
        self.capacity = capacity
        self.pattern = r"(def fun(?:_\d+)?\(.*?\):\n.*?return .*?\n)"
        # We store tuples as (-score, item) so that the highest score among the bottom items is at the root.
        self.heap = []
        self.instuction = """
### Task:
Generate **30 UNIQUE correct implementations** of a Python function named `fun`.

### Requirements:
1. **Function Signature**:
- The function must have the exact signature:
    ```python
    def fun(i1, i2, i3, i4):
    ```
- It must accept exactly **four input variables**:
    - `i1`: shape : `(2,)`
    - `i2`: shape : `(2, 2)`
    - `i3`: shape : `(2, 2)`
    - `i4`: shape : `(2, 2)`

2. **Allowed Operations**:
- Only the following operations are allowed: `@` , `+` , `.T`
- **No additional operations** (e.g., arithmetic, conditionals, loops) are allowed.

3. **Uniqueness**:
- Each implementation must use a **unique combination** of the allowed operations.
- You can use the allowed operations multiple times as needed.

4. **Output**:
- The function must return exactly **two outputs**

5. **Code Format**:
- Provide **runnable Python code** for each implementation.
- Do not include any additional text, comment, explanations, or markdown formatting (e.g., ```python ```).

### Make sure you output runable functions in your thinking 
### Examples of Unique Implementations you should modify and combine:

        """
        #### Example 1:
        self.example1 = """
#score : 6.9
def fun(i1, i2, i3, i4):
    temp1 = i1 
    temp2 = i4.T + i3.T
    o1 = temp1
    o2 = temp2
    return o1, o2
            """

        #### Example 2:
        self.example2 = """
#score : 8000
def fun(i1,i2,i3,i4):
    temp1 = i2
    temp2 = i2 + i3 @ i4
    o1 = temp1
    o2 = temp2.T
    return o1, o2
        """

    def create_prompt(self):
        if len(self.heap) == 0:
            return self.instuction + "\n#### Example 1: \n" + self.example1 + "\n#### Example 2: \n" + self.example2
        else:
            samples = self.sample_softmax_inv(num_samples=20)
            return self.instuction + "\n # Try to minimize the loss in the generated examples\n\n#### Example 1: \n" + samples[
                0][1] + "\n loss = " + str(
                    samples[0][0]) + "\n#### Example 2: \n" + samples[1][
                        1] + "\n loss = " + str(
                            samples[1][0])  #+ "\n<think>\n"

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
                replaced_hash = hashlib.sha256(
                    replaced_item.encode()).hexdigest()
                if replaced_hash in self.Hash:
                    self.Hash.remove(replaced_hash)
                self.Hash.add(new_hash)

    def get_sorted(self):
        # Returns items sorted by score (lowest first)
        return sorted([(-s, item) for s, item in self.heap],
                      key=lambda x: x[0])

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
        sorted_entries = self.get_sorted(
        )  # Sorted from best (lowest score) to worst (highest score)
        num_to_keep = n - (n // 2)
        kept_entries = sorted_entries[:num_to_keep]
        for score, item in kept_entries:
            self.add(-score, item)
        # Rebuild self.Hash from the new heap.
        self.Hash = {
            hashlib.sha256(item.encode()).hexdigest()
            for _, item in self.heap
        }

    def remove_duplicates_by_score(self):
        """
        Remove duplicate implementations based solely on score.
        If multiple implementations have the same score, only the first encountered is kept.
        """
        sorted_items = self.get_sorted(
        )  # (score, item) tuples sorted from lowest to highest score
        seen_scores = set()
        unique_items = []
        for score, item in sorted_items:
            if score in seen_scores:
                continue
            seen_scores.add(score)
            unique_items.append((score, item))
        self.heap = [(-score, item) for score, item in unique_items]
        heapq.heapify(self.heap)
        self.Hash = {
            hashlib.sha256(item.encode()).hexdigest()
            for _, item in self.heap
        }

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
        indices = np.random.choice(len(actual_entries),
                                   size=num_samples,
                                   replace=True,
                                   p=probs)
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

    def save_batch(self, llm_outputs):
        total_matches = []
        for llm_output in llm_outputs:
            matches = re.findall(self.pattern, llm_output, re.DOTALL)
            total_matches.extend(matches)
        functions = []
        for i, func in enumerate(total_matches, 1):
            if not self.seen(func):
                functions.append(func)
                score = evaluate.evaluate_graph(func)
                if score == np.inf or np.isnan(score):
                    continue
                self.add(score, func)
        #with multiprocessing.Pool() as pool:
        #    costs = pool.map(evaluate.evaluate_graph, functions)
        #for fun, cost in zip(functions,costs):
        #    if cost == np.inf or np.isnan(cost):
        #        continue
        #    self.add(cost, fun)
