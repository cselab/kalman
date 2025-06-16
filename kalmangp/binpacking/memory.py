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
import ast


class Memory:
    def __init__(self, capacity):
        self.Hash = set()
        self.capacity = capacity
        self.pattern = r"def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:\n(?:[ \t]+.*\n)*?[ \t]+return[^\n]*\n"
        # We store tuples as (-score, item) so that the highest score among the bottom items is at the root.
        self.heap = []
        self.instuction = """
### Task:
Generate **10 UNIQUE correct implementations** of a Python function named `priority`.
These functions are heuristics trying to optimize the bin packing problem.

### Requirements:
1. **Function Signature**:
- The function must have the exact signature:
    def priority(item: float, bins: np.ndarray) -> np.ndarray:
- It must accept exactly **two argument variables**:
    - item: float
    - bins: np.ndarray

2. **Uniqueness**:
- Each implementation must use a **unique combination** of allowed operations.
- You can use the allowed operations multiple times as needed.

3. **Output**:
- Array of same size as `bins` with priority score of each bin. Type: np.ndarray

4. **Code Format**:
- Provide **runnable Python code** for each implementation.
- Do not put your code in format ```python:

### Examples of Unique Implementations you should modify and combine:

        """
        #### Example 1: 
        self.example1 = """
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return np.exp(item - bins) * (item * np.std(bins)) + np.sin(np.pi * (item / bins)) + np.sin(np.pi * (item / bins)) + np.sin(np.pi * (item / bins))
"""

        #### Example 2:
        self.example2 = """ 
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return (item * np.std(bins)) * np.exp(item - bins) + np.sin(np.pi * (item / bins))
"""

    def create_prompt(self):
        if len(self.heap) == 0:
            return self.instuction + "\n#### Example 1:\n" + self.example1 + "\n#### Example 2: \n" + self.example2 + "\n"
        else:
            samples = self.sample_softmax_inv()
            return self.instuction + "\n # Try to minimize the loss in the generated examples\n\n#### Example 1:\n" + samples[0][1] + "\n\nloss = " + str(samples[0][0]) + "\n#### Example 2: \n" + samples[1][1] + "\n\nloss = " + str(samples[1][0]) #+ "\n<think>\n"

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



    def is_valid_priority_function(self, code: str) -> bool:
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "priority":
                    return True
        except SyntaxError:
            return False
        return False
    def extract_function_only(self, code: str) -> list[str]:
        print("code : ",code)
        # Match all function blocks
        pattern = r"(def priority\(item: float, bins: np\.ndarray\) -> np\.ndarray:\n(?:[ \t]+.*\n?)*)"
        matches = re.findall(pattern, code)
        print("Found matches:", len(matches))
        if matches:
            return [m.rstrip() for m in matches]
        else:
            raise ValueError("Function definition not found.")

    def clean_priority_extractor(self, text: str) -> list[str]:
        print("=== Raw LLM Output ===")
        print("text:", text)
        print("======================")

        lines = text.splitlines()
        functions = []
        current_func = []
        inside_func = False
        indent_level = None

        for i, line in enumerate(lines):
            print(f"\n[Line {i}] {repr(line)}")

            stripped = line.strip()

            if stripped.startswith("```"):
                print("-> Skipping markdown fence")
                continue

            if stripped.startswith("def priority("):
                print("-> Detected function start")
                if inside_func and current_func:
                    print("-> Closing previous function, attempting to extract...")
                    try:
                        code_block = "\n".join(current_func)
                        print(">>> Function block to extract:\n", code_block)
                        extracted = self.extract_function_only(code_block)
                        functions.extend(extracted)
                        print("✓ Extracted successfully.")
                    except ValueError as e:
                        print("✗ Extraction failed:", e)
                current_func = [line]
                inside_func = True
                indent_level = len(line) - len(line.lstrip())
                print("-> indent_level set to", indent_level)
                continue

            if inside_func:
                if stripped == "":
                    current_func.append(line)
                    print("-> Blank line inside function, added.")
                    continue
                current_indent = len(line) - len(line.lstrip())
                if current_indent > indent_level:
                    current_func.append(line)
                    print("-> Line is inside function block, added.")
                else:
                    print("-> Detected dedent, ending function block.")
                    try:
                        code_block = "\n".join(current_func)
                        print(">>> Function block to extract:\n", code_block)
                        extracted = self.extract_function_only(code_block)
                        functions.extend(extracted)
                        print("✓ Extracted successfully.")
                    except ValueError as e:
                        print("✗ Extraction failed:", e)
                    current_func = []
                    inside_func = False

                    if stripped.startswith("def priority("):
                        print("-> New function begins immediately after dedent")
                        current_func = [line]
                        inside_func = True
                        indent_level = len(line) - len(line.lstrip())
                        print("-> indent_level set to", indent_level)

        if inside_func and current_func:
            print("-> Final function at EOF, attempting to extract...")
            try:
                code_block = "\n".join(current_func)
                print(">>> Final Function block:\n", code_block)
                extracted = self.extract_function_only(code_block)
                functions.extend(extracted)
                print("✓ Final function extracted.")
            except ValueError as e:
                print("✗ Final extraction failed:", e)

        print(f"\n===> Total functions extracted: {len(functions)}")
        return functions


    def sample_batch(self, batch_size):
        batch = []
        for i in range(batch_size):
            batch.append(self.create_prompt())
        return batch


    def save_batch(self, llm_outputs, thread_id):

        total_matches = []
        sys.stdout.write(f"[Process {thread_id}]  in save_batch ")
        sys.stdout.flush()

        try:
            for llm_output in llm_outputs:
                if not isinstance(llm_output, str):
                    continue
                try:
                    sys.stdout.write(f"[Process {thread_id}] before parsing\n")

                    sys.stdout.flush()
                    total_matches.extend(re.findall(self.pattern, llm_output, re.DOTALL))

                    #total_matches.extend(self.clean_priority_extractor(llm_output))

                    sys.stdout.write(f"[Process {thread_id}] after parsing\n")
                    sys.stdout.flush()

                except re.error as regex_err:
                    sys.stdout.write(f"[Regex Error] Invalid pattern: {regex_err}\n")
                    sys.stdout.flush()
                    continue  # Make sure this is inside a loop
                    
                except (Exception, KeyboardInterrupt, SystemExit) as e:
                    sys.stdout.write(f"[Unexpected Error] {type(e).__name__}: {e}\n")
                    sys.stdout.flush()
                    continue
        except Exception as e:
            sys.stdout.write(f"[Unexpected Error] During match extraction: {e}")
            sys.stdout.flush()
            total_matches = []

        sys.stdout.write(f"[Process {thread_id}]  total_matches : {total_matches}")
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
                sys.stdout.write(f"[Process {thread_id}] before evaluate_graph")
                sys.stdout.flush()
                score = evaluate.evaluate_graph(func)
                sys.stdout.write(f"[Process {thread_id}] func :\n {func}")
                sys.stdout.write(f"[Process {thread_id}] score:\n {score}")

                sys.stdout.write(f"[Process {thread_id}] after evaluate_graph")
                sys.stdout.flush()

                if score == np.inf or np.isnan(score):
                    continue
                self.add(score, func)
            except Exception as e:
                sys.stdout.write(f"[Memory] Error evaluating function:\n{func}")
                sys.stdout.flush()
                traceback.print_exc()
