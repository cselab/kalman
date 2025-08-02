import argparse
import time
import torch
import random
import pickle
import queue
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch.multiprocessing as mp
from multiprocessing import Pool
import sys

import memory  # your custom memory module

def inference_worker(args):
    thread_id, mem, model_name, cache_dir, gpu_id, iterations = args

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True, padding_side='left')

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        use_auth_token='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'
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_auth_token='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'
    ).to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model.eval()

    sys.stdout.write(f"[Process {thread_id}] started on GPU {gpu_id}, running {iterations} iterations...\n")
    sys.stdout.flush()
        
    for step in range(iterations): 
        try:      
            prompt = mem.sample_batch(30)
            sys.stdout.write(f"[Process {thread_id}] step={step}, sampling done\n")
            sys.stdout.flush()

            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            sys.stdout.write(f"[Process {thread_id}] step={step}, text tokenized\n")
            sys.stdout.flush()

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
            inputs = inputs.to(device)

            start_time = time.time()
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_length=3024)
            end_time = time.time()
            sys.stdout.write(f"[Process {thread_id}] step={step}, Generation done\n")
            sys.stdout.flush()

            elapsed = end_time - start_time

            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            sys.stdout.write(f"[Process {thread_id}] step={step}, text decoded\n")
            sys.stdout.flush()

            mem.save_batch(generated_texts, thread_id)
            sys.stdout.write(f"[Process {thread_id}] step={step}, generation took {elapsed:.2f}s \n")
            sys.stdout.flush()
            best_score, best_item = mem.get_best()
            sys.stdout.write(f"[Process {thread_id}] best score so far: {best_score}, content:\n{best_item} \n")
            sys.stdout.flush()
        except RuntimeError as re:
            print(f"[Process {thread_id}] RuntimeError during generation:")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()  # Emergency clear
        except Exception as e:
            print(f"[Process {thread_id}] Unexpected exception during generation:")
            import traceback
            traceback.print_exc()

    return thread_id, mem

def main():
    parser = argparse.ArgumentParser(description="Outer loop reinitializes worst memories.")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--outer_iterations", type=int, default=1000)
    args = parser.parse_args()
    #deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
    #Qwen/QwQ-32B
    #deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    #mistralai/Mistral-Small-3.1-24B-Instruct-2503
    #google/codegemma-7b-pytorch
    #google/codegemma-7b
    #deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    cache_dir = ".cache"

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs found.")
    print(f"Detected {num_gpus} GPU(s). We will spawn {args.num_threads} processes.\n")

    memories = [memory.Memory(capacity=200) for _ in range(args.num_threads)]

    for outer_iter in range(args.outer_iterations):
        print(f"\n==== Outer Iteration {outer_iter} ====")
        args_list = [
            (i, memories[i], model_name, cache_dir, i % num_gpus, args.iterations)
            for i in range(args.num_threads)
        ]

        with Pool(processes=args.num_threads) as pool:
            results = pool.map(inference_worker, args_list)

        for thread_id, mem in results:
            memories[thread_id] = mem

        print("\n[Main] Collecting best scores from each thread's memory:")
        memory_infos = []
        for idx, mem in enumerate(memories):
            score, item = mem.get_best()
            if score is not None:
                memory_infos.append((score, idx, item))
                print(f"  Thread {idx}: best score {score}")
            else:
                print(f"  Thread {idx}: memory is empty.")

        if memory_infos:
            memory_infos.sort(key=lambda x: x[0])
            print("\n[Main] Sorted memory_infos (score, thread_idx):")
            for info in memory_infos:
                print(f"  {info[0]}, thread {info[1]}")

            n = len(memory_infos)
            worst_half = memory_infos[n // 2:]
            donor_candidates = memory_infos[:n // 2]

            sys.stdout.write("\n[Main] Worst half (to be reinitialized):")
            sys.stdout.flush()
            for info in worst_half:
                print(f"  Thread {info[1]} with score {info[0]}")
            print("\n[Main] Donor candidates (top half):")
            sys.stdout.write("\n[Main] Donor candidates (top half):")
            sys.stdout.flush()

            for info in donor_candidates:
                print(f"  Thread {info[1]} with score {info[0]}")

            if donor_candidates:
                for donor_idx, (score, idx, item) in enumerate(worst_half):
                    donor = donor_candidates[min(donor_idx, len(donor_candidates) - 1)]
                    donor_score, donor_idx, donor_item = donor
                    print(f"[Main] Reinitializing memory for thread {idx} with best element from thread {donor_idx}.")
                    new_mem = memory.Memory(capacity=400 )
                    new_mem.add(donor_score, donor_item)
                    memories[idx] = new_mem
            else:
                print("[Main] Not enough donor candidates; skipping reinitialization.")
        else:
            print("[Main] All memories are empty or missing; skipping reinitialization.")

    print("\nAll outer iterations have finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
