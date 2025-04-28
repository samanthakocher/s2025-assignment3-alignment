#!/usr/bin/env python3
import json
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

def parse_arguments():
    parser = argparse.ArgumentParser(description="Measure throughput on AlpacaEval dataset with Qwen2.5")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--alpaca_eval_path", type=str, required=True, help="Path to AlpacaEval dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save model predictions")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum output length")
    parser.add_argument("--baseline_throughput", type=float, default=None, help="Baseline throughput in examples/second")
    return parser.parse_args()

def format_prompt(instruction):
    # Format for Qwen2.5 models - adjust if needed
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def main():
    args = parse_arguments()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load AlpacaEval dataset
    print(f"Loading AlpacaEval dataset from {args.alpaca_eval_path}")
    with open(args.alpaca_eval_path, 'r') as f:
        alpaca_data = json.load(f)
    
    # Prepare for generation
    results = []
    total_time = 0
    total_examples = len(alpaca_data)
    
    print(f"Running inference on {total_examples} examples")
    
    for i in tqdm(range(0, total_examples, args.batch_size)):
        batch = alpaca_data[i:min(i+args.batch_size, total_examples)]
        batch_prompts = [format_prompt(item["instruction"]) for item in batch]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Measure generation time
        start_time = time.time()
        
        with torch.no_grad(), autocast():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_length,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        end_time = time.time()
        batch_time = end_time - start_time
        total_time += batch_time
        
        # Process outputs
        for j, output in enumerate(outputs):
            full_output = tokenizer.decode(output, skip_special_tokens=True)
            # Extract just the assistant's response, handling Qwen format
            response = full_output.split("assistant\n")[-1].strip()
            
            # Store result
            results.append({
                "instruction_id": batch[j].get("id", i+j),
                "instruction": batch[j]["instruction"],
                "output": response,
                "time_seconds": batch_time / len(batch)
            })
    
    # Calculate throughput
    throughput = total_examples / total_time
    
    # Save results
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults:")
    print(f"Total examples: {total_examples}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Throughput: {throughput:.2f} examples/second")
    
    if args.baseline_throughput:
        comparison = (throughput / args.baseline_throughput - 1) * 100
        print(f"Compared to baseline ({args.baseline_throughput:.2f} examples/sec): {comparison:.1f}% {'faster' if comparison > 0 else 'slower'}")

if __name__ == "__main__":
    main()