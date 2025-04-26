import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import requests
from pathlib import Path
from datasets import load_dataset
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Generate zero-shot outputs on AlpacaEval using Qwen2.5-0.5B")
    parser.add_argument("--input_file", type=str, default=None, 
                        help="Path to the AlpacaEval dataset (if not provided, will download automatically)")
    parser.add_argument("--output_file", type=str, default="qwen2.5_0.5b_predictions.json", 
                        help="Filename to save model predictions")
    parser.add_argument("--output_dir", type=str, default="./", 
                        help="Directory to save model predictions")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", 
                        help="Hugging Face model name")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--generator_name", type=str, default="qwen2.5-0.5b-base", 
                        help="Name identifier for the generator model")
    parser.add_argument("--max_samples", type=int, default=None,
                    help="Maximum number of samples to evaluate (for debugging)")

    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    print(f"Loading model and tokenizer from local path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    return model, tokenizer


def load_eval_set(_=None):
    print("Loading AlpacaEval from Hugging Face datasets (optimized)...")
    eval_path = "data/alpaca_eval/alpaca_eval.jsonl"
    data = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_outputs(model, tokenizer, eval_set, args):
    """Generate outputs for each instruction in the evaluation set."""
    results = []

    if args.max_samples is not None:
        eval_set = eval_set[:args.max_samples]

    
    for example in tqdm(eval_set, desc="Generating outputs"):
        instruction = example["instruction"]
        
        # Prepare input for the model
        inputs = tokenizer(instruction, return_tensors="pt").to(model.device)
        
        # Generate with greedy decoding
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False
            )
        
        # Decode the output and extract only the generated part
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # The output might include the instruction, so we need to extract just the generated part
        generated_text = full_output[len(instruction):].strip()
        
        # Create result entry
        result = {
            "instruction": instruction,
            "output": generated_text,
            "generator": args.generator_name,
            "dataset": example.get("dataset", "")  # Use empty string if dataset field is not available
        }
        
        results.append(result)
    
    return results

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine the full output file path
    output_file_path = os.path.join(args.output_dir, args.output_file)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Load evaluation set
    eval_set = load_eval_set(args.input_file)
    
    # Generate outputs
    results = generate_outputs(model, tokenizer, eval_set, args)
    
    # Save results
    print(f"Saving {len(results)} predictions to: {output_file_path}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Done!")

if __name__ == "__main__":
    main()