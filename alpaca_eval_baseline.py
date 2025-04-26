import json
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import requests
from pathlib import Path
from datasets import load_dataset

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
    return parser.parse_args()

def load_model_and_tokenizer(model_name):
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    return model, tokenizer


def download_alpaca_eval_data():
    """Download the AlpacaEval dataset if it doesn't exist."""
    data_dir = Path("alpaca_eval_data")
    data_dir.mkdir(exist_ok=True)
    
    data_file = data_dir / "alpaca_eval_instructions.json"
    
    if not data_file.exists():
        print("Downloading AlpacaEval dataset...")
        # URL for the AlpacaEval dataset
        url = "https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/alpaca_eval/data/alpaca_eval.json"
        
        # Download the file
        response = requests.get(url)
        if response.status_code == 200:
            with open(data_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Dataset downloaded to {data_file}")
        else:
            raise Exception(f"Failed to download dataset: status code {response.status_code}")
    else:
        print(f"Using existing dataset at {data_file}")
    
    return str(data_file)

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
    model, tokenizer = load_model_and_tokenizer("./Qwen_Qwen2_5_0_5B")
    
    # Load evaluation set
    eval_set = load_eval_set(args.input_file)
    
    # Generate outputs
    results = generate_outputs(model, tokenizer, eval_set, args)
    
    # Save results
    print(f"Saving {len(results)} predictions to: {output_file_path}")
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Evaluating Winrate against GPT-4 Turbo using Qwen2.5-3B-Instruct as annotator ===")

    # Load GPT-4 Turbo reference outputs
    gpt4_path = "alpaca_eval/data/gpt4_turbo_outputs.json"
    assert os.path.exists(gpt4_path), "Missing GPT-4 Turbo reference outputs."

    with open(gpt4_path) as f:
        gpt4_outputs = json.load(f)
    gpt4_lookup = {x["instruction"]: x["output"] for x in gpt4_outputs}

    # Load Qwen2.5-3B-Instruct annotator
    annotator_model = AutoModelForCausalLM.from_pretrained("./Qwen_Qwen2_5_3B_Instruct", local_files_only=True).to("cuda")
    annotator_tokenizer = AutoTokenizer.from_pretrained("./Qwen_Qwen2_5_3B_Instruct", local_files_only=True)

    def judge_response(prompt, model_output, gpt4_output):
        """Use Qwen2.5-3B-Instruct to choose between model and GPT-4 Turbo."""
        comparison_prompt = (
            f"Instruction: {prompt}\n\n"
            f"Response A: {model_output}\n\n"
            f"Response B: {gpt4_output}\n\n"
            f"Which response is better? Answer A or B."
        )

        inputs = annotator_tokenizer(comparison_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = annotator_model.generate(**inputs, max_new_tokens=10)
        decoded = annotator_tokenizer.decode(out[0], skip_special_tokens=True)
        return "A" if "A" in decoded else "B"

    wins = 0
    total = 0
    length_matched_wins = 0

    for result in tqdm(results, desc="Judging outputs"):
        instruction = result["instruction"]
        model_output = result["output"]
        gpt4_output = gpt4_lookup.get(instruction)
        if not gpt4_output:
            continue

        decision = judge_response(instruction, model_output, gpt4_output)
        if decision == "A":
            wins += 1
            if len(model_output) <= len(gpt4_output) * 1.1:  # <=10% longer
                length_matched_wins += 1
        total += 1

    # Final report
    winrate = 100 * wins / total
    length_winrate = 100 * length_matched_wins / total
    print(f"\nWinrate: {winrate:.2f}%")
    print(f"Length-controlled winrate: {length_winrate:.2f}%")


    print("Done!")

if __name__ == "__main__":
    main()