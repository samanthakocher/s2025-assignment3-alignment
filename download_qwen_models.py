import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 models on AlpacaEval zero-shot")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model to evaluate (e.g., Qwen/Qwen2.5-0.5B or Qwen/Qwen2.5-3B-Instruct)")
    # parser.add_argument("--alpaca_data", type=str, default="./data/alpaca_eval_instructions.json",
    #                     help="Path to AlpacaEval instructions JSON file")
    parser.add_argument("--output_path", type=str, default="./results/qwen_predictions.json",
                        help="Path to save model predictions")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for debugging)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    return parser.parse_args()

def load_alpaca_data():
    return load_dataset(
        "tatsu-lab/alpaca_eval",
        split="eval[:100]",
        download_mode="reuse_dataset_if_exists",
        )

def format_prompt(instruction, model_name):
    """Format the prompt according to model type (base or instruct)."""
    if "Instruct" in model_name:
        # For instruction-tuned models, use their specific prompt format
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    else:
        # For base models, use a simple instruction format
        prompt = f"Instruction: {instruction}\nResponse: "
    return prompt

def generate_responses(model, tokenizer, eval_set, args):
    """Generate model responses for all instructions in the eval set."""
    model.eval()
    
    # Limit samples if specified (for debugging)
    if args.max_samples:
        eval_set = eval_set[:args.max_samples]
    
    # Extract model identifier from path
    model_id = args.model_name.split("/")[-1]
    
    # Process each example
    for example in tqdm(eval_set, desc="Generating responses"):
        instruction = example["instruction"]
        prompt = format_prompt(instruction, args.model_name)
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        # Generate output
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0
            )
        
        # Decode output (skip the prompt)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Store the output in the required format
        example["output"] = generated_text.strip()
        example["generator"] = model_id  # Use model identifier
        
        # Make sure we have dataset field
        if "dataset" not in example:
            example["dataset"] = "alpaca_eval"
    
    return eval_set

def save_results(results, output_path):
    """Save results in the required format for AlpacaEval."""
    # Create output directory if it doesn't exist
    if not isinstance(results, list):
        results = [dict(r) for r in results]  # works for datasets.Dataset

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Load the model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
        trust_remote_code=True
    )

    # Save model and tokenizer locally
    local_dir = "./Qwen2_5_0_5B"
    print(f"Saving model to {local_dir}")
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    
    # Load AlpacaEval data
    # print(f"Loading AlpacaEval data from {args.alpaca_data}")
    eval_set = load_alpaca_data()
    
    # Generate responses
    print(f"Generating responses using {args.model_name}...")
    predictions = generate_responses(model, tokenizer, eval_set, args)
    
    # Save results
    model_name_safe = args.model_name.replace("/", "-")
    output_path = args.output_path.replace(".json", f"_{model_name_safe}.json")
    save_results(predictions, output_path)
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()