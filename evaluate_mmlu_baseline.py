import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 models on MMLU zero-shot")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model to evaluate (e.g., Qwen/Qwen2.5-0.5B or Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--mmlu_dir", type=str, default="./data/mmlu",
                        help="Directory containing MMLU data")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate per task (for debugging)")
    return parser.parse_args()

def load_mmlu_data(mmlu_dir):
    """Load MMLU test data for all available subjects."""
    tasks = {}
    
    # MMLU has dev, val, and test splits
    split = "test"
    
    # Get all subject files in the directory
    subject_files = []
    for root, _, files in os.walk(mmlu_dir):
        for file in files:
            if file.endswith(f"{split}.csv"):
                subject = file.replace(f"_{split}.csv", "")
                subject_files.append((subject, os.path.join(root, file)))
    
    # Load each subject file
    for subject, file_path in subject_files:
        try:
            df = pd.read_csv(file_path)
            df.columns = ["question", "A", "B", "C", "D", "answer"]
            tasks[subject] = df
            print(f"Loaded {len(df)} examples for subject: {subject}")
        except Exception as e:
            print(f"Error loading {subject}: {e}")
            
    return tasks

def format_prompt(question, choices):
    """Format the prompt for zero-shot evaluation."""
    prompt = f"Question: {question}\n\n"
    for choice_key, choice_value in choices.items():
        prompt += f"{choice_key}. {choice_value}\n"
    prompt += "\nAnswer:"
    return prompt

def extract_answer(response):
    """Parse model output to extract the answer."""
    # Look for a single letter answer (A, B, C, or D)
    response = response.strip()
    
    # First check if the first token is the answer
    if response and response[0] in "ABCD":
        return response[0], True
    
    # Check for patterns like "The answer is A"
    for choice in "ABCD":
        patterns = [
            f"answer is {choice}",
            f"answer: {choice}",
            f"choose {choice}",
            f"selected {choice}",
            f"option {choice}",
            f"{choice} is correct",
            f"I choose {choice}",
            f"I select {choice}",
            f"I pick {choice}"
        ]
        for pattern in patterns:
            if pattern.lower() in response.lower():
                return choice, True
    
    # Return the first occurrence of A, B, C, or D in the response
    for char in response:
        if char in "ABCD":
            return char, True
    
    # Return empty string and False to indicate parsing failure
    return "", False

def evaluate_model(model, tokenizer, tasks, args):
    """Evaluate the model on all tasks."""
    all_results = {}
    all_metrics = {}
    parsing_failures = []
    start_time = time.time()
    total_examples = 0
    
    for subject, examples in tasks.items():
        print(f"\nEvaluating subject: {subject}")
        
        # Limit number of samples if specified
        if args.max_samples:
            examples = examples.head(args.max_samples)
        
        subject_results = []
        correct = 0
        total = 0
        subject_failures = 0
        
        for _, row in tqdm(examples.iterrows(), total=len(examples)):
            # Format the prompt
            question = row["question"]
            choices = {"A": row["A"], "B": row["B"], "C": row["C"], "D": row["D"]}
            ground_truth = row["answer"]
            prompt = format_prompt(question, choices)
            
            # Generate model output
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    temperature=0.0  # Use greedy decoding for deterministic outputs
                )
            
            generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            model_answer, parsing_succeeded = extract_answer(generated_text)
            
            # Track parsing failures
            if not parsing_succeeded:
                subject_failures += 1
                parsing_failures.append({
                    "subject": subject,
                    "question": question,
                    "choices": choices,
                    "ground_truth": ground_truth,
                    "model_output": generated_text,
                    "prompt": prompt
                })
            
            # Check if the answer is correct
            is_correct = model_answer == ground_truth
            if is_correct:
                correct += 1
            total += 1
            
            # Save result
            result = {
                "question": question,
                "choices": choices,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "is_correct": is_correct,
                "prompt": prompt,
                "model_output": generated_text,
                "parsing_succeeded": parsing_succeeded
            }
            subject_results.append(result)
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "parsing_failures": subject_failures,
            "parsing_failure_rate": subject_failures / total if total > 0 else 0
        }
        
        all_results[subject] = subject_results
        all_metrics[subject] = metrics
        
        print(f"Subject: {subject}, Accuracy: {accuracy:.4f} ({correct}/{total}), Parsing failures: {subject_failures}")
    
    # Calculate average accuracy across all subjects
    subject_accuracies = [metrics["accuracy"] for metrics in all_metrics.values()]
    avg_accuracy = sum(subject_accuracies) / len(subject_accuracies) if subject_accuracies else 0
    
    total_failures = sum(metrics["parsing_failures"] for metrics in all_metrics.values())
    total_examples = sum(metrics["total"] for metrics in all_metrics.values())
    
    all_metrics["average"] = {
        "accuracy": avg_accuracy,
        "total_parsing_failures": total_failures,
        "total_examples": total_examples,
        "parsing_failure_rate": total_failures / total_examples if total_examples > 0 else 0
    }
    
    end_time = time.time()
    total_time = end_time - start_time
    examples_per_second = total_examples / total_time

    print(f"\nAverage accuracy across {len(tasks)} subjects: {avg_accuracy:.4f}")
    print(f"Total parsing failures: {total_failures}/{total_examples} ({total_failures/total_examples*100:.2f}%)")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Throughput: {examples_per_second:.2f} examples/second")
    
    return all_results, all_metrics, parsing_failures

def save_results(all_results, all_metrics, args):
    """Save the evaluation results and metrics to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    model_name_safe = args.model_name.replace("/", "-")
    results_path = os.path.join(args.output_dir, f"{model_name_safe}_mmlu_results.json")
    metrics_path = os.path.join(args.output_dir, f"{model_name_safe}_mmlu_metrics.json")
    
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
        
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
        
    print(f"Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")
    
    # Create a summary CSV for easier analysis
    summary_data = []
    for subject, metrics in all_metrics.items():
        if subject != "average":
            summary_data.append({
                "subject": subject,
                "accuracy": metrics["accuracy"],
                "correct": metrics["correct"],
                "total": metrics["total"]
            })
    
    # Add average at the end
    summary_data.append({
        "subject": "AVERAGE",
        "accuracy": all_metrics["average"]["accuracy"],
        "correct": "-",
        "total": "-"
    })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(args.output_dir, f"{model_name_safe}_mmlu_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

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
    
    # Load MMLU data
    print(f"Loading MMLU data from {args.mmlu_dir}")
    tasks = load_mmlu_data(args.mmlu_dir)
    print(f"Loaded {len(tasks)} MMLU subjects")
    
    # Evaluate model
    all_results, all_metrics, parsing_failures = evaluate_model(model, tokenizer, tasks, args)
    
    # Save results
    save_results(all_results, all_metrics, args)
    
    # Save parsing failures
    model_name_safe = args.model_name.replace("/", "-")
    failures_path = os.path.join(args.output_dir, f"{model_name_safe}_parsing_failures.json")
    with open(failures_path, "w") as f:
        json.dump(parsing_failures, f, indent=2)
    print(f"Parsing failures saved to {failures_path}")

if __name__ == "__main__":
    main()