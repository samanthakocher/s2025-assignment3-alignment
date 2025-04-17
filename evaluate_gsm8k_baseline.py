import os
import json
import argparse
import numpy as np
import re
import time
import torch
import random
from tqdm import tqdm
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5 models on GSM8K zero-shot")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B",
                        help="Model to evaluate (e.g., Qwen/Qwen2.5-0.5B or Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--gsm8k_split", type=str, default="test",
                        help="GSM8K split to evaluate (train or test)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for debugging)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--mmlu_dir", type=str, default="./data/mmlu",
                        help="Directory containing MMLU data (not used for GSM8K evaluation)")
    parser.add_argument("--analyze_errors", action="store_true",
                        help="Perform additional error analysis on incorrect predictions")
    parser.add_argument("--num_error_samples", type=int, default=10,
                        help="Number of incorrect examples to sample for error analysis")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def load_gsm8k_data(split="test", max_samples=None):
    """Load GSM8K dataset."""
    print(f"Loading GSM8K {split} dataset")
    dataset = datasets.load_dataset("gsm8k", "main")[split]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Loaded {len(dataset)} examples from GSM8K {split} split")
    return dataset

def format_prompt(question, is_instruct=False):
    """Format the prompt for zero-shot evaluation."""
    if is_instruct:
        # Format for instruct models
        return f"Solve this math problem step by step:\n\n{question}\n\nAnswer:"
    else:
        # Format for base models
        return f"Question: {question}\n\nAnswer:"

def extract_answer(generated_text):
    """Parse model output to extract the final numerical answer."""
    # Try to find the answer in the format "The answer is X" or similar
    answer_patterns = [
        r"The answer is\s*([\d\.\,\-]+)",
        r"The final answer is\s*([\d\.\,\-]+)",
        r"Therefore, the answer is\s*([\d\.\,\-]+)",
        r"Thus, the answer is\s*([\d\.\,\-]+)",
        r"So, the answer is\s*([\d\.\,\-]+)",
        r"Therefore,\s*([\d\.\,\-]+)",
        r"Thus,\s*([\d\.\,\-]+)",
        r"So the answer is\s*([\d\.\,\-]+)",
        r"So, the result is\s*([\d\.\,\-]+)",
        r"The result is\s*([\d\.\,\-]+)",
        r"=([\d\.\,\-]+)",
        r"= ([\d\.\,\-]+)",
        r"[\$]?(\d+(?:,\d+)*(?:\.\d+)?)"  # Looks for $ followed by numbers, commas, and decimal points
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, generated_text)
        if matches:
            # Return the last match, as it's likely the final answer
            return matches[-1].strip().replace(",", ""), True
    
    # If no patterns match, return empty string and False to indicate parsing failure
    return "", False

def evaluate_correctness(extracted_answer, ground_truth):
    """Check if the extracted answer matches the ground truth."""
    # Clean up both answers
    extracted_clean = extracted_answer.strip().replace(",", "").replace("$", "")
    ground_truth_clean = ground_truth.strip().replace(",", "").replace("$", "")
    
    try:
        extracted_num = float(extracted_clean)
        ground_truth_num = float(ground_truth_clean)
        return abs(extracted_num - ground_truth_num) < 1e-5, True
    except (ValueError, TypeError):
        # If we can't convert to numbers, do string comparison
        return extracted_clean == ground_truth_clean, False

def evaluate_model(model, tokenizer, dataset, args):
    """Evaluate the model on the GSM8K dataset."""
    print("\nEvaluating model on GSM8K")
    
    is_instruct = "instruct" in args.model_name.lower()
    results = []
    correct = 0
    total = 0
    parsing_failures = []
    incorrect_examples = []
    
    generation_times = []
    start_time = time.time()
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        question = example["question"]
        # Extract the answer from the format: "Step 1... Step 2... #### 42"
        ground_truth_full = example["answer"]
        ground_truth = ground_truth_full.split("####")[-1].strip()
        
        # Format prompt
        prompt = format_prompt(question, is_instruct)
        
        # Generate model output and measure time
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        
        gen_start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False  # Use greedy decoding for deterministic outputs
            )
        gen_end_time = time.time()
        gen_time = gen_end_time - gen_start_time
        generation_times.append(gen_time)
        
        # Extract the generated text (excluding the prompt)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        extracted_answer, parsing_succeeded = extract_answer(generated_text)
        
        # Track parsing failures
        if not parsing_succeeded:
            parsing_failures.append({
                "question": question,
                "ground_truth": ground_truth,
                "model_output": generated_text,
                "prompt": prompt
            })
        
        # Evaluate correctness
        is_correct, valid_numbers = evaluate_correctness(extracted_answer, ground_truth)
        
        if is_correct:
            correct += 1
        else:
            incorrect_examples.append({
                "id": idx,
                "question": question,
                "ground_truth_full": ground_truth_full,
                "ground_truth_answer": ground_truth,
                "model_output": generated_text,
                "extracted_answer": extracted_answer,
                "parsing_succeeded": parsing_succeeded
            })
        
        total += 1
        
        # Save result
        result = {
            "id": idx,
            "question": question,
            "ground_truth_full": ground_truth_full,
            "ground_truth_answer": ground_truth,
            "model_output": generated_text,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "parsing_succeeded": parsing_succeeded,
            "valid_numbers": valid_numbers,
            "prompt": prompt,
            "generation_time": gen_time
        }
        results.append(result)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(dataset)}, Accuracy: {correct / (idx + 1):.4f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
    examples_per_second = 1.0 / avg_generation_time if avg_generation_time > 0 else 0
    
    # Calculate metrics
    metrics = {
        "accuracy": correct / total if total > 0 else 0,
        "correct": correct,
        "total": total,
        "parsing_failures": len(parsing_failures),
        "parsing_failure_rate": len(parsing_failures) / total if total > 0 else 0,
        "total_time_seconds": total_time,
        "avg_generation_time_seconds": avg_generation_time,
        "examples_per_second": examples_per_second,
        "model_name": args.model_name,
        "gsm8k_split": args.gsm8k_split,
        "max_samples": args.max_samples if args.max_samples else len(dataset),
        "max_new_tokens": args.max_new_tokens
    }
    
    print(f"\nFinal Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({correct}/{total})")
    print(f"Parsing failures: {len(parsing_failures)}/{total} ({len(parsing_failures)/total*100:.2f}%)")
    print(f"Average generation time: {avg_generation_time:.4f} seconds per example")
    print(f"Throughput: {examples_per_second:.2f} examples/second")
    print(f"Total time: {total_time:.2f} seconds")
    
    return results, metrics, parsing_failures, incorrect_examples, generation_times

def analyze_errors(incorrect_examples, num_samples, seed=42):
    """Analyze a sample of incorrect examples to identify error patterns."""
    if not incorrect_examples:
        return {"error_analysis": "No incorrect examples found."}
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Sample random incorrect examples
    sampled_errors = random.sample(incorrect_examples, min(num_samples, len(incorrect_examples)))
    
    # Categorize errors
    error_categories = {
        "calculation_error": 0,
        "reasoning_error": 0,
        "parsing_error": 0,
        "incomplete_solution": 0,
        "other": 0
    }
    
    # Analyze each example
    for example in sampled_errors:
        model_output = example["model_output"]
        
        # Simple heuristics to categorize errors
        if not example["parsing_succeeded"]:
            error_categories["parsing_error"] += 1
        elif "=" in model_output and len(model_output.split("\n")) > 2:
            # If there are calculations but wrong answer
            error_categories["calculation_error"] += 1
        elif len(model_output.split("\n")) <= 2 or len(model_output) < 50:
            # Very short answers might indicate incomplete solutions
            error_categories["incomplete_solution"] += 1
        elif "step" in model_output.lower() or "first" in model_output.lower():
            # Has steps but still got it wrong
            error_categories["reasoning_error"] += 1
        else:
            error_categories["other"] += 1
    
    analysis = {
        "sampled_incorrect_examples": sampled_errors,
        "error_categories": error_categories,
        "total_sampled": len(sampled_errors)
    }
    
    return analysis

def save_results(results, metrics, parsing_failures, error_analysis, args):
    """Save the evaluation results and metrics to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results
    model_name_safe = args.model_name.replace("/", "-")
    results_path = os.path.join(args.output_dir, f"{model_name_safe}_gsm8k_{args.gsm8k_split}_results.json")
    metrics_path = os.path.join(args.output_dir, f"{model_name_safe}_gsm8k_{args.gsm8k_split}_metrics.json")
    failures_path = os.path.join(args.output_dir, f"{model_name_safe}_gsm8k_{args.gsm8k_split}_parsing_failures.json")
    analysis_path = os.path.join(args.output_dir, f"{model_name_safe}_gsm8k_{args.gsm8k_split}_error_analysis.json")
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
        
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(failures_path, "w") as f:
        json.dump(parsing_failures, f, indent=2)
        
    with open(analysis_path, "w") as f:
        json.dump(error_analysis, f, indent=2)
        
    print(f"Results saved to {results_path}")
    print(f"Metrics saved to {metrics_path}")
    print(f"Parsing failures saved to {failures_path}")
    print(f"Error analysis saved to {analysis_path}")
    
    # Generate a summary report that addresses the assignment questions
    report = generate_summary_report(metrics, parsing_failures, error_analysis)
    report_path = os.path.join(args.output_dir, f"{model_name_safe}_gsm8k_{args.gsm8k_split}_summary_report.txt")
    
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Summary report saved to {report_path}")

def generate_summary_report(metrics, parsing_failures, error_analysis):
    """Generate a summary report answering the assignment questions."""
    report = "GSM8K Zero-Shot Evaluation Summary Report\n"
    report += "=" * 40 + "\n\n"
    
    # 1. Parsing failures
    report += "1. Parsing Failures\n"
    report += "-" * 20 + "\n"
    report += f"Number of model generations that failed parsing: {len(parsing_failures)}\n"
    report += f"Parsing failure rate: {metrics['parsing_failure_rate']:.2%}\n\n"
    
    if parsing_failures:
        report += "Examples of generations that couldn't be parsed:\n\n"
        for i, failure in enumerate(parsing_failures[:3]):  # Show up to 3 examples
            report += f"Example {i+1}:\n"
            report += f"Question: {failure['question']}\n"
            report += f"Ground truth: {failure['ground_truth']}\n"
            report += f"Model output: {failure['model_output']}\n\n"
    
    # 2. Generation time and throughput
    report += "2. Generation Time and Throughput\n"
    report += "-" * 20 + "\n"
    report += f"Average time per example: {metrics['avg_generation_time_seconds']:.4f} seconds\n"
    report += f"Throughput: {metrics['examples_per_second']:.2f} examples/second\n\n"
    
    # 3. Model performance
    report += "3. Model Performance\n"
    report += "-" * 20 + "\n"
    report += f"The {metrics['model_name']} model achieved an accuracy of {metrics['accuracy']:.2%} "
    report += f"({metrics['correct']}/{metrics['total']}) on the GSM8K {metrics['gsm8k_split']} dataset "
    report += f"in a zero-shot setting.\n\n"
    
    # 4. Error analysis
    report += "4. Error Analysis\n"
    report += "-" * 20 + "\n"
    
    if "error_categories" in error_analysis:
        categories = error_analysis["error_categories"]
        total = sum(categories.values())
        report += "Based on analysis of sampled incorrect examples, the model makes the following types of errors:\n"
        for category, count in categories.items():
            if count > 0:
                report += f"- {category.replace('_', ' ').title()}: {count} examples ({count/total:.2%})\n"
        
        report += "\nSample of incorrect examples:\n\n"
        for i, example in enumerate(error_analysis["sampled_incorrect_examples"][:3]):  # Show up to 3 examples
            report += f"Example {i+1}:\n"
            report += f"Question: {example['question']}\n"
            report += f"Ground truth: {example['ground_truth_answer']}\n"
            report += f"Model answer: {example['extracted_answer']}\n"
            report += f"Model output (truncated): {example['model_output'][:200]}...\n\n"
    else:
        report += "No incorrect examples were found for error analysis.\n"
    
    return report

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load the model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
        trust_remote_code=True
    )
    
    # Load GSM8K data
    dataset = load_gsm8k_data(args.gsm8k_split, args.max_samples)
    
    # Evaluate model
    results, metrics, parsing_failures, incorrect_examples, generation_times = evaluate_model(model, tokenizer, dataset, args)
    
    # Analyze errors if requested
    if args.analyze_errors and incorrect_examples:
        error_analysis = analyze_errors(incorrect_examples, args.num_error_samples, args.seed)
    else:
        error_analysis = {"error_analysis": "Error analysis not performed"}
    
    # Save results and generate report
    save_results(results, metrics, parsing_failures, error_analysis, args)

if __name__ == "__main__":
    main()