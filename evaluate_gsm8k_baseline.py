import os
import json
import re
import time
import torch
import numpy as np
import datasets
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any, Tuple

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"  # Replace with "Qwen/Qwen2.5-3B-Instruct" as needed
OUTPUT_DIR = "gsm8k_evaluation_results"
SPLIT = "test"  # Can be "train" or "test"
MAX_SAMPLES = None  # Set to a number for testing with fewer examples
USE_FP16 = True  # Set to True for mixed precision
MAX_NEW_TOKENS = 512  # Maximum number of tokens to generate
EVAL_BATCH_SIZE = 1  # Adjust based on GPU memory

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer."""
    print(f"Loading model and tokenizer: {model_name}")
    
    # Determine if it's an instruct model
    is_instruct = "instruct" in model_name.lower()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32,
        device_map="auto"
    )
    
    return model, tokenizer, is_instruct

def load_gsm8k_dataset(split: str = "test", max_samples: int = None) -> List[Dict[str, Any]]:
    """Load GSM8K dataset."""
    print(f"Loading GSM8K {split} dataset")
    dataset = datasets.load_dataset("gsm8k", "main")[split]
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    return dataset

def format_prompt(question: str, is_instruct: bool) -> str:
    """Format the prompt based on model type."""
    if is_instruct:
        # Format for instruct models
        return f"Solve this math problem step by step:\n\n{question}\n\nAnswer:"
    else:
        # Format for base models
        return f"Question: {question}\n\nAnswer:"

def extract_answer(generated_text: str) -> str:
    """Extract the final answer from generated text."""
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
            return matches[-1].strip().replace(",", "")
    
    # If no patterns match, return the last line that contains a number
    lines = generated_text.split('\n')
    for line in reversed(lines):
        if re.search(r'\d', line):
            # Extract any number from the line
            number_match = re.search(r'[\$]?(\d+(?:,\d+)*(?:\.\d+)?)', line)
            if number_match:
                return number_match.group(1).replace(",", "")
    
    return ""

def evaluate_correctness(extracted_answer: str, ground_truth: str) -> bool:
    """Check if the extracted answer matches the ground truth."""
    # Clean up both answers
    extracted_clean = extracted_answer.strip().replace(",", "").replace("$", "")
    ground_truth_clean = ground_truth.strip().replace(",", "").replace("$", "")
    
    try:
        extracted_num = float(extracted_clean)
        ground_truth_num = float(ground_truth_clean)
        return abs(extracted_num - ground_truth_num) < 1e-5
    except (ValueError, TypeError):
        # If we can't convert to numbers, do string comparison
        return extracted_clean == ground_truth_clean

def process_gsm8k_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: List[Dict[str, Any]],
    is_instruct: bool
) -> List[Dict[str, Any]]:
    """Process GSM8K examples and generate model responses."""
    results = []
    correct_count = 0
    
    for idx, example in enumerate(tqdm(dataset)):
        question = example["question"]
        ground_truth_answer = example["answer"].split("####")[-1].strip()
        
        # Format the prompt
        prompt = format_prompt(question, is_instruct)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Deterministic generation
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer part (remove the prompt from the beginning)
        answer_part = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
        
        # Extract final answer
        extracted_answer = extract_answer(answer_part)
        
        # Evaluate correctness
        is_correct = evaluate_correctness(extracted_answer, ground_truth_answer)
        if is_correct:
            correct_count += 1
        
        # Store results
        result = {
            "id": idx,
            "question": question,
            "ground_truth": example["answer"],
            "ground_truth_answer": ground_truth_answer,
            "model_generation": answer_part,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct
        }
        
        results.append(result)
        
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} examples. Current accuracy: {correct_count / (idx + 1):.4f}")
    
    return results

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate evaluation metrics."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    
    metrics = {
        "total_examples": total,
        "correct_answers": correct,
        "accuracy": correct / total if total > 0 else 0.0
    }
    
    return metrics

def main():
    # Load model and tokenizer
    model, tokenizer, is_instruct = load_model_and_tokenizer(MODEL_NAME)
    
    # Load dataset
    dataset = load_gsm8k_dataset(SPLIT, MAX_SAMPLES)
    
    # Process examples
    start_time = time.time()
    results = process_gsm8k_examples(model, tokenizer, dataset, is_instruct)
    total_time = time.time() - start_time
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    metrics["total_time_seconds"] = total_time
    metrics["average_time_per_example"] = total_time / len(results) if results else 0
    
    # Add model info to metrics
    model_name_safe = MODEL_NAME.replace("/", "_")
    metrics["model_name"] = MODEL_NAME
    metrics["dataset_split"] = SPLIT
    metrics["sample_count"] = len(results)
    
    print("\n===== Final Results =====")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: GSM8K ({SPLIT})")
    print(f"Sample Count: {len(results)}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct_answers']} / {metrics['total_examples']})")
    print(f"Total Time: {metrics['total_time_seconds']:.2f} seconds")
    print(f"Average Time Per Example: {metrics['average_time_per_example']:.2f} seconds")
    
    # Save results and metrics
    output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{model_name_safe}_{SPLIT}_results.json")
    metrics_file = os.path.join(OUTPUT_DIR, f"gsm8k_{model_name_safe}_{SPLIT}_metrics.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()