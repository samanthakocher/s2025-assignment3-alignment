import os
import json
import time
import random
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"  # Instruction-tuned model
ZERO_SHOT_MODEL_ID = "Qwen/Qwen2.5-0.5B"  # Base model
BATCH_SIZE = 1  # Adjust based on your GPU memory
NUM_SAMPLES = 100  # Number of examples to evaluate (set to None for full dataset)

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def format_gsm8k_example(example, instruction_format=True):
    """Format GSM8K example according to instruction tuning format for Qwen models"""
    question = example["question"]
    
    if instruction_format:
        # Qwen2.5-Instruct format
        prompt = f"""<|im_start|>user
Solve the following math problem step by step:

{question}<|im_end|>

<|im_start|>assistant
"""
    else:
        # Qwen2.5 base model format
        prompt = f"""
{question}

Let's solve this step by step:
"""
    
    return prompt.strip()

def extract_answer(output):
    """Extract the final numerical answer from the model's output"""
    # Look for the final answer pattern, typically after "Therefore" or "The answer is" or at the end
    patterns = [
        r"Therefore,?\s+the\s+answer\s+is\s+([\$\-\d\,\.]+)",
        r"Thus,?\s+the\s+answer\s+is\s+([\$\-\d\,\.]+)",
        r"So,?\s+the\s+answer\s+is\s+([\$\-\d\,\.]+)",
        r"The\s+answer\s+is\s+([\$\-\d\,\.]+)",
        r"The\s+final\s+answer\s+is\s+([\$\-\d\,\.]+)",
        r"The\s+result\s+is\s+([\$\-\d\,\.]+)",
        r"=\s*([\$\-\d\,\.]+)(?:\s|$)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            # Get the last match as it's likely the final answer
            answer_str = matches[-1]
            # Remove $ and commas for numerical comparison
            answer_str = answer_str.replace("$", "").replace(",", "")
            try:
                return float(answer_str)
            except ValueError:
                continue
    
    # As a fallback, look for the last number in the text
    numbers = re.findall(r"([\-\d\,\.]+)", output)
    if numbers:
        try:
            # Clean and convert the last number found
            last_number = numbers[-1].replace(",", "")
            return float(last_number)
        except ValueError:
            pass
    
    return None

def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=300,  # Need longer outputs for step-by-step solutions
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    generation_time = end_time - start_time
    
    output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Clean any remaining special tokens from Qwen format
    output = output.replace("<|im_end|>", "").strip()
    
    # Extract the numerical answer
    predicted_answer = extract_answer(output)
    
    return predicted_answer, output, generation_time

def evaluate_model(model, tokenizer, instruction_format=True):
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if NUM_SAMPLES is not None and NUM_SAMPLES < len(dataset):
        # Create a fixed random subset for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
        dataset = dataset.select(indices)
    
    correct_count = 0
    total_count = 0
    total_time = 0
    incorrect_examples = []
    
    for example in tqdm(dataset, desc="Evaluating GSM8K"):
        prompt = format_gsm8k_example(example, instruction_format)
        
        # Get ground truth answer
        ground_truth = float(example["answer"].split("####")[1].strip())
        
        # Get model prediction
        prediction, full_output, gen_time = get_model_prediction(model, tokenizer, prompt)
        total_time += gen_time
        
        # Check if prediction is correct (allowing for small floating point differences)
        is_correct = False
        if prediction is not None:
            is_correct = abs(prediction - ground_truth) < 1e-6
        
        if is_correct:
            correct_count += 1
        else:
            incorrect_examples.append({
                "question": example["question"],
                "correct_answer": ground_truth,
                "predicted_answer": prediction,
                "full_output": full_output,
                "full_prompt": prompt
            })
        
        total_count += 1
    
    accuracy = correct_count / total_count
    throughput = total_count / total_time
    
    return {
        "accuracy": accuracy,
        "total_examples": total_count,
        "correct_count": correct_count,
        "total_time": total_time,
        "throughput": throughput,
        "incorrect_examples": incorrect_examples
    }

def main():
    # Load and evaluate instruction-tuned model
    print("Loading Qwen2.5-3B-Instruct model...")
    ft_model, ft_tokenizer = load_model_and_tokenizer(MODEL_ID)
    
    print("Evaluating instruction-tuned model...")
    ft_results = evaluate_model(ft_model, ft_tokenizer, instruction_format=True)
    
    # Load and evaluate zero-shot baseline model
    print("Loading Qwen2.5-0.5B model...")
    zs_model, zs_tokenizer = load_model_and_tokenizer(ZERO_SHOT_MODEL_ID)
    
    print("Evaluating zero-shot baseline model...")
    zs_results = evaluate_model(zs_model, zs_tokenizer, instruction_format=False)
    
    # Print results
    print("\n===== Results =====")
    print(f"Qwen2.5-3B-Instruct accuracy: {ft_results['accuracy']:.4f}")
    print(f"Qwen2.5-0.5B accuracy: {zs_results['accuracy']:.4f}")
    print(f"Qwen2.5-3B-Instruct throughput: {ft_results['throughput']:.2f} examples/second")
    print(f"Qwen2.5-0.5B throughput: {zs_results['throughput']:.2f} examples/second")
    
    # Sample 10 random incorrect examples
    if len(ft_results['incorrect_examples']) > 10:
        sampled_errors = random.sample(ft_results['incorrect_examples'], 10)
    else:
        sampled_errors = ft_results['incorrect_examples']
    
    print("\n===== Error Analysis =====")
    for i, example in enumerate(sampled_errors):
        print(f"\nError Example {i+1}:")
        print(f"Question: {example['question']}")
        print(f"Correct answer: {example['correct_answer']}")
        print(f"Predicted answer: {example['predicted_answer']}")
        print("Model Output:")
        print(example['full_output'][:300] + "..." if len(example['full_output']) > 300 else example['full_output'])
    
    # Compare to zero-shot errors for the same questions
    print("\n===== Zero-Shot Baseline Outputs for the Same Questions =====")
    for i, example in enumerate(sampled_errors):
        # Find the corresponding question in zero-shot results
        zs_example = next((ex for ex in zs_results['incorrect_examples'] 
                           if ex['question'] == example['question']), None)
        
        if zs_example:
            print(f"\nZero-Shot Error Example {i+1}:")
            print(f"Question: {zs_example['question']}")
            print(f"Correct answer: {zs_example['correct_answer']}")
            print(f"Predicted answer: {zs_example['predicted_answer']}")
            print("Model Output:")
            print(zs_example['full_output'][:300] + "..." if len(zs_example['full_output']) > 300 else zs_example['full_output'])
    
    # Save detailed results to file
    results = {
        "instruction_tuned_model": {
            "model": "Qwen2.5-3B-Instruct",
            "accuracy": ft_results['accuracy'],
            "throughput": ft_results['throughput'],
            "total_time": ft_results['total_time'],
            "total_examples": ft_results['total_examples']
        },
        "zero_shot_baseline": {
            "model": "Qwen2.5-0.5B",
            "accuracy": zs_results['accuracy'],
            "throughput": zs_results['throughput'],
            "total_time": zs_results['total_time'],
            "total_examples": zs_results['total_examples']
        },
        "error_examples": [{
            "question": example['question'],
            "correct_answer": example['correct_answer'],
            "instruction_tuned": {
                "prediction": example['predicted_answer'],
                "output": example['full_output']
            },
            "zero_shot": {
                "prediction": next((ex['predicted_answer'] for ex in zs_results['incorrect_examples'] 
                                   if ex['question'] == example['question']), None),
                "output": next((ex['full_output'] for ex in zs_results['incorrect_examples'] 
                               if ex['question'] == example['question']), None)
            }
        } for example in sampled_errors]
    }
    
    with open("gsm8k_qwen_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()