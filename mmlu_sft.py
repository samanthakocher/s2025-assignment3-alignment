import os
import json
import time
import random
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
MMLU_TASKS = ["high_school_mathematics", "high_school_physics", "high_school_chemistry", 
             "high_school_biology", "high_school_computer_science",
             "college_mathematics", "college_physics", "college_chemistry", 
             "college_biology", "college_computer_science"]
NUM_SAMPLES = None  # Set to None to evaluate all examples, or to a number for a subset

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def format_mmlu_example(example, instruction_format=True):
    """Format MMLU example according to instruction tuning format for Qwen models"""
    question = example["question"]
    choices = [example[f"choice{i}"] for i in range(4)]
    
    if instruction_format:
        # Qwen2.5-Instruct format
        prompt = f"""<|im_start|>user
Answer the following multiple-choice question. Choose the option that best answers the question.

Question: {question}
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}
<|im_end|>

<|im_start|>assistant
"""
    else:
        # Qwen2.5 base model format
        prompt = f"""
Question: {question}
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}
Answer:
"""
    
    return prompt.strip()

def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=5,  # We only need a short answer
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    end_time = time.time()
    
    generation_time = end_time - start_time
    
    output = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    # Clean any remaining special tokens from Qwen format
    output = output.replace("<|im_end|>", "").strip()
    
    # Extract the predicted answer (A, B, C, or D)
    for option in ["A", "B", "C", "D"]:
        if output.startswith(option) or output.startswith("The answer is " + option) or output == option:
            return option, generation_time
    
    # If no clear option found, return the first character as fallback
    if output:
        return output[0], generation_time
    else:
        return "X", generation_time  # Indicates no answer

def evaluate_model(model, tokenizer, instruction_format=True):
    correct_count = 0
    total_count = 0
    total_time = 0
    incorrect_examples = []
    
    for task in MMLU_TASKS:
        dataset = load_dataset("cais/mmlu", task, split="test")
        
        if NUM_SAMPLES is not None and NUM_SAMPLES < len(dataset):
            # Create a fixed random subset for reproducibility
            np.random.seed(42)
            indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
            dataset = dataset.select(indices)
        
        for example in tqdm(dataset, desc=f"Evaluating {task}"):
            prompt = format_mmlu_example(example, instruction_format)
            
            prediction, gen_time = get_model_prediction(model, tokenizer, prompt)
            total_time += gen_time
            
            # Convert answer index to letter
            correct_answer = "ABCD"[example["answer"]]
            
            if prediction == correct_answer:
                correct_count += 1
            else:
                if instruction_format:  # Only collect incorrect examples for fine-tuned model
                    incorrect_examples.append({
                        "task": task,
                        "question": example["question"],
                        "choices": [example[f"choice{i}"] for i in range(4)],
                        "correct_answer": correct_answer,
                        "predicted_answer": prediction,
                        "model_output": prompt + " " + prediction
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
        print(f"Task: {example['task']}")
        print(f"Question: {example['question']}")
        print(f"Choices:")
        for j, choice in enumerate(example['choices']):
            print(f"  {chr(65+j)}. {choice}")
        print(f"Correct answer: {example['correct_answer']}")
        print(f"Predicted answer: {example['predicted_answer']}")
    
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
        "error_examples": sampled_errors
    }
    
    with open("mmlu_qwen_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()