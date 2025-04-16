import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_mmlu_dataset():
    """Load MMLU dataset from HuggingFace datasets."""
    print("Loading MMLU dataset...")
    mmlu = load_dataset("cais/mmlu", "all")
    return mmlu["test"], mmlu["validation"]

def format_prompt(example):
    """Format a single MMLU example as a prompt for zero-shot evaluation."""
    prompt = f"Question: {example['question']}\n"
    prompt += f"A: {example['choices'][0]}\n"
    prompt += f"B: {example['choices'][1]}\n"
    prompt += f"C: {example['choices'][2]}\n"
    prompt += f"D: {example['choices'][3]}\n"
    prompt += "Answer: "
    return prompt

def load_model_and_tokenizer():
    """Load Qwen2.5-0.5B model and tokenizer."""
    print("Loading Qwen2.5-0.5B model and tokenizer...")
    model_name = "Qwen/Qwen2.5-0.5B"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    return model, tokenizer

def generate_prediction(model, tokenizer, prompt, max_new_tokens=10):
    """Generate prediction for a single example."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,  # For deterministic outputs
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def get_answer_from_generation(generation):
    """Extract the answer (A, B, C, or D) from the model's generation."""
    # First, check if the first character is a valid answer
    if generation and generation[0] in "ABCD":
        return generation[0]
    
    # Check for answer patterns in the full generation
    for answer in ["A", "B", "C", "D"]:
        if answer in generation:
            return answer
    
    # If no pattern matches, return empty string
    return ""

def evaluate_responses(examples, predictions, answers):
    """Calculate accuracy metrics for the model's predictions."""
    correct = 0
    total = len(predictions)
    
    results = []
    
    for i, (example, pred, ans) in enumerate(zip(examples, predictions, answers)):
        is_correct = pred == ans
        if is_correct:
            correct += 1
        
        results.append({
            "question": example["question"],
            "choices": example["choices"],
            "subject": example["subject"],
            "predicted_answer": pred,
            "correct_answer": ans,
            "is_correct": is_correct
        })
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, results

def save_results(results, accuracy, output_dir="mmlu_results"):
    """Save evaluation results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results as JSON
    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Calculate per-subject accuracy
    df = pd.DataFrame(results)
    subject_metrics = df.groupby("subject")["is_correct"].agg(["mean", "count"]).reset_index()
    subject_metrics.columns = ["Subject", "Accuracy", "Count"]
    subject_metrics = subject_metrics.sort_values("Accuracy", ascending=False)
    
    # Save per-subject metrics as CSV
    subject_metrics.to_csv(os.path.join(output_dir, "subject_metrics.csv"), index=False)
    
    # Save overall results
    overall_results = {
        "overall_accuracy": accuracy,
        "total_examples": len(results),
        "subject_count": len(subject_metrics)
    }
    
    with open(os.path.join(output_dir, "overall_results.json"), "w") as f:
        json.dump(overall_results, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    return subject_metrics

def main():
    """Main function to run the MMLU evaluation."""
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load MMLU dataset
    test_dataset, val_dataset = load_mmlu_dataset()
    
    # Use validation set instead of full test set for faster evaluation (optional)
    dataset_to_evaluate = val_dataset
    print(f"Using dataset with {len(dataset_to_evaluate)} examples")
    
    # Process examples and generate predictions
    predictions = []
    formatted_prompts = []
    
    print("Generating predictions...")
    for example in tqdm(dataset_to_evaluate):
        prompt = format_prompt(example)
        formatted_prompts.append(prompt)
        
        # Generate prediction
        generation = generate_prediction(model, tokenizer, prompt)
        answer = get_answer_from_generation(generation)
        predictions.append(answer)
    
    # Get correct answers
    correct_answers = [example["answer"] for example in dataset_to_evaluate]
    
    # Evaluate predictions
    accuracy, results = evaluate_responses(dataset_to_evaluate, predictions, correct_answers)
    
    print(f"Overall accuracy: {accuracy:.4f}")
    
    # Save results
    subject_metrics = save_results(results, accuracy)
    
    # Print top and bottom 5 subjects by accuracy
    print("\nTop 5 subjects by accuracy:")
    print(subject_metrics.head(5)[["Subject", "Accuracy", "Count"]])
    
    print("\nBottom 5 subjects by accuracy:")
    print(subject_metrics.tail(5)[["Subject", "Accuracy", "Count"]])

if __name__ == "__main__":
    main()