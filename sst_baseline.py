import json
import time
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_predictions(csv_path="data/simple_safety_tests/simple_safety_tests.csv", 
                        model_id="Qwen/Qwen2.5-0.5B", 
                        output_file="safety_test_predictions.jsonl"):
    """Generate and save model predictions for SimpleSafetyTests dataset from CSV."""
    
    print(f"Loading model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Load the SimpleSafetyTests from CSV
    print(f"Loading SimpleSafetyTests from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Check if "instructions" column exists, otherwise use the first column
    if "instructions" not in df.columns:
        instruction_column = df.columns[0]
        print(f"No 'instructions' column found. Using '{instruction_column}' column instead.")
    else:
        instruction_column = "instructions"
    
    total_time = 0
    total_examples = 0
    
    with open(output_file, 'w') as f:
        for i, row in df.iterrows():
            prompt = row[instruction_column]
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate prediction and measure time
            start_time = time.time()
            with torch.no_grad():
                output_ids = model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    do_sample=False
                )
            end_time = time.time()
            
            # Calculate generation time
            generation_time = end_time - start_time
            total_time += generation_time
            total_examples += 1
            
            # Decode the output, removing the input prompt
            output_text = tokenizer.decode(
                output_ids[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Create result object and write to file
            result = {
                "prompts_final": prompt,
                "output": output_text
            }
            f.write(json.dumps(result) + "\n")
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(df)} examples. Last example took {generation_time:.2f} seconds.")
    
    # Calculate and print throughput statistics
    avg_time_per_example = total_time / total_examples
    throughput = total_examples / total_time
    
    print(f"\nProcessed {total_examples} examples in {total_time:.2f} seconds")
    print(f"Average time per example: {avg_time_per_example:.4f} seconds")
    print(f"Throughput: {throughput:.4f} examples/second")
    
    return {
        "total_examples": total_examples,
        "total_time": total_time,
        "avg_time_per_example": avg_time_per_example,
        "throughput": throughput
    }

def analyze_safety_results(results_file):
    """Analyze safety evaluation results."""
    safe_count = 0
    total_count = 0
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            if result.get("is_safe", False):
                safe_count += 1
            total_count += 1
    
    safety_proportion = safe_count / total_count if total_count > 0 else 0
    print(f"Proportion of safe outputs: {safety_proportion:.4f} ({safe_count}/{total_count})")
    return safety_proportion

def sample_unsafe_examples(results_file, n=10):
    """Sample n random unsafe examples from results."""
    import random
    
    unsafe_examples = []
    
    with open(results_file, 'r') as f:
        for line in f:
            result = json.loads(line)
            if not result.get("is_safe", True):
                unsafe_examples.append(result)
    
    # Sample n random examples or all if fewer than n
    sampled = random.sample(unsafe_examples, min(n, len(unsafe_examples)))
    
    print(f"Sampled {len(sampled)} unsafe examples out of {len(unsafe_examples)} total unsafe examples")
    return sampled

if __name__ == "__main__":
    # Generate predictions
    csv_path = "data/simple_safety_tests/simple_safety_tests.csv"
    output_file = "qwen25_05b_safety_predictions.jsonl"
    stats = generate_predictions(csv_path=csv_path, model_id="Qwen/Qwen2.5-0.5B", output_file=output_file)
    
    print("\nTo evaluate safety with Qwen2.5-3B-Instruct, run:")
    print(f"python scripts/evaluate_safety.py \\\n"
          f"  --input-path {output_file} \\\n"
          f"  --model-name-or-path /home/shared/Qwen2.5-3B-Instruct \\\n"
          f"  --num-gpus 2 \\\n"
          f"  --output-path safety_evaluation_results.jsonl")