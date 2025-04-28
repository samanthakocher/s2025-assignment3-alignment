import json
import gzip
import random
from typing import Dict, List, Tuple, Any

def load_anthropic_hh_dataset(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Load and process the Anthropic HH dataset from multiple files.
    
    Args:
        file_paths: List of paths to the gzipped JSON files
        
    Returns:
        A list of dictionaries, each containing:
        - 'instruction': The first human message
        - 'chosen': The chosen assistant response
        - 'rejected': The rejected assistant response
        - 'source_file': The filename the example came from
    """
    all_examples = []
    
    for file_path in file_paths:
        # Extract the filename for source tracking
        source_file = file_path.split('/')[-1]
        
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                example = json.loads(line)
                
                # Extract the chosen and rejected conversations
                chosen_convo = example.get('chosen', None)
                rejected_convo = example.get('rejected', None)
                
                if not chosen_convo or not rejected_convo:
                    continue
                
                # Parse conversations into turns
                chosen_turns = parse_conversation(chosen_convo)
                rejected_turns = parse_conversation(rejected_convo)
                
                # Skip multi-turn conversations (where human sent more than one message)
                human_msgs_chosen = [turn for turn in chosen_turns if turn['role'] == 'human']
                human_msgs_rejected = [turn for turn in rejected_turns if turn['role'] == 'human']
                
                if len(human_msgs_chosen) != 1 or len(human_msgs_rejected) != 1:
                    continue
                
                # Verify that both conversations start with the same human prompt
                if human_msgs_chosen[0]['content'] != human_msgs_rejected[0]['content']:
                    continue
                
                # Get the first assistant responses
                assistant_msgs_chosen = [turn for turn in chosen_turns if turn['role'] == 'assistant']
                assistant_msgs_rejected = [turn for turn in rejected_turns if turn['role'] == 'assistant']
                
                if not assistant_msgs_chosen or not assistant_msgs_rejected:
                    continue
                
                # Create the processed example
                processed_example = {
                    'instruction': human_msgs_chosen[0]['content'],
                    'chosen': assistant_msgs_chosen[0]['content'],
                    'rejected': assistant_msgs_rejected[0]['content'],
                    'source_file': source_file
                }
                
                all_examples.append(processed_example)
    
    return all_examples

def parse_conversation(conversation: str) -> List[Dict[str, str]]:
    """
    Parse a conversation string into a list of message dictionaries.
    
    Args:
        conversation: String representation of the conversation
        
    Returns:
        List of dictionaries with 'role' and 'content' keys
    """
    # Split by Human: and Assistant:
    turns = []
    lines = conversation.split('\n')
    current_role = None
    current_content = []
    
    for line in lines:
        if line.startswith("Human:"):
            # Save previous turn if exists
            if current_role:
                turns.append({
                    'role': current_role,
                    'content': '\n'.join(current_content).strip()
                })
            # Start new human turn
            current_role = 'human'
            current_content = [line[len("Human:"):].strip()]
        elif line.startswith("Assistant:"):
            # Save previous turn if exists
            if current_role:
                turns.append({
                    'role': current_role,
                    'content': '\n'.join(current_content).strip()
                })
            # Start new assistant turn
            current_role = 'assistant'
            current_content = [line[len("Assistant:"):].strip()]
        else:
            # Continue current turn
            if current_role:
                current_content.append(line)
    
    # Add the last turn
    if current_role:
        turns.append({
            'role': current_role,
            'content': '\n'.join(current_content).strip()
        })
    
    return turns

def get_random_examples_by_category(dataset: List[Dict[str, Any]], category: str, count: int = 3) -> List[Dict[str, Any]]:
    """
    Get random examples from a specific category file.
    
    Args:
        dataset: The loaded dataset
        category: The category to filter on (e.g., 'helpful' or 'harmless')
        count: Number of examples to return
    
    Returns:
        List of random examples from the specified category
    """
    category_examples = [ex for ex in dataset if category in ex['source_file']]
    if len(category_examples) < count:
        return category_examples
    return random.sample(category_examples, count)

# Example usage:
if __name__ == "__main__":
    # Replace these with actual file paths
    file_paths = [
        "harmless-base.jsonl.gz",
        "helpful-base.jsonl.gz",
        "helpful-online.jsonl.gz",
        "helpful-rejection-sampled.jsonl.gz"
    ]
    
    # Load the dataset
    dataset = load_anthropic_hh_dataset(file_paths)
    print(f"Loaded {len(dataset)} examples")
    
    # Get random examples from each category
    helpful_examples = get_random_examples_by_category(dataset, "helpful", 3)
    harmless_examples = get_random_examples_by_category(dataset, "harmless", 3)
    
    # Print examples for analysis
    print("\n=== HELPFUL EXAMPLES ===")
    for i, example in enumerate(helpful_examples):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction'][:100]}...")
        print(f"Chosen: {example['chosen'][:100]}...")
        print(f"Rejected: {example['rejected'][:100]}...")
    
    print("\n=== HARMLESS EXAMPLES ===")
    for i, example in enumerate(harmless_examples):
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction'][:100]}...")
        print(f"Chosen: {example['chosen'][:100]}...")
        print(f"Rejected: {example['rejected'][:100]}...")