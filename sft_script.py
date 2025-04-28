import os
import logging
import math
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class InstructionDataset(Dataset):
    """Dataset for instruction tuning data"""
    
    def __init__(self, tokenizer, dataset, max_length=512):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        instruction = item["instruction"]
        input_text = item.get("input", "")
        response = item["output"]
        
        # Format the input based on whether there's additional input or just an instruction
        if input_text:
            prompt = f"<|im_start|>user\n{instruction}\n\n{input_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
        
        completion = f"{response}<|im_end|>"
        full_text = prompt + completion
        
        # Tokenize the full text
        encodings = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"][0]
        attention_mask = encodings["attention_mask"][0]
        
        # Create labels: -100 for prompt tokens (we don't want to compute loss on them)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        prompt_length = len(prompt_ids)
        
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # Mask out the prompt tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch):
    """Collate function to create batches with padding"""
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # Set random seeds for reproducibility
    set_seed(42)
    
    # Define training parameters
    model_name = "Qwen/Qwen2.5-0.5B"  # Using Qwen2.5-0.5B as specified
    dataset_name = "tatsu-lab/alpaca"  # Standard instruction dataset
    output_dir = "./qwen-finetuned-t4"
    
    # Training hyperparameters (adjusted for T4 GPU)
    per_device_batch_size = 1  # Using batch size of 1 per device as requested
    gradient_accumulation_steps = 32  # Accumulate to get effective batch size of 32
    learning_rate = 2e-5
    max_seq_length = 512
    max_train_time_minutes = 30  # Train for about 30 minutes
    warmup_ratio = 0.03
    logging_steps = 25  # More frequent logging to track progress
    eval_every_n_steps = 100  # More frequent evaluation
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not defined
    
    # Load the model
    logger.info(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Using float32 as specified
    )
    
    # Get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    # Create train and eval datasets
    # Take a smaller subset for faster training
    if len(dataset["train"]) > 10000:
        train_data = dataset["train"].select(range(10000))
    else:
        train_data = dataset["train"]
    
    train_dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset=train_data,
        max_length=max_seq_length
    )
    
    # Create a small validation set
    if "validation" in dataset:
        eval_data = dataset["validation"]
        if len(eval_data) > 1000:
            eval_data = eval_data.select(range(1000))
    else:
        # If no validation split, use a portion of the train data
        train_size = int(0.9 * len(train_dataset))
        eval_size = min(len(train_dataset) - train_size, 1000)  # Cap at 1000 examples
        train_dataset, eval_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, eval_size]
        )
        eval_data = train_data.select([train_dataset.indices[i] for i in range(eval_size)])
    
    eval_dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset=eval_data,
        max_length=max_seq_length
    )
    
    # Create data loaders
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_batch_size,
        collate_fn=collate_fn,
    )
    
    # Define optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Estimate total steps based on time limit
    # We'll adjust during training based on actual step time
    estimated_steps_per_second = 0.1  # Initial guess for T4 GPU
    max_steps = int((max_train_time_minutes * 60) * estimated_steps_per_second)
    
    # Create learning rate scheduler with cosine decay
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(max_steps * warmup_ratio),
        num_training_steps=max_steps,
    )
    
    # Initial evaluation
    logger.info("Running initial evaluation")
    model.eval()
    initial_eval_loss = 0.0
    initial_eval_steps = 0
    
    for eval_batch in tqdm(eval_dataloader, desc="Initial evaluation"):
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        
        with torch.no_grad():
            eval_outputs = model(
                input_ids=eval_batch["input_ids"],
                attention_mask=eval_batch["attention_mask"],
                labels=eval_batch["labels"],
            )
        
        initial_eval_loss += eval_outputs.loss.item()
        initial_eval_steps += 1
        
        # Limit initial evaluation time
        if initial_eval_steps >= 100:
            break
    
    initial_eval_loss = initial_eval_loss / initial_eval_steps
    logger.info(f"Initial evaluation loss: {initial_eval_loss:.4f}")
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Batch size per device = {per_device_batch_size}")
    logger.info(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Target training time = {max_train_time_minutes} minutes")
    
    model.train()
    start_time = time.time()
    global_step = 0
    steps_in_epoch = 0
    epoch = 0
    
    # Lists to track metrics for learning curve
    train_losses = []
    train_step_times = []
    eval_losses = []
    eval_steps = []
    
    # Training loop with time limit
    train_iterator = iter(train_dataloader)
    continue_training = True
    
    while continue_training:
        try:
            batch = next(train_iterator)
        except StopIteration:
            # Restart iterator for next epoch
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)
            epoch += 1
            steps_in_epoch = 0
            logger.info(f"Starting epoch {epoch+1}")
        
        steps_in_epoch += 1
        
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        step_start_time = time.time()
        
        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        
        # Scale loss by gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        
        # Record training loss
        train_losses.append(loss.item())
        
        # Backward pass with gradient accumulation
        loss.backward()
        
        if ((steps_in_epoch % gradient_accumulation_steps == 0) or 
            (steps_in_epoch == len(train_dataloader))):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Record step time for tracking
            step_time = time.time() - step_start_time
            train_step_times.append(step_time)
            
            # Update estimated steps per second
            if len(train_step_times) > 10:
                avg_step_time = sum(train_step_times[-10:]) / 10
                estimated_steps_per_second = 1.0 / avg_step_time
            
            # Log training metrics
            if global_step % logging_steps == 0:
                elapsed_time = (time.time() - start_time) / 60.0  # minutes
                logger.info(
                    f"Step {global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"LR: {lr_scheduler.get_last_lr()[0]:.8f} | "
                    f"Elapsed: {elapsed_time:.2f} min"
                )
            
            # Evaluate model
            if global_step % eval_every_n_steps == 0:
                model.eval()
                eval_loss = 0.0
                eval_step_count = 0
                
                logger.info("Running evaluation")
                for eval_batch in eval_dataloader:
                    eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
                    
                    with torch.no_grad():
                        eval_outputs = model(
                            input_ids=eval_batch["input_ids"],
                            attention_mask=eval_batch["attention_mask"],
                            labels=eval_batch["labels"],
                        )
                    
                    eval_loss += eval_outputs.loss.item()
                    eval_step_count += 1
                    
                    # Limit evaluation time
                    if eval_step_count >= 100:
                        break
                
                avg_eval_loss = eval_loss / eval_step_count
                eval_losses.append(avg_eval_loss)
                eval_steps.append(global_step)
                
                logger.info(f"Evaluation results - Loss: {avg_eval_loss:.4f}, Perplexity: {math.exp(avg_eval_loss):.4f}")
                model.train()
                
                # Save a checkpoint
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Check if we've reached the time limit
        elapsed_minutes = (time.time() - start_time) / 60.0
        if elapsed_minutes >= max_train_time_minutes:
            logger.info(f"Reached time limit of {max_train_time_minutes} minutes. Stopping training.")
            continue_training = False
    
    # Final evaluation
    model.eval()
    final_eval_loss = 0.0
    final_eval_steps = 0
    
    logger.info("Running final evaluation")
    for eval_batch in tqdm(eval_dataloader, desc="Final evaluation"):
        eval_batch = {k: v.to(device) for k, v in eval_batch.items()}
        
        with torch.no_grad():
            eval_outputs = model(
                input_ids=eval_batch["input_ids"],
                attention_mask=eval_batch["attention_mask"],
                labels=eval_batch["labels"],
            )
        
        final_eval_loss += eval_outputs.loss.item()
        final_eval_steps += 1
        
        # Limit final evaluation time
        if final_eval_steps >= 100:
            break
    
    final_eval_loss = final_eval_loss / final_eval_steps
    eval_losses.append(final_eval_loss)
    eval_steps.append(global_step)
    
    total_training_time = (time.time() - start_time) / 60.0
    logger.info(f"Total training time: {total_training_time:.2f} minutes")
    logger.info(f"Final evaluation loss: {final_eval_loss:.4f}, Perplexity: {math.exp(final_eval_loss):.4f}")
    
    # Save the final model and tokenizer
    logger.info("***** Training completed *****")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Saved model and tokenizer to {output_dir}")
    
    # Generate and save the learning curve
    plt.figure(figsize=(10, 6))
    
    # Plot training loss (smoothed)
    def moving_average(data, window_size=10):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    if len(train_losses) > 10:
        smoothed_train_losses = moving_average(train_losses)
        plt.plot(range(len(smoothed_train_losses)), smoothed_train_losses, label='Training Loss (Smoothed)')
    else:
        plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
        
    plt.plot(eval_steps, eval_losses, 'ro-', label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'))
    plt.close()
    
    # Save the loss values for future reference
    np.save(os.path.join(output_dir, 'train_losses.npy'), np.array(train_losses))
    np.save(os.path.join(output_dir, 'eval_losses.npy'), np.array(eval_losses))
    np.save(os.path.join(output_dir, 'eval_steps.npy'), np.array(eval_steps))
    
    # Log final statistics
    logger.info(f"Final training statistics:")
    logger.info(f"  Initial validation loss: {initial_eval_loss:.4f}")
    logger.info(f"  Final validation loss: {final_eval_loss:.4f}")
    logger.info(f"  Validation loss improvement: {initial_eval_loss - final_eval_loss:.4f}")
    logger.info(f"  Final validation perplexity: {math.exp(final_eval_loss):.4f}")
    logger.info(f"  Total training steps: {global_step}")

if __name__ == "__main__":
    main()