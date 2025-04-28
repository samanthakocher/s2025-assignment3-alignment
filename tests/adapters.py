#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

# Imports for run_parse_mmlu_response
import re
from typing import Any

# Imports for get_packed_sft_dataset
import random



def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    # Simple minimal dataset definition
    class SimpleMapDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    if shuffle:
        random.shuffle(data)
    
    # Tokenize all examples
    all_token_ids = []
    
    for example in data:
        if "prompt" in example and "response" in example:
            instruction = example["prompt"]
            response = example["response"]
            
            # Format for instruction tuning
            formatted_text = f"{instruction}\n{response}"
            
            # Important fix: Append to all_token_ids instead of extend
            tokens = tokenizer.encode(formatted_text)
            all_token_ids.extend(tokens)
    
    # Pack the tokens into fixed-length sequences
    packed_data = []
    total_sequences = len(all_token_ids) // seq_length

    for i in range(total_sequences):
        start = i * seq_length
        end = start + seq_length
        sequence = all_token_ids[start:end]
        tensor = torch.tensor(sequence, dtype=torch.long)
        packed_data.append({"input_ids": tensor, "labels": tensor.clone()})
    
    packed_data = packed_data[:75]

    return SimpleMapDataset(packed_data)

def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    # Normalize the model output
    model_output = model_output.strip().upper()

    # Try to extract answer letter directly from common patterns
    match = re.search(r"\b([ABCD])\b", model_output)
    if match:
        return match.group(1)
    
    # If model output includes the full answer, try matching it
    options = mmlu_example["options"]
    for idx, option in enumerate(options):
        option_clean = re.escape(option.strip().upper())
        # Match as full word/phrase, not substring
        if re.search(rf"\b{option_clean}\b", model_output):
            return chr(ord("A") + idx)
        
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    # Find all numbers (integer or decimal)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", model_output)

    if not numbers:
        return None
    
    return numbers[-1]


def compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    # Format inputs using Alpaca template
    def format_alpaca(instruction, response=None):
        formatted = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
        if response:
            formatted += f" {response}<|endoftext|>"
        return formatted
    
    # Format the full sequences (prompt + response)
    prompt_only = format_alpaca(prompt)
    chosen_full = format_alpaca(prompt, response_chosen)
    rejected_full = format_alpaca(prompt, response_rejected)
    
    # Tokenize all inputs
    inputs_prompt = tokenizer(prompt_only, return_tensors="pt")
    inputs_chosen = tokenizer(chosen_full, return_tensors="pt")
    inputs_rejected = tokenizer(rejected_full, return_tensors="pt")
    
    # Move to the same device as the model
    device = next(lm.parameters()).device
    inputs_prompt = {k: v.to(device) for k, v in inputs_prompt.items()}
    inputs_chosen = {k: v.to(device) for k, v in inputs_chosen.items()}
    inputs_rejected = {k: v.to(device) for k, v in inputs_rejected.items()}
    
    # Get the length of the prompt to identify the start of the responses
    prompt_length = inputs_prompt['input_ids'].size(1)
    
    # Compute log probs for chosen response
    with torch.no_grad():
        # Policy model
        outputs_chosen_policy = lm(
            input_ids=inputs_chosen['input_ids'],
            attention_mask=inputs_chosen['attention_mask'],
            return_dict=True
        )
        logits_chosen_policy = outputs_chosen_policy.logits[:, :-1]  # Shift left for next token prediction
        
        # Reference model
        outputs_chosen_ref = lm_ref(
            input_ids=inputs_chosen['input_ids'],
            attention_mask=inputs_chosen['attention_mask'],
            return_dict=True
        )
        logits_chosen_ref = outputs_chosen_ref.logits[:, :-1]  # Shift left for next token prediction
        
        # Create log probabilities
        log_probs_chosen_policy = F.log_softmax(logits_chosen_policy, dim=-1)
        log_probs_chosen_ref = F.log_softmax(logits_chosen_ref, dim=-1)
        
        # Gather the log probs for the actual next tokens
        next_tokens_chosen = inputs_chosen['input_ids'][:, 1:]  # Shift right for actual next tokens
        gathered_log_probs_chosen_policy = torch.gather(
            log_probs_chosen_policy, 
            dim=-1, 
            index=next_tokens_chosen.unsqueeze(-1)
        ).squeeze(-1)
        gathered_log_probs_chosen_ref = torch.gather(
            log_probs_chosen_ref, 
            dim=-1, 
            index=next_tokens_chosen.unsqueeze(-1)
        ).squeeze(-1)
        
        # Create a mask that selects only the response tokens (after prompt)
        response_mask_chosen = torch.zeros_like(gathered_log_probs_chosen_policy)
        response_mask_chosen[:, prompt_length-1:] = 1  # -1 because of the shifted indices
        
        # Policy model logs for chosen
        chosen_response_log_probs_policy = (gathered_log_probs_chosen_policy * response_mask_chosen).sum() / response_mask_chosen.sum()
        
        # Reference model logs for chosen
        chosen_response_log_probs_ref = (gathered_log_probs_chosen_ref * response_mask_chosen).sum() / response_mask_chosen.sum()
        
    # Compute log probs for rejected response
    with torch.no_grad():
        # Policy model
        outputs_rejected_policy = lm(
            input_ids=inputs_rejected['input_ids'],
            attention_mask=inputs_rejected['attention_mask'],
            return_dict=True
        )
        logits_rejected_policy = outputs_rejected_policy.logits[:, :-1]  # Shift left for next token prediction
        
        # Reference model
        outputs_rejected_ref = lm_ref(
            input_ids=inputs_rejected['input_ids'],
            attention_mask=inputs_rejected['attention_mask'],
            return_dict=True
        )
        logits_rejected_ref = outputs_rejected_ref.logits[:, :-1]  # Shift left for next token prediction
        
        # Create log probabilities
        log_probs_rejected_policy = F.log_softmax(logits_rejected_policy, dim=-1)
        log_probs_rejected_ref = F.log_softmax(logits_rejected_ref, dim=-1)
        
        # Gather the log probs for the actual next tokens
        next_tokens_rejected = inputs_rejected['input_ids'][:, 1:]  # Shift right for actual next tokens
        gathered_log_probs_rejected_policy = torch.gather(
            log_probs_rejected_policy, 
            dim=-1, 
            index=next_tokens_rejected.unsqueeze(-1)
        ).squeeze(-1)
        gathered_log_probs_rejected_ref = torch.gather(
            log_probs_rejected_ref, 
            dim=-1, 
            index=next_tokens_rejected.unsqueeze(-1)
        ).squeeze(-1)
        
        # Create a mask that selects only the response tokens (after prompt)
        response_mask_rejected = torch.zeros_like(gathered_log_probs_rejected_policy)
        response_mask_rejected[:, prompt_length-1:] = 1  # -1 because of the shifted indices
        
        # Policy model logs for rejected
        rejected_response_log_probs_policy = (gathered_log_probs_rejected_policy * response_mask_rejected).sum() / response_mask_rejected.sum()
        
        # Reference model logs for rejected
        rejected_response_log_probs_ref = (gathered_log_probs_rejected_ref * response_mask_rejected).sum() / response_mask_rejected.sum()
    
    # Calculate log ratios
    chosen_log_ratio = chosen_response_log_probs_policy - chosen_response_log_probs_ref
    rejected_log_ratio = rejected_response_log_probs_policy - rejected_response_log_probs_ref
    
    # Compute the DPO loss: -log(sigmoid(β * (r_w - r_l)))
    # where r_w = log_ratio for chosen and r_l = log_ratio for rejected
    dpo_loss = -torch.log(torch.sigmoid(beta * (chosen_log_ratio - rejected_log_ratio)))
    
    return dpo_loss
