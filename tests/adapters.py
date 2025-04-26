#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# Imports for run_parse_mmlu_response
import re
from typing import Any

# Imports for get_packed_sft_dataset
import random
import math


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
    # Read and tokenize all examples into a flat list of token IDs
    all_token_ids = []

    # Check if the file is gzipped
    is_gzipped = str(dataset_path).endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'

    with open_func(dataset_path, mode) as f:
        examples = []
        for line in f:
            example = json.loads(line)
            
            # Check for the right keys (could be instruction/response or prompt/completion)
            instruction_key = "instruction" if "instruction" in example else "prompt"
            response_key = "response" if "response" in example else "completion"
            
            instruction = example[instruction_key]
            response = example[response_key]
            
            # Build: [BOS] + instruction_ids + [EOS] + response_ids + [EOS]
            bos_id = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            eos_id = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

            token_ids = bos_id + instruction_ids + eos_id + response_ids + eos_id
            examples.append(token_ids)
    
    # Shuffle examples if specified
    if shuffle:
        random.shuffle(examples)
    
    # Flatten all examples into a single list of token IDs
    for example in examples:
        all_token_ids.extend(example)
    
    # Create a custom dataset class
    class PackedDataset(Dataset):
        def __init__(self, token_ids, seq_length):
            self.token_ids = token_ids
            self.seq_length = seq_length
            
            # Calculate number of complete sequences
            self.num_sequences = math.ceil(len(self.token_ids) / self.seq_length)
        
        def __len__(self):
            return self.num_sequences
        
        def __getitem__(self, idx):
            if idx >= len(self):
                raise IndexError(f"Index {idx} out of bounds for dataset of length {len(self)}")
            
            start_idx = idx * self.seq_length
            end_idx = start_idx + self.seq_length
            
            sequence = self.token_ids[start_idx:end_idx]

            # If the sequence is too short, pad it
            if len(sequence) < self.seq_length:
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                sequence = sequence + [pad_token_id] * (self.seq_length - len(sequence))
            
            # For causal language modeling, input_ids and labels are the same
            return {
                "input_ids": torch.tensor(sequence, dtype=torch.long),
                "labels": torch.tensor(sequence, dtype=torch.long),
            }
    
    return PackedDataset(all_token_ids, seq_length)


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
    # Use DataLoader to handle batching
    from torch.utils.data import DataLoader
    
    # Create a DataLoader with the specified parameters
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False  # Include partial batches at the end
    )
    
    return data_loader


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
    raise NotImplementedError
