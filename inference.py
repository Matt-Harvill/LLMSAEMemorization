from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import os
from tqdm import tqdm

CHUNK_LENGTH = 500  # Length of each input chunk in tokens
CHUNK_OFFSET = 10   # Number of tokens to generate/compare
BATCH_SIZE = 16
DEBUG = False
DEBUG_TRUNCATE_TOKENS = 5000

def get_output_filename() -> str:
    """Generate output filename based on chunk parameters"""
    return f"memorization_results_L{CHUNK_LENGTH}_O{CHUNK_OFFSET}.csv"

def setup_model_and_tokenizer(model_name: str = "meta-llama/Meta-Llama-3-8B"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16, 
        device_map=device
    )
    return model, tokenizer, device

def create_chunks(tokenized_text: List[int], tokenizer, chunk_length: int = CHUNK_LENGTH) -> List[Tuple[List[int], List[int]]]:
    """Create chunks of (input_tokens, true_continuation_tokens)"""
    chunks = []
    for i in range(0, len(tokenized_text) - chunk_length - CHUNK_OFFSET, CHUNK_OFFSET):
        input_tokens = tokenized_text[i:i + chunk_length]
        true_continuation = tokenized_text[i + chunk_length:i + chunk_length + CHUNK_OFFSET]
        # Only add if we have a full continuation
        if len(true_continuation) == CHUNK_OFFSET and len(input_tokens) == chunk_length:
            chunks.append((input_tokens, true_continuation))
    return chunks

def process_batch(
    model, 
    tokenizer, 
    batch_chunks: List[Tuple[List[int], List[int]]], 
    device: str
) -> List[Dict]:
    """Process a batch of chunks and return results"""
    # Prepare input tokens and convert to tensor
    batch_input_tokens = [chunk[0] for chunk in batch_chunks]
    batch_true_continuations = [chunk[1] for chunk in batch_chunks]
    
    # Create input tensor directly from tokens
    input_tensor = torch.tensor(batch_input_tokens, device=device)
    
    # Generate continuations
    outputs = model.generate(
        input_ids=input_tensor,
        max_new_tokens=CHUNK_OFFSET,
        pad_token_id=tokenizer.eos_token_id,  # Use eos_token_id as pad_token_id
    )
    
    results = []
    for i in range(len(batch_chunks)):
        input_tokens = batch_input_tokens[i]
        true_continuation_tokens = batch_true_continuations[i]
        
        # Get generated continuation (excluding prompt)
        generated_tokens = outputs[i, len(input_tokens):len(input_tokens) + CHUNK_OFFSET].tolist()
        
        # Convert to tensors for comparison
        gen_tensor = torch.tensor(generated_tokens, device=device)
        true_tensor = torch.tensor(true_continuation_tokens, device=device)
        
        # Compare tokens
        matches = (gen_tensor == true_tensor)
        matching_tokens = matches.sum().item()
        perfect_match = matching_tokens == CHUNK_OFFSET
        
        # Decode for text representation (only for display/storage)
        input_text = tokenizer.decode(input_tokens, skip_special_tokens=True)
        
        # Decode each token individually for colored output
        if DEBUG:
            # Print input in blue
            print("\033[94m" + input_text + "\033[0m", end="")
            
            # Print each continuation token with its own color
            for j in range(CHUNK_OFFSET):
                token_text = tokenizer.decode([generated_tokens[j]], skip_special_tokens=True)
                # Green for match, red for mismatch
                color = "\033[92m" if matches[j] else "\033[91m"
                print(color + token_text + "\033[0m", end="")
            print()  # New line after continuation
            
            print(f"Matching tokens: {matching_tokens}/{CHUNK_OFFSET}")
            print(f"Perfect match: {perfect_match}")
            print("\n")
        
        # Get full text versions for storage
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        true_continuation_text = tokenizer.decode(true_continuation_tokens, skip_special_tokens=True)
        
        results.append({
            "input_text": input_text,
            "input_tokens": input_tokens,
            "generated_continuation_text": generated_text,
            "generated_continuation_tokens": generated_tokens,
            "true_continuation_text": true_continuation_text,
            "true_continuation_tokens": true_continuation_tokens,
            "matching_tokens": matching_tokens,
            "perfect_match": perfect_match
        })
    
    return results

def main():
    # Check if output file already exists
    output_file = get_output_filename()
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping experiment.")
        return
        
    # Setup
    model, tokenizer, device = setup_model_and_tokenizer()
    
    # Load and tokenize input text
    with open("alice_in_wonderland.txt", "r") as f:
        text = f.read()
    tokenized_text = tokenizer.encode(text, add_special_tokens=False)
    
    # For testing, limit tokens
    if DEBUG:
        tokenized_text = tokenized_text[:DEBUG_TRUNCATE_TOKENS]
    print(f"Total tokens: {len(tokenized_text)}")
    
    # Create chunks
    chunks = create_chunks(tokenized_text, tokenizer)
    print(f"Number of chunks: {len(chunks)}")
    
    # Initialize results DataFrame
    results_df = pd.DataFrame(columns=[
        "input_text",
        "input_tokens",
        "generated_continuation_text",
        "generated_continuation_tokens",
        "true_continuation_text",
        "true_continuation_tokens",
        "matching_tokens",
        "perfect_match"
    ])

    # Process chunks in batches with progress bar
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for i in range(0, len(chunks), BATCH_SIZE):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            batch_results = process_batch(model, tokenizer, batch_chunks, device)
            
            # Add results to DataFrame
            for result in batch_results:
                results_df.loc[len(results_df)] = result
            
            pbar.update(1)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"Perfect matches: {results_df['perfect_match'].sum()}/{len(results_df)}")

if __name__ == "__main__":
    main()