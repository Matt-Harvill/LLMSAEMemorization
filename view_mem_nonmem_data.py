import pandas as pd
import sys
import os
from enum import Enum
from transformers import AutoTokenizer

class MatchFilter(Enum):
    DEFAULT = "default"        # Show all matches
    PERFECT_MATCH = "perfect"  # Show only perfect matches
    NO_MATCH = "no_match"      # Show only matches with matching_tokens=0

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def color_code_differences(generated, ground_truth):
    """Color code the differences between generated and ground truth text"""
    result = ""
    min_len = min(len(generated), len(ground_truth))
    
    # Compare common length
    for i in range(min_len):
        if generated[i] == ground_truth[i]:
            # Matching character in green
            result += f"\033[92m{generated[i]}\033[0m"
        else:
            # Different character in red
            result += f"\033[91m{generated[i]}\033[0m"
    
    # Handle any remaining characters in generated text (in red)
    if len(generated) > min_len:
        result += f"\033[91m{generated[min_len:]}\033[0m"
            
    return result

def color_code_token_differences(generated_tokens, ground_truth_tokens, tokenizer):
    """Color code the differences between generated and ground truth tokens"""
    result = ""
    min_len = min(len(generated_tokens), len(ground_truth_tokens))
    
    # Compare common length
    for i in range(min_len):
        if generated_tokens[i] == ground_truth_tokens[i]:
            # Matching token in green
            result += f"\033[92m[{tokenizer.decode(generated_tokens[i])}]\033[0m"
        else:
            # Different token in red
            result += f"\033[91m[{tokenizer.decode(generated_tokens[i])}]\033[0m"
    
    # Handle any remaining tokens in generated text (in red)
    if len(generated_tokens) > min_len:
        result += f"\033[91m{generated_tokens[min_len:]}\033[0m"
            
    return result

def display_match(row, index, total, tokenizer):
    """Display a match with both character and token-level differences"""
    clear_screen()
    print(f"Match {index + 1}/{total}")
    print("-" * 80)
    
    # Character-level view
    print("Character-level view:")
    print("-" * 40)
    # Input text in blue
    print("\033[94m" + row['input_text'] + "\033[0m", end="")
    
    # Color-coded continuation (character level)
    colored_continuation = color_code_differences(
        row['generated_continuation_text'],
        row['true_continuation_text']
    )
    print(colored_continuation)
    print()
    
    # Token-level view
    print("\nToken-level view:")
    print("-" * 40)
    
    # Tokenize input and continuations
    input_tokens = tokenizer.encode(row['input_text'], add_special_tokens=False)
    generated_tokens = tokenizer.encode(row['generated_continuation_text'], add_special_tokens=False)
    ground_truth_tokens = tokenizer.encode(row['true_continuation_text'], add_special_tokens=False)
    
    # # Concatenated view
    # print("Concatenated view:")
    # input_tokens_str = "".join([f"[{tokenizer.decode(token)}]" for token in input_tokens])
    # print("\033[94m" + input_tokens_str + "\033[0m", end=" ")
    colored_token_continuation = color_code_token_differences(
        generated_tokens,
        ground_truth_tokens,
        tokenizer
    )
    # print(colored_token_continuation)
    
    # Token-by-token view
    # print("\nToken-by-token view:")
    print("Input tokens:")
    print("\033[94m" + "".join([f"[{tokenizer.decode(token)}]" for token in input_tokens]) + "\033[0m")
    
    print("\nGenerated continuation tokens:")
    # print("\033[91m" + "".join([f"[{tokenizer.decode(token)}]" for token in generated_tokens]) + "\033[0m")
    # Instead of all red, make this just the colored_token_continuation without the input tokens
    print(colored_token_continuation)
    
    print("\nGround truth continuation tokens:")
    print("\033[92m" + "".join([f"[{tokenizer.decode(token)}]" for token in ground_truth_tokens]) + "\033[0m")
    
    # Show match status
    match_status = "PERFECT MATCH" if row['perfect_match'] else "PARTIAL MATCH"
    print(f"\nStatus: {match_status}")
    
    print("-" * 80)
    print("\nNavigation:")
    print("n: next | p: previous | q: quit | number: jump to that number")
    
def main():
    # Check if file exists
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python view_mem_nonmem_data.py <results_file.csv> [filter_type]")
        print("Filter types: default, perfect, no_match")
        sys.exit(1)
        
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: File {results_file} not found")
        sys.exit(1)
    
    # Parse filter type
    filter_type = MatchFilter.DEFAULT
    if len(sys.argv) == 3:
        try:
            filter_type = MatchFilter(sys.argv[2])
        except ValueError:
            print("Invalid filter type. Using default.")
    
    # Initialize tokenizer
    print("Loading Llama tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Apply filtering
    if filter_type == MatchFilter.PERFECT_MATCH:
        df = df[df['perfect_match'] == True]
    elif filter_type == MatchFilter.NO_MATCH:
        df = df[df['matching_tokens'] == 0]
    
    if len(df) == 0:
        print("No matches found in the results with the current filter!")
        sys.exit(0)
    
    print(f"Found {len(df)} total matches")
    perfect_count = len(df[df['perfect_match'] == True])
    print(f"Perfect matches: {perfect_count}")
    print(f"Partial matches: {len(df) - perfect_count}")
    print(f"Current filter: {filter_type.value}")
    input("Press Enter to start viewing...")
    
    current_index = 0
    while True:
        display_match(df.iloc[current_index], current_index, len(df), tokenizer)
        
        command = input("\nEnter command: ").strip().lower()
        
        if command == 'q':
            break
        elif command == 'n':
            current_index = (current_index + 1) % len(df)
        elif command == 'p':
            current_index = (current_index - 1) % len(df)
        elif command.isdigit():
            idx = int(command) - 1
            if 0 <= idx < len(df):
                current_index = idx
            else:
                input(f"Invalid number. Press Enter to continue...")
        else:
            input("Invalid command. Press Enter to continue...")

if __name__ == "__main__":
    main() 