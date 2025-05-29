import pandas as pd
import sys
import os

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def display_match(row, index, total):
    """Display a single perfect match with nice formatting"""
    clear_screen()
    print(f"Perfect match {index + 1}/{total}")
    print("-" * 80)
    
    # Input text in blue
    print("\033[94m" + row['input_text'] + "\033[0m", end="")
    
    # Generated continuation in green (since it's a perfect match)
    print("\033[92m" + row['generated_continuation_text'] + "\033[0m")
    
    print("-" * 80)
    print("\nNavigation:")
    print("n: next | p: previous | q: quit | number: jump to that number")
    
def main():
    # Check if file exists
    if len(sys.argv) != 2:
        print("Usage: python view_perfect_matches.py <results_file.csv>")
        sys.exit(1)
        
    results_file = sys.argv[1]
    if not os.path.exists(results_file):
        print(f"Error: File {results_file} not found")
        sys.exit(1)
    
    # Load results
    df = pd.read_csv(results_file)
    
    # Filter for perfect matches
    perfect_matches = df[df['perfect_match'] == True].reset_index(drop=True)
    
    if len(perfect_matches) == 0:
        print("No perfect matches found in the results!")
        sys.exit(0)
    
    print(f"Found {len(perfect_matches)} perfect matches")
    input("Press Enter to start viewing...")
    
    current_index = 0
    while True:
        display_match(perfect_matches.iloc[current_index], current_index, len(perfect_matches))
        
        command = input("\nEnter command: ").strip().lower()
        
        if command == 'q':
            break
        elif command == 'n':
            current_index = (current_index + 1) % len(perfect_matches)
        elif command == 'p':
            current_index = (current_index - 1) % len(perfect_matches)
        elif command.isdigit():
            idx = int(command) - 1
            if 0 <= idx < len(perfect_matches):
                current_index = idx
            else:
                input(f"Invalid number. Press Enter to continue...")
        else:
            input("Invalid command. Press Enter to continue...")

if __name__ == "__main__":
    main() 