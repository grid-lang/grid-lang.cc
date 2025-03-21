import sys
import argparse
import os
from .main import run_grid

def main():
    parser = argparse.ArgumentParser(description='Grid Language Interpreter')
    parser.add_argument('filename', help='Path to the .grid file to execute')
    parser.add_argument('--debug', action='store_true', help='Output grid state in CSV format')
    
    args = parser.parse_args()
    
    try:
        with open(args.filename, 'r') as file:
            code = file.read()
        
        # Run the interpreter and get CSV output if debug mode is enabled
        csv_output = run_grid(code, debug=args.debug)
        
        # If in debug mode and we have CSV output, save it to a file
        if args.debug and csv_output is not None:
            # Generate CSV filename by replacing .grid extension with .csv
            csv_filename = os.path.splitext(args.filename)[0] + '.csv'
            with open(csv_filename, 'w', newline='') as csvfile:
                csvfile.write(csv_output)
            print(f"\nGrid state saved to: {csv_filename}")
            
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 