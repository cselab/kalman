import re

def keep_best_functions_only(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if "best score so far:" in line:
            # Keep the score line and the function that follows
            output_lines.append(line)
            # Skip until the next 'def aproximate' is found
            while i < len(lines) and not lines[i].strip().startswith("def aproximate"):
                i += 1
            # Add the function block
            while i < len(lines):
                output_lines.append(lines[i])
                if lines[i].strip() == "":
                    break
                i += 1
        else:
            i += 1

    with open(output_file, 'w') as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    input_path = "pytorch_18207988.out"     # replace with your input filename
    output_path = "pytorch_1820798822.out"     # the cleaned output file
    keep_best_functions_only(input_path, output_path)
    print(f"Filtered file saved to {output_path}")
