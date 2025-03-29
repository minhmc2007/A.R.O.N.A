import json
import sys

def merge_json(file1, file2, output_file):
    try:
        with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        if not isinstance(data1, list) or not isinstance(data2, list):
            print("Error: Both JSON files must contain a list of objects.")
            return

        merged_data = data1 + data2

        with open(output_file, 'w', encoding='utf-8') as outfile:
            json.dump(merged_data, outfile, indent=4, ensure_ascii=False)

        print(f"Merged JSON saved to {output_file}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python merger.py <file1.json> <file2.json> <output.json>")
    else:
        merge_json(sys.argv[1], sys.argv[2], sys.argv[3])