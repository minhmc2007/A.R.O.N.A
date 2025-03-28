import json

with open("data.json", "r") as f:
    data = json.load(f)

for item in data:
    print("Instruction:", item["instruction"])
    print("Response:", item["response"])
    print("-" * 30)