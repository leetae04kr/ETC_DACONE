import json

with open('code/maincode.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"--- Cell {i} ({cell['cell_type']}) ---")
    print("".join(cell['source'])[:200] + ("..." if len("".join(cell['source'])) > 200 else ""))
