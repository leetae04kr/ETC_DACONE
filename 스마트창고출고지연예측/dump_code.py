import json

with open('code/maincode.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('all_code.py', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"# === CELL {i} ({cell['cell_type']}) ===\n")
        out.write("".join(cell['source']))
        out.write("\n\n")
