import json
import os
import sys

with open(r"c:\Users\ANSH SINGH\Desktop\capstone1\v4\v4-capstone.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

out_f = open(r"c:\Users\ANSH SINGH\Desktop\capstone1\v4\v4_output.txt", "w", encoding="utf-8")

def myprint(*args, **kwargs):
    print(*args, **kwargs, file=out_f)

myprint(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    source_lines = cell.get('source', [])
    source_preview = "".join(source_lines)[:100].strip().replace('\n', ' ')
    myprint("=" * 60)
    myprint(f"CELL {i} [{cell.get('cell_type')}]: {source_preview}")
    outputs = cell.get('outputs', [])
    if not outputs:
        myprint("  (No outputs)")
    for out in outputs:
        out_type = out.get('output_type')
        if out_type == 'stream':
            text = "".join(out.get('text', []))
            myprint(f"  --- Stream ({out.get('name')}) ---")
            myprint(text.strip())
        elif out_type == 'error':
            myprint(f"  --- ERROR: {out.get('ename')} ---")
            myprint(out.get('evalue'))
            tb = out.get('traceback', [])
            if tb:
                myprint("\n".join(tb[-5:]))
        elif 'data' in out:
            data = out.get('data', {})
            for k in data.keys():
                if k == 'text/plain':
                    myprint(f"  --- text/plain ---: {data[k]}")
                else:
                    myprint(f"  --- {k} data present ---")

out_f.close()
print("Done writing v4_output.txt")
