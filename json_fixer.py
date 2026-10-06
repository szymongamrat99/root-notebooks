import json
import re

input_filename = "signal_kinfit_test.ipynb"
output_filename = "naprawiony_notebook.ipynb"

with open(input_filename, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# Szukamy bloku "cells": [ ... ]
cells_match = re.search(r'"cells"\s*:\s*\[(.*)\]\s*,\s*"metadata"', text, re.DOTALL)
if not cells_match:
    cells_match = re.search(r'"cells"\s*:\s*\[(.*)', text, re.DOTALL)

raw_cells_text = cells_match.group(1) if cells_match else text

# Dzielimy na poszczególne komórki po kluczu "cell_type"
cell_blocks = re.split(r'\{\s*"cell_type"', raw_cells_text)

extracted_cells = []

for block in cell_blocks:
    if not block.strip():
        continue
    
    # Rozpoznanie typu komórki
    cell_type = "code"
    if '"markdown"' in block[:30]:
        cell_type = "markdown"
    elif '"raw"' in block[:30]:
        cell_type = "raw"
        
    # Wyciąganie zawartości "source": [...]
    source_match = re.search(r'"source"\s*:\s*\[(.*?)\]\s*(?:,|\})', block, re.DOTALL)
    if source_match:
        raw_source = source_match.group(1)
        lines = re.findall(r'"((?:[^"\\]|\\.)*)"', raw_source)
        
        # Odkodowywanie znaków specjalnych
        clean_lines = []
        for line in lines:
            try:
                decoded = bytes(line, "utf-8").decode("unicode_escape")
                clean_lines.append(decoded)
            except Exception:
                clean_lines.append(line + "\n")
                
        if clean_lines:
            cell_data = {
                "cell_type": cell_type,
                "metadata": {},
                "source": clean_lines
            }
            
            # Pola wymagane tylko dla komórek kodu
            if cell_type == "code":
                cell_data["outputs"] = []
                cell_data["execution_count"] = None
                
            extracted_cells.append(cell_data)

# Budujemy strukturę czystego notebooka od zera
new_notebook = {
    "cells": extracted_cells,
    "metadata": {
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Zapisujemy nowy plik .ipynb
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(new_notebook, f, indent=2, ensure_ascii=False)

print(f"Sukces! Odzyskano {len(extracted_cells)} komórek. Utworzono plik {output_filename}.")