import json
import glob
from pathlib import Path

# Merge text chunks
all_chunks = []
for file in glob.glob('data/clean_chunks/clean_*.json'):
    with open(file, 'r') as f:
        chunks = json.load(f)
        all_chunks.extend(chunks)

# Save the merged chunks
Path('data/clean_chunks/clean_text_chunks.json').parent.mkdir(parents=True, exist_ok=True)
with open('data/clean_chunks/clean_text_chunks.json', 'w') as f:
    json.dump(all_chunks, f, indent=2)

print(f'Merged {len(all_chunks)} chunks into clean_text_chunks.json')
