import glob
import os
from typing import List, Dict

def load_markdown_files(folder_path: str) -> List[Dict]:
    md_files = glob.glob(os.path.join(folder_path, "**/*.md"), recursive=True)
    mdx_files = glob.glob(os.path.join(folder_path, "**/*.mdx"), recursive=True)
    all_files = md_files + mdx_files

    docs = []
    for path in all_files:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                docs.append({
                    "filename": os.path.relpath(path, folder_path),
                    "content": content
                })
        except UnicodeDecodeError as e:
            print(f"❌ Error reading {path}: {e}")
    return docs