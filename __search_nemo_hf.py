from pathlib import Path
import re

root = Path("C:/Users/Zain_/anaconda3/envs/audio-notes-gpu/Lib/site-packages/nemo")
pattern = re.compile(r"class\s+HFHubMixin")

for path in root.rglob("*.py"):
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        continue
    if pattern.search(text):
        print(path)
