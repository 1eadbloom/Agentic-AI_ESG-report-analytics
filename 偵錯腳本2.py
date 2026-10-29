import json
from pathlib import Path
data = json.loads(Path("data/processed/2024年永續報告書(中)中信證券000616.json").read_text(encoding="utf-8"))
from collections import Counter
print(Counter(s["pillar"] for s in data))