import json
from pathlib import Path

data = json.loads(Path("data/processed/2024年永續報告書(中)中信證券000616.json").read_text(encoding="utf-8"))
for s in data:
        if s["pillar"] == "E":
            print(s["section_id"], "|", s["text"][:60])