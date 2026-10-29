import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import Counter
from esg_utils import get_collection

_, c = get_collection()
print("總筆數:", c.count())
metas = c.get(include=["metadatas"])["metadatas"]
print(Counter(m.get("doc_type") for m in metas))