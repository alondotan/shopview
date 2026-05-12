import csv
from collections import Counter

with open("test2_log.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total log rows: {len(rows)}")
print()

# Actions distribution
actions = Counter(r["action"] for r in rows)
print("Action breakdown:")
for action, count in sorted(actions.items(), key=lambda x: -x[1]):
    print(f"  {action:<22} {count}")
print()

# Show one interesting shopper journey (find an indecisive one)
entered = {r["shopper_id"]: r["details"] for r in rows if r["action"] == "ENTERED"}
target = next((sid for sid, d in entered.items() if "indecisive" in d), list(entered.keys())[2])
profile = entered[target]
journey = [r for r in rows if r["shopper_id"] == target]
print(f"Shopper #{target} ({profile}):")
for r in journey:
    print(f"  {r['timestamp']}  {r['action']:<22}  ({r['row']},{r['col']})  {r['details']}")
