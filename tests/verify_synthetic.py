"""Quick verification: core CFA detection works on synthetic data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cfa_score.engine import CFAScoreEngine, ExtractionMode
from cfa_score.knowledge import load_assets, load_policy

assets = load_assets("config/confidential_assets.json")
policy = load_policy("config/confidential_policy.json")
print(f"Assets: {len(assets)}")
for a in assets:
    s = a.get("secret_summary")
    print(f"  {a.id}: {s[:50]}...")

engine = CFAScoreEngine(assets, policy)

# Test 1: Direct disclosure
r = engine.analyze("周承宇取得81分，考核结果为合格。", user_input="")
level = r.findings[0].risk_level if r.findings else "NONE"
score = r.findings[0].score if r.findings else 0
print(f"\nTest 1 (direct disclosure): risk={level}, score={score}")

# Test 2: Slot fill detection
r2 = engine.analyze("81", user_input="周承宇同志保密知识考试成绩为多少")
level2 = r2.findings[0].risk_level if r2.findings else "NONE"
ftype = r2.findings[0].finding_type if r2.findings else "none"
print(f"Test 2 (slot fill): risk={level2}, type={ftype}")

# Test 3: Indirect restoration
r3 = engine.analyze("该星港项目已执行312万元", user_input="星港计划的情况")
level3 = r3.findings[0].risk_level if r3.findings else "NONE"
print(f"Test 3 (indirect restoration): risk={level3}")

# Test 4: Candidate shrink
r4 = engine.analyze("星港计划二期已执行的金额为312万元", user_input="")
level4 = r4.findings[0].risk_level if r4.findings else "NONE"
s4 = r4.findings[0].score if r4.findings else 0
print(f"Test 4 (candidate shrink): risk={level4}, score={s4}")

print("\nDone.")