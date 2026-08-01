"""One-off diagnostic: summarize fingerprint spans and warp-path runs from a report JSON."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else r"Movies\Sinister\sinister.diag.report.json"
r = json.load(open(path))
print("confidence:", r["confidence"], " mode:", r["mode"], " speed:", r.get("speed_stretch"))
print()
print("=== fingerprint spans (ad_start-ad_end -> offset, matches) ===")
for s in r["fingerprint_spans"]:
    flag = ""
    if abs(s["offset"] - 45.5) > 3.0:
        flag = "   <-- SUSPECT"
    print(f'  {s["ad_start"]:7.0f}-{s["ad_end"]:7.0f}  ->  {s["offset"]:+9.2f} s   ({s["matches"]:5d} m){flag}')
print()
print("=== warp path: contiguous runs by offset (jump > 2 s starts new run) ===")
pts = r["warp_path"]["points"]
runs = []
cur = [pts[0]]
for p in pts[1:]:
    if abs((p["target_time"] - p["source_time"]) - (cur[-1]["target_time"] - cur[-1]["source_time"])) > 2.0:
        runs.append(cur)
        cur = []
    cur.append(p)
runs.append(cur)
for run in runs:
    o0 = run[0]["target_time"] - run[0]["source_time"]
    o1 = run[-1]["target_time"] - run[-1]["source_time"]
    dur = run[-1]["source_time"] - run[0]["source_time"]
    mc = sum(p["confidence"] for p in run) / len(run)
    print(f'  ad {run[0]["source_time"]:7.1f}-{run[-1]["source_time"]:7.1f} ({dur:6.1f}s, {len(run):4d} pts)  '
          f'offset {o0:+8.2f} .. {o1:+8.2f}  meanconf {mc:.2f}')
