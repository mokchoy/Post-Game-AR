"""Tag detection summary for one or more runs (needs the Step 46 tag_log). Usage: python3 scripts/tag_stats.py runs/<id> [...]"""
import json
import sys
from collections import Counter
import numpy as np

for run in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{run}/frames.jsonl")]
    log = [r["diagnostics"]["tag_log"] for r in rows]
    n = len(rows)
    side = [t["anchor_min_side_px"] for t in log if t["anchor_min_side_px"]]
    rms = [t["anchor_rms_px"] for t in log if t["anchor_rms_px"] is not None]
    last = rows[-1]["diagnostics"]
    print(f"{run[-17:]} frames {n} | tag ok {np.mean([t['anchor_status'] == 'ok' for t in log]):.0%} | statuses {dict(Counter(t['anchor_status'][:30] for t in log).most_common(3))}"
          f" | min side px median {np.median(side) if side else 0:.0f} | rms median {np.median(rms) if rms else 0:.2f}"
          f" | poses {dict(Counter(r['actual_pose_source'] for r in rows))} | keyframes {last['keyframes']} stable {last['stable_landmarks']}")
