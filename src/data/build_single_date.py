#!/usr/bin/env python3
"""Select one forecast date per event from a memory-free dataset.

Seeded event groups sample early, middle, or late thirds of each event's
date sequence, using relative position (t - 1) / (T_Q - 1). Empty thirds use
the date nearest their centre. Selected rows retain their content and source
order; selection.jsonl and build_stats.json record the sampling details.
"""
import argparse
import collections
import json
import random
import statistics
from pathlib import Path

BINS = {"early": (0.0, 1 / 3), "middle": (1 / 3, 2 / 3), "late": (2 / 3, 1.0 + 1e-9)}
CENTRES = {"early": 1 / 6, "middle": 0.5, "late": 5 / 6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="Memory-free input JSONL.")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL; sidecar files are written beside it.")
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    rows = [json.loads(l) for l in a.input.open()]
    by_event: dict = collections.OrderedDict()
    for i, r in enumerate(rows):
        by_event.setdefault(r["event_id"], []).append((i, r))
    for ev, lst in by_event.items():
        lst.sort(key=lambda x: x[1]["metadata"]["forecast_date"])
        dates = [x[1]["metadata"]["forecast_date"] for x in lst]
        assert len(set(dates)) == len(dates), f"duplicate forecast_date in event {ev}"

    rng = random.Random(a.seed)
    events = list(by_event)
    rng.shuffle(events)
    n = len(events)
    groups = {
        ev: ("early" if k < n / 3 else "middle" if k < 2 * n / 3 else "late")
        for k, ev in enumerate(events)
    }

    chosen = []  # (source_index, row, record)
    fallback = 0
    for ev, lst in by_event.items():
        T = len(lst)
        g = groups[ev]
        lo, hi = BINS[g]
        cands = [(t, (t - 1) / (T - 1) if T > 1 else 0.0) for t in range(1, T + 1)]
        in_bin = [(t, r) for t, r in cands if lo <= r < hi]
        if not in_bin:
            fallback += 1
            in_bin = [min(cands, key=lambda tr: abs(tr[1] - CENTRES[g]))]
        t, r_t = rng.choice(in_bin)
        src_idx, row = lst[t - 1]
        chosen.append((src_idx, row, {
            "event_id": ev,
            "slug": row.get("slug"),
            "T_Q": T,
            "group": g,
            "t": t,
            "r_t": round(r_t, 4),
            "forecast_date": row["metadata"]["forecast_date"],
            "label": row.get("label"),
            "belief_kind": row.get("belief_kind"),
        }))
    chosen.sort(key=lambda x: x[0])

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for _, row, _rec in chosen:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (a.out.parent / "selection.jsonl").open("w") as f:
        for _, _row, rec in chosen:
            f.write(json.dumps(rec) + "\n")

    recs = [rec for _, _, rec in chosen]
    stats = {
        "input": str(a.input),
        "output": str(a.out),
        "seed": a.seed,
        "events": n,
        "rows_in": len(rows),
        "rows_out": len(chosen),
        "fallbacks": fallback,
        "group_counts": dict(collections.Counter(r["group"] for r in recs)),
        "r_t_mean_by_group": {
            g: round(statistics.mean(r["r_t"] for r in recs if r["group"] == g), 4)
            for g in BINS
        },
        "r_t_overall_mean": round(statistics.mean(r["r_t"] for r in recs), 4),
        "r_t_hist_10bins": dict(sorted(collections.Counter(min(int(r["r_t"] * 10), 9) for r in recs).items())),
        "t_hist": dict(sorted(collections.Counter(r["t"] for r in recs).items())),
        "T_Q_hist": dict(sorted(collections.Counter(r["T_Q"] for r in recs).items())),
        "label_dist_out": dict(collections.Counter(str(r["label"]) for r in recs).most_common(6)),
        "kind_out": dict(collections.Counter(r["belief_kind"] for r in recs)),
        "kind_in": dict(collections.Counter(r["belief_kind"] for r in rows)),
    }
    (a.out.parent / "build_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
