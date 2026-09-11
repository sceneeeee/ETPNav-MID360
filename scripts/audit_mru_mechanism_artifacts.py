#!/usr/bin/env python3
"""Read-only audit of frozen waypoint telemetry; writes derived audit artifacts only."""
import json
import math
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "mru_mechanism_audit"
BINS = [i / 4 for i in range(1, 13)]
RUNS = {
    "Original ETP": ROOT / "outputs/original_baseline20_valid_renderer/original_baseline20_valid_renderer/telemetry_ep_ckpt_59_val_unseen_r0_w1.json",
    "Geometry V0-BNMS": ROOT / "outputs/geometry_v0_bounded_nms20_valid_renderer/geometry_v0_bounded_nms20_valid_renderer/telemetry_ep_ckpt_59_val_unseen_r0_w1.json",
    "MRU-BNMS": ROOT / "outputs/geometry_mru_bnms20_valid_renderer/geometry_mru_bnms20_valid_renderer/telemetry_ep_ckpt_59_val_unseen_r0_w1.json",
}

def q(xs, p):
    xs = sorted(xs)
    if not xs: return None
    z = (len(xs) - 1) * p; lo, hi = math.floor(z), math.ceil(z)
    return xs[lo] if lo == hi else xs[lo] + (xs[hi]-xs[lo])*(z-lo)

def nearest_bin(x): return min(BINS, key=lambda b: abs(b-x))
def hist(xs, mapped=False):
    h = {f"{b:.2f}": 0 for b in BINS}
    for x in xs:
        b = nearest_bin(x) if mapped else x
        key = f"{b:.2f}"
        if key in h: h[key] += 1
        else: h.setdefault(key, 0); h[key] += 1
    n = len(xs)
    return {k: v/n if n else 0 for k,v in h.items() if v or k in {f"{b:.2f}" for b in BINS}}
def entropy(h):
    return -sum(p * math.log2(p) for p in h.values() if p)
def stats(xs, map_bins=False):
    h=hist(xs, map_bins)
    return {"count":len(xs), "mean":mean(xs) if xs else None, "std_sample":stdev(xs) if len(xs)>1 else 0,
            "median":q(xs,.5), "iqr":q(xs,.75)-q(xs,.25) if xs else None,
            "p5":q(xs,.05), "p25":q(xs,.25), "p75":q(xs,.75), "p95":q(xs,.95),
            "min":min(xs) if xs else None, "max":max(xs) if xs else None,
            "distinct_distances":len(set(xs)), "histogram":h, "entropy_bits":entropy(h)}
def decision_stats(cands):
    ds=[]
    for cs in cands:
        xs=[c['distance'] for c in cs]
        if not xs: continue
        counts={x:xs.count(x) for x in set(xs)}; modal=max(counts.values())
        ds.append({"candidate_count":len(xs), "unique_radial_bins":len(counts),
                   "modal_fraction":modal/len(xs), "range":max(xs)-min(xs),
                   "std_sample":stdev(xs) if len(xs)>1 else 0,
                   "all_same":len(counts)==1, "has_1_5":1.5 in counts,
                   "all_le_1_5":max(xs)<=1.5,
                   "one_point_five_modal": counts.get(1.5,0)==modal and 1.5 in counts,
                   "all_1_5":len(counts)==1 and 1.5 in counts})
    def agg(field): return [x[field] for x in ds]
    return {"decision_count":len(ds), "mean_unique_radial_bins":mean(agg('unique_radial_bins')),
            "median_unique_radial_bins":median(agg('unique_radial_bins')),
            "fraction_one_unique_bin":mean(agg('all_same')),
            "fraction_modal_ge_80pct":mean(x['modal_fraction']>=.8 for x in ds),
            "fraction_has_1_5":mean(agg('has_1_5')), "fraction_1_5_modal":mean(agg('one_point_five_modal')),
            "fraction_all_1_5":mean(agg('all_1_5')), "fraction_all_le_1_5":mean(agg('all_le_1_5')), "records":ds}

allout={}
for name, path in RUNS.items():
    data=json.loads(path.read_text())
    steps=[s for ep in data['episodes'].values() for s in ep['waypoint_trace']]
    candidate_lists=[s.get('candidates',[]) for s in steps]
    candidates=[c['distance'] for cs in candidate_lists for c in cs]
    selected=[s['selected_waypoint']['distance'] for s in steps if s.get('selected_waypoint') and s['selected_waypoint'].get('distance') is not None]
    allout[name]={"source":str(path.relative_to(ROOT)), "candidate":stats(candidates),
                  "selected":stats(selected, True), "per_decision":decision_stats(candidate_lists)}
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'radial_audit.json').write_text(json.dumps(allout,indent=2)+"\n")

def fmt(x): return '—' if x is None else (f'{x:.4f}' if isinstance(x,float) else str(x))
lines=['# MRU mechanism audit — derived radial statistics','', 'All numbers are calculated read-only from the three frozen telemetry JSON files. Entropy is Shannon entropy in bits. Selected-waypoint histograms map continuous selected distances to the nearest 0.25-m ETP bin; their means/quantiles use the stored continuous values.','', '## Candidate and selected distributions','', '| Run | Level | n | mean | sd | median | IQR | P5 | P25 | P75 | P95 | min–max | distinct | entropy (bits) |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for name, x in allout.items():
 for level in ('candidate','selected'):
  s=x[level]; lines.append(f"| {name} | {level} | {s['count']} | {fmt(s['mean'])} | {fmt(s['std_sample'])} | {fmt(s['median'])} | {fmt(s['iqr'])} | {fmt(s['p5'])} | {fmt(s['p25'])} | {fmt(s['p75'])} | {fmt(s['p95'])} | {fmt(s['min'])}–{fmt(s['max'])} | {s['distinct_distances']} | {fmt(s['entropy_bits'])} |")
lines += ['', '## Normalized radial histograms (candidate)', '']
for name,x in allout.items(): lines += [f'### {name}', '', '`'+', '.join(f'{k}:{v:.3f}' for k,v in x['candidate']['histogram'].items())+'`','']
lines += ['## Per-decision diversity', '', '| Run | decisions | mean / median unique bins | one bin | ≥80% one bin | has 1.5 m | 1.5 m modal | all 1.5 m | all ≤1.5 m |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for name,x in allout.items():
 d=x['per_decision']; lines.append(f"| {name} | {d['decision_count']} | {d['mean_unique_radial_bins']:.3f} / {d['median_unique_radial_bins']:.3f} | {d['fraction_one_unique_bin']:.3%} | {d['fraction_modal_ge_80pct']:.3%} | {d['fraction_has_1_5']:.3%} | {d['fraction_1_5_modal']:.3%} | {d['fraction_all_1_5']:.3%} | {d['fraction_all_le_1_5']:.3%} |")
(OUT/'radial_audit.md').write_text('\n'.join(lines)+'\n')
print(OUT/'radial_audit.json')
