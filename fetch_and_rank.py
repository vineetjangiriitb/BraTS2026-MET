#!/usr/bin/env python3
"""
BraTS 2026 Task 1 (Brain Metastases) — Leaderboard fetcher + DELPHI-style ranker.

Fetches all ACCEPTED/SCORED submissions from Synapse evaluation queue 9619537,
resolves team names, applies the DELPHI-based rank-sum algorithm, and writes:
  - data/submissions_raw.json   (full API response)
  - data/team_names.json        (id → name map)
  - data/ranked_results.json    (final ranked output consumed by index.html)

Re-run this script whenever you want to refresh the leaderboard.

DELPHI approximation note
--------------------------
True DELPHI requires per-case metric arrays across all test cases so that
500k pairwise permutation tests can be run on the per-case distributions.
The Synapse public API only exposes per-submission aggregate scores (means).
We therefore implement the rank-sum step exactly, but skip the statistical
tiering step (p-value matrix). The HTML page clearly labels this limitation.
"""

import json
import time
import subprocess
import sys
from collections import defaultdict

# ── constants ────────────────────────────────────────────────────────────────

SYNAPSE_API = "https://repo-prod.prod.sagebase.org/repo/v1"
TABLE_ID     = "syn74508245"
EVAL_ID      = "9619537"

# The 12 metrics used for ranking (8 lesionwise + 4 small-instance F1).
# All higher-is-better.
RANKING_METRICS = [
    "lesionwise_dsc_mean_et",
    "lesionwise_nsd_mean_et",
    "lesionwise_dsc_mean_rc",
    "lesionwise_nsd_mean_rc",
    "lesionwise_dsc_mean_tc",
    "lesionwise_nsd_mean_tc",
    "lesionwise_dsc_mean_wt",
    "lesionwise_nsd_mean_wt",
    "small_instance_f1_et",
    "small_instance_f1_tc",
    "small_instance_f1_wt",
    "small_instance_f1_rc",
]

# Best-submission selection uses the same metrics as final ranking.
# This keeps the per-team submission choice aligned with the leaderboard itself.
SELECTION_METRICS = list(RANKING_METRICS)

SQL = (
    "SELECT id, createdOn, submitterid, "
    "lesionwise_dsc_mean_et, lesionwise_nsd_mean_et, "
    "lesionwise_dsc_mean_rc, lesionwise_nsd_mean_rc, "
    "lesionwise_dsc_mean_tc, lesionwise_nsd_mean_tc, "
    "lesionwise_dsc_mean_wt, lesionwise_nsd_mean_wt, "
    "small_instance_tp_et, small_instance_fn_et, small_instance_fp_et, small_instance_f1_et, "
    "small_instance_tp_tc, small_instance_fn_tc, small_instance_fp_tc, small_instance_f1_tc, "
    "small_instance_tp_wt, small_instance_fn_wt, small_instance_fp_wt, small_instance_f1_wt, "
    "small_instance_tp_rc, small_instance_fn_rc, small_instance_fp_rc, small_instance_f1_rc "
    f"FROM {TABLE_ID} "
    f"WHERE evaluationid = {EVAL_ID} "
    "AND status = 'ACCEPTED' AND submission_status = 'SCORED' "
    "ORDER BY createdOn DESC"
)

# ── helpers ──────────────────────────────────────────────────────────────────

def synapse_get(path):
    r = subprocess.run(
        ["curl", "-sf", f"{SYNAPSE_API}{path}", "-H", "Accept: application/json"],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"GET {path} failed: {r.stderr}")
    return json.loads(r.stdout)

def synapse_post(path, body):
    r = subprocess.run(
        ["curl", "-sf", "-X", "POST", f"{SYNAPSE_API}{path}",
         "-H", "Content-Type: application/json",
         "-H", "Accept: application/json",
         "-d", json.dumps(body)],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode != 0:
        raise RuntimeError(f"POST {path} failed: {r.stderr}")
    return json.loads(r.stdout)


def rankdata_avg(values):
    """Rank values descending (highest → rank 1), ties get average rank. Pure stdlib."""
    indexed = sorted(enumerate(values), key=lambda x: -x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


# ── fetch submissions ────────────────────────────────────────────────────────

def fetch_submissions():
    print("Starting async query …")
    resp = synapse_post(f"/entity/{TABLE_ID}/table/query/async/start", {"query": {"sql": SQL, "limit": 2000}})
    token = resp["token"]
    print(f"  token={token}, polling …")
    for attempt in range(20):
        time.sleep(2)
        result = synapse_get(f"/entity/{TABLE_ID}/table/query/async/get/{token}")
        if "queryResult" in result:
            print(f"  done after {attempt+1} poll(s)")
            return result
        if result.get("progressMessage"):
            print(f"  {result['progressMessage']}")
    raise TimeoutError("Async query did not complete")


# ── resolve team names ───────────────────────────────────────────────────────

def resolve_team_names(submitter_ids):
    names = {}
    for sid in submitter_ids:
        try:
            data = synapse_get(f"/team/{sid}")
            names[sid] = data.get("name", f"Team_{sid}")
        except Exception:
            names[sid] = f"Team_{sid}"
    print(f"Resolved {len(names)} team names")
    return names


# ── DELPHI rank-sum ──────────────────────────────────────────────────────────

def compute_ranking(records, team_names):
    """
    DELPHI-style rank-sum algorithm.
    Step 1: keep best submission per team (highest mean on all ranking metrics).
    Step 2: rank all teams on each metric separately (higher=better; NaN=not ranked, no penalty).
    Step 3: sum ranks per team → overall rank.
    Returns list of dicts sorted by rank_sum ascending.
    """
    by_team = defaultdict(list)
    for rec in records:
        by_team[rec["submitterid"]].append(rec)

    def selection_mean(rec):
        vals = [float(rec[m]) for m in SELECTION_METRICS if rec.get(m) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    best_by_team = {sid: max(recs, key=selection_mean) for sid, recs in by_team.items()}
    teams = sorted(best_by_team.keys())
    N, M = len(teams), len(RANKING_METRICS)

    # Build score matrix
    score_matrix = [[None] * M for _ in range(N)]
    for i, tid in enumerate(teams):
        rec = best_by_team[tid]
        for j, metric in enumerate(RANKING_METRICS):
            v = rec.get(metric)
            if v is not None:
                score_matrix[i][j] = float(v)

    # Rank per metric
    rank_matrix = [[None] * M for _ in range(N)]
    for j in range(M):
        valid_rows = [(i, score_matrix[i][j]) for i in range(N) if score_matrix[i][j] is not None]
        if not valid_rows:
            continue
        idxs, vals = zip(*valid_rows)
        ranks = rankdata_avg(list(vals))
        for k, i in enumerate(idxs):
            rank_matrix[i][j] = ranks[k]

    # Sum ranks
    rank_sums = []
    for i in range(N):
        rs = sum(v for v in rank_matrix[i] if v is not None)
        rank_sums.append(rs)

    sorted_indices = sorted(range(N), key=lambda i: rank_sums[i])

    results = []
    for pos, i in enumerate(sorted_indices):
        tid = teams[i]
        rec = best_by_team[tid]
        scores = {m: (float(rec[m]) if rec.get(m) is not None else None) for m in RANKING_METRICS}
        ranks = {RANKING_METRICS[j]: rank_matrix[i][j] for j in range(M)}
        n_scored = sum(1 for v in rank_matrix[i] if v is not None)
        # Timestamp → ms integer in createdOn
        ts_ms = rec["createdOn"]
        results.append({
            "rank": pos + 1,
            "team_id": tid,
            "team_name": team_names.get(tid, tid),
            "rank_sum": rank_sums[i],
            "n_metrics_scored": n_scored,
            "submission_id": rec["id"],
            "submitted_ms": int(ts_ms) if ts_ms else None,
            "scores": scores,
            "metric_ranks": ranks,
        })

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Fetch
    raw = fetch_submissions()
    with open(os.path.join(out_dir, "submissions_raw.json"), "w") as f:
        json.dump(raw, f, indent=2)

    headers = [h["name"] for h in raw["queryResult"]["queryResults"]["headers"]]
    api_rows = raw["queryResult"]["queryResults"]["rows"]

    def parse_row(row):
        return {headers[k]: row["values"][k] for k in range(len(headers))}

    records = [parse_row(r) for r in api_rows]
    print(f"Fetched {len(records)} submissions")

    # 2. Resolve team names
    submitter_ids = sorted({r["submitterid"] for r in records})
    team_names = resolve_team_names(submitter_ids)
    with open(os.path.join(out_dir, "team_names.json"), "w") as f:
        json.dump(team_names, f, indent=2)

    # 3. Rank
    ranked = compute_ranking(records, team_names)
    print(f"\n=== DELPHI rank-sum results ({len(ranked)} teams) ===")
    for r in ranked:
        dsc = r["scores"]["lesionwise_dsc_mean_et"]
        dsc_str = f"{dsc:.4f}" if dsc is not None else "N/A"
        print(f"  #{r['rank']:2d}  {r['team_name']:30s}  rank_sum={r['rank_sum']:6.1f}  DSC_ET={dsc_str}")

    output = {
        "generated_at_ms": int(time.time() * 1000),
        "eval_id": EVAL_ID,
        "table_id": TABLE_ID,
        "n_total_submissions": len(records),
        "n_teams": len(ranked),
        "ranking_metrics": RANKING_METRICS,
        "teams": ranked,
    }
    out_path = os.path.join(out_dir, "ranked_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
