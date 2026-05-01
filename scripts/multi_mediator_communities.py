#!/usr/bin/env python3
"""
Multi-Mediator Joint Mediation Analysis — Communities 3, 9, 11

Estimates the causal effect of each community's sentiment on ideology (Y),
decomposed into:
  - NDE: Natural Direct Effect (pathway NOT through the other two community sentiments)
  - NIE: Natural Indirect Effect jointly mediated through BOTH other community sentiments

The mediator vector M = (F_A, F_B) is treated jointly: NIE captures the
combined indirect pathway through both remaining communities simultaneously,
and each mediator's counterfactual is fitted separately then combined in the
outcome model.

Causal roles per round:
  Treatment (F_T):  sentiment community_X                       (continuous raw score)
  Joint Mediators:  [sentiment community_A, sentiment community_B]
  Confounders (W):  PCA-compressed topic-presence indicators    (min_presence filter applied)
  Collider (X):     article text                                (never conditioned on)
  Outcome (Y):      int_bias  (0=Left, 1=Center, 2=Right)

Three round-robin configurations:
  community_3  → mediators: (community_9,  community_11)
  community_9  → mediators: (community_3,  community_11)
  community_11 → mediators: (community_3,  community_9)

Expected behaviour under min_presence=50:
  T=community_3  (N=132): both mediators filtered → degenerate, no NDE/NIE, logged and skipped
  T=community_9  (N=139): community_11 survives; community_3 may be filtered → 1 or 2 mediators
  T=community_11 (N=772): community_9 survives; community_3 may be filtered → 1 or 2 mediators

Output per case (inside --output-root/<dataset>/<dir_name>/):
  <dir_name>.log               — full verbose log
  mediation_summary_table.csv  — TE/NDE/NIE per contrast
  cate_heterogeneity_summary.csv
  cate_by_ideology.png
  mediation_forest_plot.png    — if available

Usage
-----
# All data/*.pkl files, defaults
python examples/multi_mediator_communities.py

# Override bootstrap count and output root
python examples/multi_mediator_communities.py --n-bootstrap 500 --output-root results/mm_test

# Specific dataset files
python examples/multi_mediator_communities.py data/allsides_df_test.pkl --n-bootstrap 500
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from utils.ate_cate_analysis import (
    TreatmentConfig, run_stratified_analysis,
    InsufficientTopicPresenceError, InsufficientSamplesError,
)
from utils.ate_cate_reporting import (
    print_mediation_summary,
    export_mediation_results,
    plot_mediation_results,
)
from utils.community_utils import build_community_df, resolve_topic_name


# ── Analysis hyperparameters ──────────────────────────────────────────────────

APPLY_PCA_TO_CONFOUNDERS = True
PCA_VARIANCE_THRESHOLD   = 0.95
PCA_MAX_COMPONENTS       = 50
MIN_TOPIC_PRESENCE       = 50   # matches existing community mediation runs
BOOTSTRAP_SAMPLES        = [2000]

# Communities under analysis
COMMUNITY_IDS = [3, 9, 11]

# Round-robin: each community as treatment, other two as joint mediators
COMMUNITY_CASES = [
    {'treatment': 3,  'mediators': [9,  11]},
    {'treatment': 9,  'mediators': [3,  11]},
    {'treatment': 11, 'mediators': [3,  9]},
]


# ── Utilities ─────────────────────────────────────────────────────────────────

class TeeStream:
    """Write to both the original stdout and a log file simultaneously."""

    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file

    def write(self, msg):
        self.original.write(msg)
        self.log_file.write(msg)

    def flush(self):
        self.original.flush()
        self.log_file.flush()


def make_dir_name(treatment_id: int, mediator_ids: list, n_bootstrap: int) -> str:
    med_str = '_'.join(str(m) for m in sorted(mediator_ids))
    return f"bin_pca_t_community_{treatment_id}_m_communities_{med_str}_{n_bootstrap}"


# ── Single case runner ────────────────────────────────────────────────────────

def run_case(
    community_df: pd.DataFrame,
    treatment_id: int,
    mediator_ids: list,
    n_bootstrap: int,
    output_base_dir: str,
    min_topic_presence: int = MIN_TOPIC_PRESENCE,
    pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD,
):
    """
    Run one multi-mediator mediation case and save all outputs.

    Parameters
    ----------
    community_df : DataFrame with 'topic community_X' / 'sentiment community_X' columns
    treatment_id : community ID used as treatment (e.g. 9)
    mediator_ids : list of community IDs used as joint mediators (e.g. [3, 11])
    n_bootstrap  : bootstrap iterations for NDE/NIE CIs
    output_base_dir : root directory for this dataset's outputs
    min_topic_presence : minimum co-occurrence count to retain a mediator/confounder
    pca_variance_threshold : fraction of variance retained by PCA on W
    """
    treatment_name = f'community_{treatment_id}'
    # Column names that must survive the min_presence filter to count as mediators
    requested_mediator_cols = [f'sentiment community_{m}' for m in mediator_ids]

    dir_name   = make_dir_name(treatment_id, mediator_ids, n_bootstrap)
    output_dir = os.path.join(output_base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"{dir_name}.log")
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeStream(original_stdout, log_file)

    try:
        start = time.perf_counter()

        print("=" * 80)
        print(f"MULTI-MEDIATOR MEDIATION ANALYSIS")
        print(f"  Treatment : community_{treatment_id}")
        print(f"  Mediators : communities {mediator_ids}  (joint vector M)")
        print(f"  Bootstrap : {n_bootstrap}")
        print("=" * 80)

        # Validate treatment column exists
        treatment_resolved = resolve_topic_name(community_df, treatment_name)
        config = TreatmentConfig(topic_name=treatment_resolved, min_samples=50)

        print(f"\nRequested joint mediators: {requested_mediator_cols}")
        print(f"These will be filtered to whichever survive the "
              f"min_topic_presence={min_topic_presence} co-occurrence threshold.")

        # ── Core analysis ──────────────────────────────────────────────────
        results = run_stratified_analysis(
            community_df,
            config,
            include_other_sentiments=True,
            include_topic_indicators=True,
            estimate_mediation=True,
            mediator_subset=requested_mediator_cols,
            verbose=True,
            random_state=42,
            n_bootstrap=n_bootstrap,
            apply_pca_to_confounders=APPLY_PCA_TO_CONFOUNDERS,
            pca_variance_threshold=pca_variance_threshold,
            pca_max_components=PCA_MAX_COMPONENTS,
            min_topic_presence=min_topic_presence,
        )

        # Report which mediators actually survived
        actual_mediator_cols = results['overall']['data'].get('mediator_cols', [])
        if actual_mediator_cols:
            print(f"\nActive mediators after filter ({len(actual_mediator_cols)}): "
                  f"{actual_mediator_cols}")
            filtered_out = [c for c in requested_mediator_cols
                            if c not in actual_mediator_cols]
            if filtered_out:
                print(f"Filtered out (below min_presence={min_topic_presence}): "
                      f"{filtered_out}")
        else:
            print("\nNo mediators survived the min_presence filter.")

        print_mediation_summary(results, treatment_resolved)

        # ── Export raw results ─────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("EXPORTING RESULTS")
        print("=" * 80)
        all_results = {treatment_resolved: results}
        export_mediation_results(all_results, output_dir=output_dir)

        # ── Summary table ─────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)

        med_res = results['overall'].get('mediation_results') or {}
        primary_key = next(iter(med_res), None)

        rows = []
        for label, result in med_res.items():
            if result is None:
                continue
            prop_reliable = result.get('prop_mediated_reliable', True)
            rows.append({
                'Treatment':      f'community_{treatment_id}',
                'Mediators':      str(sorted(actual_mediator_cols)),
                'Contrast':       label,
                'n_mediators':    result['n_mediators'],
                'N':              result['n_samples'],
                'TE':             f"{result['te']:.4f}",
                'TE_CI':          f"[{result['te_ci'][0]:.4f}, {result['te_ci'][1]:.4f}]",
                'NDE':            f"{result['nde']:.4f}",
                'NDE_CI':         f"[{result['nde_ci'][0]:.4f}, {result['nde_ci'][1]:.4f}]",
                'NIE':            f"{result['nie']:.4f}",
                'NIE_CI':         f"[{result['nie_ci'][0]:.4f}, {result['nie_ci'][1]:.4f}]",
                'Prop_Mediated':  f"{result['prop_mediated']:.1%}" if prop_reliable else "N/A",
                'decomp_error':   f"{result['decomposition_error']:.6f}",
                'n_boot_success': result.get('n_bootstrap_success', n_bootstrap),
            })

        if rows:
            summary_df = pd.DataFrame(rows)
            print(f"\nMain results (primary contrast: {primary_key}):")
            print(summary_df.to_string(index=False))
            summary_path = os.path.join(output_dir, 'mediation_summary_table.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSummary table saved to: {summary_path}")
        else:
            print("\nNo mediation results available (check mediator filter warnings above).")

        # ── CATE heterogeneity ─────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("CATE HETEROGENEITY BY OUTLET IDEOLOGY")
        print("=" * 80)

        het = results['overall'].get('cate_heterogeneity') or {}
        het_rows = []
        for ideology_label in ['Left', 'Center', 'Right']:
            r = het.get(ideology_label, {})
            if not r or r.get('note'):
                continue
            row = {
                'Treatment':   f'community_{treatment_id}',
                'Ideology':    ideology_label,
                'N':           r['n'],
                'Mean_CATE':   round(r['mean_cate'], 4),
                'Std_CATE':    round(r['std_cate'], 4),
                'SE_CI_Lower': round(r['se_ci_lower'], 4),
                'SE_CI_Upper': round(r['se_ci_upper'], 4),
            }
            if 'forest_ci_lower' in r:
                row['Forest_CI_Lower'] = round(r['forest_ci_lower'], 4)
                row['Forest_CI_Upper'] = round(r['forest_ci_upper'], 4)
            het_rows.append(row)

        if het_rows:
            het_df = pd.DataFrame(het_rows)
            print("\nCATEs grouped by outlet ideology:")
            print(het_df.to_string(index=False))
            het_path = os.path.join(output_dir, 'cate_heterogeneity_summary.csv')
            het_df.to_csv(het_path, index=False)
            print(f"\nCATE heterogeneity saved to: {het_path}")
        else:
            print("\nNo CATE heterogeneity results.")

        # ── Plots ──────────────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)

        try:
            figs = plot_mediation_results(results, treatment_resolved, save_dir=output_dir)
            print(f"Generated {len(figs)} plot(s).")
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")

        elapsed = (time.perf_counter() - start) / 60
        print(f"\nCase completed in {elapsed:.4f} minutes.")
        print("=" * 80 + "\n")

    except InsufficientSamplesError as e:
        print(f"\nSKIPPED — insufficient articles for treatment community_{treatment_id}:")
        print(f"  {e}")
        print("=" * 80 + "\n")

    except InsufficientTopicPresenceError as e:
        print(f"\nSKIPPED — all requested mediators {requested_mediator_cols} were "
              f"removed by the min_presence={min_topic_presence} filter:")
        print(f"  {e}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nERROR in case T=community_{treatment_id}: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")

    finally:
        sys.stdout = original_stdout
        log_file.close()


# ── CLI entry point ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        'input_files', nargs='*',
        help='Dataset .pkl files. Defaults to all data/*.pkl',
    )
    p.add_argument(
        '--communities', default='communities.json',
        help='Path to communities JSON (default: communities.json)',
    )
    p.add_argument(
        '--output-root',
        default=f'results/multi_mediator_communities_3_9_11_min_pres_{MIN_TOPIC_PRESENCE}',
        help='Root directory for all outputs.',
    )
    p.add_argument(
        '--n-bootstrap', type=int, default=None,
        help='Override BOOTSTRAP_SAMPLES with a single value (e.g. 500).',
    )
    p.add_argument(
        '--min-topic-presence', type=int, default=MIN_TOPIC_PRESENCE,
        help=f'Min co-occurrence count to keep a topic in W/M (default: {MIN_TOPIC_PRESENCE}).',
    )
    p.add_argument(
        '--pca-variance', type=float, default=PCA_VARIANCE_THRESHOLD,
        help=f'Fraction of variance retained by PCA on confounders (default: {PCA_VARIANCE_THRESHOLD}).',
    )
    return p


def main():
    args = build_parser().parse_args()
    bootstrap_list = [args.n_bootstrap] if args.n_bootstrap else BOOTSTRAP_SAMPLES

    input_files = args.input_files or sorted(str(p) for p in Path('data').glob('*.pkl'))
    if not input_files:
        print("No .pkl files found in data/ — nothing to do.")
        return

    # Load communities JSON and restrict to the three communities of interest
    with open(args.communities) as f:
        all_communities = json.load(f)
    communities_subset = {
        str(k): v for k, v in all_communities.items()
        if int(k) in COMMUNITY_IDS
    }

    print("=" * 80)
    print("MULTI-MEDIATOR COMMUNITY MEDIATION ANALYSIS")
    print(f"  Communities    : {COMMUNITY_IDS}")
    print(f"  Framework      : round-robin — each community as treatment,")
    print(f"                   other two as JOINT mediator vector M")
    print(f"  Min presence   : {args.min_topic_presence}")
    print(f"  Bootstrap B    : {bootstrap_list}")
    print(f"  Treatment enc. : continuous (raw sentiment score)")
    print(f"  Confounders    : PCA on topic indicators (var >= {args.pca_variance})")
    print("=" * 80)
    print(f"\nDatasets ({len(input_files)}):")
    for p in input_files:
        print(f"  {p}")

    total_start = time.perf_counter()

    for pkl_path in input_files:
        stem = Path(pkl_path).stem
        output_base_dir = os.path.join(args.output_root, stem)

        print(f"\n{'=' * 80}")
        print(f"DATASET : {pkl_path}")
        print(f"Output  : {output_base_dir}")
        print(f"{'=' * 80}")

        print("\nLoading data...")
        df = pd.read_pickle(str(pkl_path))
        print(f"  {len(df)} articles loaded.")

        print("\nBuilding community DataFrame (communities 3, 9, 11)...")
        community_df = build_community_df(df, communities_subset, verbose=True)

        for case in COMMUNITY_CASES:
            for n_bootstrap in bootstrap_list:
                t_id  = case['treatment']
                m_ids = case['mediators']
                dir_name = make_dir_name(t_id, m_ids, n_bootstrap)
                print(f"\n{'─' * 60}")
                print(f"Running: {stem}/{dir_name}")
                print(f"  T=community_{t_id}  |  M=communities {m_ids}  |  B={n_bootstrap}")
                print(f"{'─' * 60}")

                run_start = time.perf_counter()
                run_case(
                    community_df=community_df,
                    treatment_id=t_id,
                    mediator_ids=m_ids,
                    n_bootstrap=n_bootstrap,
                    output_base_dir=output_base_dir,
                    min_topic_presence=args.min_topic_presence,
                    pca_variance_threshold=args.pca_variance,
                )
                run_elapsed = (time.perf_counter() - run_start) / 60
                print(f"  Finished in {run_elapsed:.2f} min → {output_base_dir}/{dir_name}/")

    total_elapsed = (time.perf_counter() - total_start) / 60
    print(f"\n{'=' * 80}")
    print(f"All runs completed in {total_elapsed:.2f} minutes.")
    print("=" * 80)


if __name__ == '__main__':
    main()
