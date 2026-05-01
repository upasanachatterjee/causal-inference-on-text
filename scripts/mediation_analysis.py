#!/usr/bin/env python3
"""
Mediation Analysis for Thesis: Politics and Donald Trump Sentiments

Decomposes total effect into:
- Natural Direct Effect (NDE): Effect NOT through other sentiments
- Natural Indirect Effect (NIE): Effect THROUGH other sentiments

This script analyzes how sentiment toward a topic affects ideology through
direct pathways vs. indirect pathways mediated by sentiments toward other topics.

Runs two treatment/mediator configurations across multiple bootstrap sample sizes.

Output files per case directory
--------------------------------
mediation_summary.csv           — primary results (Q25→Q75 percentile contrast)
mediation_forest_plot.png       — NDE/NIE forest plot with 95 % CI bars
cate_by_ideology.png            — CATE distribution by outlet ideology
cate_heterogeneity_summary.csv  — CATE statistics per ideology group
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import time
import pandas as pd
import os
from utils.ate_cate_analysis import (
    TreatmentConfig, run_stratified_analysis,
    InsufficientTopicPresenceError, InsufficientSamplesError,
)
from utils.ate_cate_reporting import (
    print_mediation_summary,
    format_mediation_table,
    export_mediation_results,
    plot_mediation_results
)
from utils.community_utils import build_community_df, resolve_topic_name


# Define the configurations to run
CASES = [
    {'treatment': 'Donald Trump', 'mediator': 'Politics'},
    {'treatment': 'Politics', 'mediator': 'Donald Trump'},
]

# B=5000 adds negligible CI precision vs B=2000 (0.001 width reduction) and
# costs 85-92 min per run.  Stop at B=2000 for the current topic pair.
BOOTSTRAP_SAMPLES = [2000]

# Dimensionality reduction on confounders (W).
# With 624 confounder columns and N=233–594 samples the estimators are
# severely under-determined.  PCA compresses W to the minimum number of
# principal components that retain PCA_VARIANCE_THRESHOLD of the variance,
# capped at PCA_MAX_COMPONENTS.  M (mediators) is never reduced.
APPLY_PCA_TO_CONFOUNDERS = True
PCA_VARIANCE_THRESHOLD = 0.95  # retain 90 % of variance in W
PCA_MAX_COMPONENTS = 50        # hard cap: never keep more than 50 PCs
# Confounder-mediator consistency: a topic must appear in at least this many
# articles in the subsample for its indicator to be kept in W and its sentiment
# to be kept in M.  Topics below this threshold contribute negligibly to any
# retained PCA component, so they cannot meaningfully control the mediator path.
MIN_TOPIC_PRESENCE = 50


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input_files', nargs='*',
                   help='Dataset .pkl files. Defaults to all data/*.pkl')
    p.add_argument('--output-root', default=f'results/mediation_analysis_min_pres_{MIN_TOPIC_PRESENCE}_with_communities_11_3',)
    p.add_argument('--n-bootstrap', type=int, default=None,
                   help='Override BOOTSTRAP_SAMPLES list with a single value.')
    p.add_argument('--communities', default=None, metavar='PATH',
                   help='Path to communities JSON. Enables community-level cases.')
    p.add_argument('--community-cases', nargs='+', default=None, metavar='T:M',
                   help='Community pairs to run, e.g. "11:1 1:11 11:0". '
                        'Format: TREATMENT_ID:MEDIATOR_ID. '
                        'Required when --communities is given.')
    p.add_argument('--communities-only', action='store_true',
                   help='Skip hardcoded topic CASES; run only community cases.')
    p.add_argument('--min-topic-presence', type=int, default=MIN_TOPIC_PRESENCE,
                   help=f'Min articles per topic to keep it in W/M (default: {MIN_TOPIC_PRESENCE}).')
    p.add_argument('--pca-variance', type=float, default=PCA_VARIANCE_THRESHOLD,
                   help=f'Fraction of variance retained by PCA on confounders '
                        f'(default: {PCA_VARIANCE_THRESHOLD}).')
    return p


def parse_community_cases(raw: list) -> list:
    """Parse ['11:1', '1:11'] → [('community_11', 'community_1'), ...]"""
    result = []
    for item in raw:
        t_id, m_id = item.split(':')
        result.append((f'community_{t_id.strip()}', f'community_{m_id.strip()}'))
    return result


def make_dir_name(treatment: str, mediator: str, n_bootstrap: int) -> str:
    """Build directory/file name from treatment, mediator, and bootstrap count."""
    t_slug = treatment.replace(' ', '_')
    m_slug = mediator.replace(' ', '_')
    return f"bin_pca_t_{t_slug}_m_{m_slug}_{n_bootstrap}"


def setup_logger():
    """Return a TeeStream class that tees stdout to a log file."""
    class TeeStream:
        """Write to both the original stream and a file."""
        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file

        def write(self, msg):
            self.original.write(msg)
            self.log_file.write(msg)

        def flush(self):
            self.original.flush()
            self.log_file.flush()

    return TeeStream


def run_single_case(
    df: pd.DataFrame,
    treatment: str,
    mediator: str,
    n_bootstrap: int,
    output_base_dir: str = 'results',
    apply_pca_to_confounders: bool = APPLY_PCA_TO_CONFOUNDERS,
    pca_variance_threshold: float = PCA_VARIANCE_THRESHOLD,
    pca_max_components: int = PCA_MAX_COMPONENTS,
    min_topic_presence: int = MIN_TOPIC_PRESENCE,
):
    """Run a single mediation analysis case and save results."""
    treatment = resolve_topic_name(df, treatment)
    mediator  = resolve_topic_name(df, mediator)
    dir_name = make_dir_name(treatment, mediator, n_bootstrap)
    output_dir = os.path.join(output_base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, f"{dir_name}.log")
    log_file = open(log_path, 'w')
    tee_cls = setup_logger()
    original_stdout = sys.stdout
    sys.stdout = tee_cls(original_stdout, log_file)

    try:
        start_time = time.perf_counter()

        print("=" * 80)
        print(f"MEDIATION ANALYSIS — Treatment: {treatment}, Mediator: {mediator}")
        print(f"Bootstrap samples: {n_bootstrap}")
        print("=" * 80)

        print(f"\nDataset: {len(df)} articles")

        key_mediators = [f'sentiment {mediator}']
        print(f"\nMediators: {key_mediators}")

        # ------------------------------------------------------------------ #
        # Primary analysis: continuous (raw score) treatment with             #
        # percentile-based contrasts (Q25→Q75 and P10→P90).                   #
        # ------------------------------------------------------------------ #
        config = TreatmentConfig(topic_name=treatment, min_samples=50)

        results = run_stratified_analysis(
            df,
            config,
            include_other_sentiments=True,
            include_topic_indicators=True,
            estimate_mediation=True,
            mediator_subset=key_mediators,
            verbose=True,
            random_state=42,
            n_bootstrap=n_bootstrap,
            apply_pca_to_confounders=apply_pca_to_confounders,
            pca_variance_threshold=pca_variance_threshold,
            pca_max_components=pca_max_components,
            min_topic_presence=min_topic_presence,
        )

        print_mediation_summary(results, treatment)

        # Export primary results
        print("\n" + "=" * 80)
        print("EXPORTING RESULTS")
        print("=" * 80)

        all_results = {treatment: results}
        export_mediation_results(all_results, output_dir=output_dir)

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)

        # Continuous-treatment contrasts are keyed by percentile labels
        # (e.g. "Q25 → Q75 (−23.0 → 45.0)").  Use the first available key.
        primary_key = 'primary contrast'   # overwritten per topic below

        summary_data = []
        for topic_name, res in all_results.items():
            mediation_results = res['overall'].get('mediation_results', {}) or {}

            key = next(iter(mediation_results), None)
            if key is None:
                continue
            primary_key = key
            result = mediation_results.get(key)
            if result is not None:
                prop_reliable = result.get('prop_mediated_reliable', True)
                summary_data.append({
                    'Topic': topic_name,
                    'Contrast': key,
                    'TE': f"{result['te']:.3f}",
                    'TE_CI': f"[{result['te_ci'][0]:.3f}, {result['te_ci'][1]:.3f}]",
                    'NDE': f"{result['nde']:.3f}",
                    'NIE': f"{result['nie']:.3f}",
                    'Prop_Mediated': f"{result['prop_mediated']:.1%}" if prop_reliable else "N/A",
                    'N': result['n_samples']
                })

        summary_df = pd.DataFrame(summary_data)
        print(f"\nMain Results ({primary_key} Contrast):")
        print(summary_df.to_string(index=False))

        summary_path = os.path.join(output_dir, 'mediation_summary_table.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary table saved to: {summary_path}")

        # CATE heterogeneity by outlet ideology
        print("\n" + "=" * 80)
        print("CATE HETEROGENEITY BY OUTLET IDEOLOGY")
        print("=" * 80)

        cate_heterogeneity = results['overall'].get('cate_heterogeneity', {})
        if cate_heterogeneity:
            het_rows = []
            for label in ['Left', 'Center', 'Right']:
                r = cate_heterogeneity.get(label, {})
                if not r or r.get('note'):
                    continue
                row: dict = {
                    'Treatment': treatment,
                    'Ideology': label,
                    'N': r['n'],
                    'Mean_CATE': round(r['mean_cate'], 4),
                    'Std_CATE': round(r['std_cate'], 4),
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
                print(f"\nCATE heterogeneity summary saved to: {het_path}")
            else:
                print("\nNo valid ideology groups in CATE heterogeneity results.")
        else:
            print("\nNo CATE heterogeneity results available.")

        # ------------------------------------------------------------------ #
        # Visualisations                                                      #
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)

        try:
            figs = plot_mediation_results(results, treatment, save_dir=output_dir)
            print(f"Generated {len(figs)} plot(s).")
        except Exception as e:
            print(f"\nWarning: Plot generation failed: {e}")

        elapsed = (time.perf_counter() - start_time) / 60
        print(f"\nThis case took {elapsed:.4f} minutes.")
        print("=" * 80 + "\n")

    except InsufficientSamplesError as e:
        print(f"\nSKIPPED: insufficient articles for treatment topic.")
        print(f"  {e}")
        print("=" * 80 + "\n")
        return

    except InsufficientTopicPresenceError as e:
        print(f"\nSKIPPED: min_topic_presence filter removed the required mediator.")
        print(f"  {e}")
        print("=" * 80 + "\n")
        return

    finally:
        sys.stdout = original_stdout
        log_file.close()


def main():
    args = build_parser().parse_args()

    # Load communities
    communities = None
    community_cases = []
    if args.communities:
        with open(args.communities) as f:
            communities = json.load(f)
        if not args.community_cases:
            print("Available communities:")
            for cid, topics in communities.items():
                preview = ', '.join(topics[:3])
                suffix = '...' if len(topics) > 3 else ''
                print(f"  {cid}: {len(topics)} topics — {preview}{suffix}")
            print("\nSpecify pairs with --community-cases T_ID:M_ID, "
                  "e.g.: --community-cases 11:1 1:11")
            return
        community_cases = parse_community_cases(args.community_cases)

    bootstrap_list = [args.n_bootstrap] if args.n_bootstrap else BOOTSTRAP_SAMPLES

    input_files = args.input_files or sorted(
        str(p) for p in Path('data').glob('*.pkl')
    )
    if not input_files:
        print("\nNo .pkl files found in data/  — nothing to do.")
        return

    total_start = time.perf_counter()

    print("=" * 80)
    print("MEDIATION ANALYSIS FOR THESIS — BATCH RUN")
    print("=" * 80)
    print(f"\nFound {len(input_files)} dataset(s):")
    for p in input_files:
        print(f"  {p}")

    for pkl_path in input_files:
        stem = Path(pkl_path).stem
        output_base_dir = os.path.join(args.output_root, stem)

        print(f"\n{'=' * 80}")
        print(f"DATASET: {pkl_path}")
        print(f"Output root: {output_base_dir}")
        print(f"{'=' * 80}")

        print("\nLoading data...")
        df = pd.read_pickle(str(pkl_path))
        print(f"Dataset: {len(df)} articles")

        # Existing hardcoded topic-level cases
        if not args.communities_only:
            for case in CASES:
                for n_bootstrap in bootstrap_list:
                    treatment = case['treatment']
                    mediator = case['mediator']
                    dir_name = make_dir_name(treatment, mediator, n_bootstrap)
                    print(f"\nRunning: {stem}/{dir_name}")
                    run_start = time.perf_counter()
                    run_single_case(df, treatment, mediator, n_bootstrap,
                                    output_base_dir=output_base_dir,
                                    min_topic_presence=args.min_topic_presence,
                                    pca_variance_threshold=args.pca_variance)
                    run_elapsed = (time.perf_counter() - run_start) / 60
                    print(f"  Done: {output_base_dir}/{dir_name}/ "
                          f"({run_elapsed:.4f} minutes)")

        # Community-level cases
        if community_cases:
            print(f"\nBuilding community dataframe ({len(communities)} communities)...")
            community_df = build_community_df(df, communities)
            print(f"  Community columns: "
                  f"{sum(1 for c in community_df.columns if c.startswith('topic '))} topics")
            for (treatment, mediator) in community_cases:
                for n_bootstrap in bootstrap_list:
                    dir_name = make_dir_name(treatment, mediator, n_bootstrap)
                    print(f"\nRunning: {stem}/{dir_name}")
                    run_start = time.perf_counter()
                    run_single_case(community_df, treatment, mediator, n_bootstrap,
                                    output_base_dir=output_base_dir,
                                    min_topic_presence=args.min_topic_presence,
                                    pca_variance_threshold=args.pca_variance)
                    run_elapsed = (time.perf_counter() - run_start) / 60
                    print(f"  Done: {output_base_dir}/{dir_name}/ "
                          f"({run_elapsed:.4f} minutes)")

    total_elapsed = (time.perf_counter() - total_start) / 60
    print(f"\nAll runs completed in {total_elapsed:.4f} minutes.")
    print("=" * 80)


if __name__ == '__main__':
    main()
