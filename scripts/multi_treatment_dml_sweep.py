#!/usr/bin/env python3
"""
Multi-Treatment DML Sweep

Estimates the causal effect of EVERY topic sentiment on ideology simultaneously,
using LinearDML with a multivariate treatment matrix.  Unlike the single-topic
mediation analysis, this script uses the full dataset (N ~ 11 k articles) and
produces one ATE estimate per topic sentiment.

Treatment encoding: raw sentiment score, standardised to unit variance.
ATE = "effect of 1 SD increase in sentiment on ideology score."

Usage
-----
    # all .pkl files in data/ (default — no args needed)
    python examples/multi_treatment_dml_sweep.py

    # explicit file(s)
    python examples/multi_treatment_dml_sweep.py data/allsides_df_test.pkl
    python examples/multi_treatment_dml_sweep.py data/file_a.pkl data/file_b.pkl

    # custom settings
    python examples/multi_treatment_dml_sweep.py \\
        --output-root results/mt_sweep \\
        --n-bootstrap 500 \\
        --min-presence 30

Optional flags
--------------
  --output-root PATH    Root directory for results (default: results/multi_treatment)
  --min-presence N      Min articles per topic to include (default: 50)
  --max-topics N        Max topics to analyse, by article count (default: 10; 0 = all)
  --n-bootstrap N       Bootstrap iterations for CIs (default: 500)
  --no-pca              Skip PCA on confounders (topic presence matrix)
  --random-state N      Random seed (default: 42)

Output per input file  (under <output-root>/<input-stem>/)
----------------------------------------------------------
  ate_per_sentiment.csv   ATE table
  forest_plot.png         Forest plot
  run_log.txt             Full console output
"""

import sys
import argparse
import json
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.ate_cate_analysis import run_multi_treatment_dml
from utils.community_utils import build_community_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_forest(ate_table: pd.DataFrame, title: str, save_path: str,
                x_label: str = 'ATE') -> None:
    """Horizontal forest plot sorted by |ATE|, top-40 topics."""
    df = ate_table.copy().reset_index(drop=True)
    if len(df) > 40:
        df = df.head(40)

    n = len(df)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * n + 2)))

    y = np.arange(n)
    colors = ['#1f4e79' if sig else '#888888' for sig in df['significant']]

    ax.barh(y, df['ate'], color=colors, alpha=0.75, height=0.6)
    ax.errorbar(
        df['ate'], y,
        xerr=[df['ate'] - df['ci_lower'], df['ci_upper'] - df['ate']],
        fmt='none', color='black', capsize=3, linewidth=1.2,
    )
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_yticks(y)
    ax.set_yticklabels(df['topic'], fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#1f4e79', alpha=0.75, label='Significant (95% CI excl. 0)'),
        Patch(facecolor='#888888', alpha=0.75, label='Not significant'),
    ], loc='lower right', fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Forest plot → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helper (returns the results dict)
# ─────────────────────────────────────────────────────────────────────────────

def _run_dml(
    df: pd.DataFrame,
    output_dir: str,
    stem: str,
    min_presence: int,
    n_bootstrap: int,
    use_pca: bool,
    random_state: int,
    verbose: bool,
    n_jobs: int = 1,
    max_topics: int | None = None,
) -> dict:
    """Run multi-treatment DML and save CSV + forest plot to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'─' * 60}")
    print(f"Output → {output_dir}")
    print(f"{'─' * 60}")

    t0 = time.perf_counter()

    results = run_multi_treatment_dml(
        df,
        min_topic_presence=min_presence,
        apply_pca_to_confounders=use_pca,
        normalize_treatment=True,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
        max_topics=max_topics,
    )

    ate_table = results['ate_table']
    n_sig = int(ate_table['significant'].sum())

    print(f"\n  Topics analysed : {results['n_treatments']}")
    print(f"  N articles used : {results['n_samples']:,}")
    print(f"  Significant ATEs: {n_sig} / {results['n_treatments']}")

    # Top-20 summary table
    disp = ['topic', 'ate', 'ci_lower', 'ci_upper', 'significant', 'n_topic_articles']
    top = ate_table.head(20)[disp].copy()
    for col in ('ate', 'ci_lower', 'ci_upper'):
        top[col] = top[col].map('{:+.4f}'.format)
    print(f"\n  Top 20 by |ATE|:")
    print(top.to_string(index=False))

    if n_sig > 0:
        print(f"\n  Significant topics ({n_sig}):")
        sig = ate_table[ate_table['significant']][disp].copy()
        for col in ('ate', 'ci_lower', 'ci_upper'):
            sig[col] = sig[col].map('{:+.4f}'.format)
        print(sig.to_string(index=False))
    else:
        print("\n  No significant ATEs at the 95% level.")

    # Save CSV
    csv_path = os.path.join(output_dir, 'ate_per_sentiment.csv')
    ate_table.to_csv(csv_path, index=False)
    print(f"\n  ATE table → {csv_path}")

    # Forest plot
    x_label = 'ATE (effect of 1 SD increase in sentiment on ideology)'
    plot_title = (
        f"Multi-Treatment DML — {stem}\n"
        f"Continuous (raw score, standardised to 1 SD unit)  |  "
        f"N={results['n_samples']:,}  |  "
        f"{n_sig}/{results['n_treatments']} significant"
    )
    plot_forest(
        ate_table,
        title=plot_title,
        save_path=os.path.join(output_dir, 'forest_plot.png'),
        x_label=x_label,
    )

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n  Run completed in {elapsed:.2f} min.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-file orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_sweep_for_file(
    input_path: str,
    output_root: str,
    min_presence: int,
    n_bootstrap: int,
    use_pca: bool,
    random_state: int = 42,
    verbose: bool = True,
    communities: dict | None = None,
    communities_only: bool = False,
    n_jobs: int = 1,
    max_topics: int | None = None,
) -> None:
    stem = Path(input_path).stem
    output_dir = os.path.join(output_root, stem)
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'run_log.txt')
    log_file = open(log_path, 'w')

    class _Tee:
        def __init__(self, orig, fobj):
            self.orig = orig
            self.fobj = fobj

        def write(self, msg):
            self.orig.write(msg)
            self.fobj.write(msg)

        def flush(self):
            self.orig.flush()
            self.fobj.flush()

    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    try:
        t_file = time.perf_counter()

        print("=" * 80)
        print("MULTI-TREATMENT DML SWEEP")
        print(f"Input : {input_path}")
        print(f"Output: {output_dir}")
        print("=" * 80)

        # ── load ───────────────────────────────────────────────────────────
        print(f"\nLoading {input_path} …")
        if input_path.endswith('.pkl'):
            df = pd.read_pickle(input_path)
        elif input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported format: {input_path}. Use .pkl or .csv")
        print(f"  {len(df):,} rows × {len(df.columns)} columns")

        n_topic = sum(1 for c in df.columns if c.startswith('topic '))
        n_sent  = sum(1 for c in df.columns if c.startswith('sentiment '))
        print(f"  Topic columns   : {n_topic}")
        print(f"  Sentiment columns: {n_sent}")
        if 'int_bias' not in df.columns:
            raise ValueError("Dataset must have an 'int_bias' outcome column.")

        # ── settings ──────────────────────────────────────────────────────
        print(f"\nSettings:")
        print(f"  min_topic_presence: {min_presence}")
        print(f"  n_bootstrap       : {n_bootstrap}")
        print(f"  apply_pca         : {use_pca}")
        print(f"  random_state      : {random_state}")
        print(f"  max_topics        : {max_topics or 'all'}")

        # ── topic-level run ───────────────────────────────────────────────
        if not communities_only:
            _run_dml(
                df=df,
                output_dir=output_dir,
                stem=stem,
                min_presence=min_presence,
                n_bootstrap=n_bootstrap,
                use_pca=use_pca,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs,
                max_topics=max_topics,
            )

        # ── community-level analysis (optional) ───────────────────────────
        if communities is not None:
            print(f"\n{'─' * 60}")
            print("COMMUNITY-LEVEL ANALYSIS")
            print(f"  Building community dataframe ({len(communities)} communities) …")
            community_df = build_community_df(df, communities)
            n_comm = sum(1 for c in community_df.columns if c.startswith('topic '))
            print(f"  Community topic columns: {n_comm}")
            comm_output_dir = os.path.join(output_dir, 'community')
            os.makedirs(comm_output_dir, exist_ok=True)

            _run_dml(
                df=community_df,
                output_dir=comm_output_dir,
                stem=f'{stem} [communities]',
                min_presence=min_presence,
                n_bootstrap=n_bootstrap,
                use_pca=use_pca,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs,
            )

        elapsed = (time.perf_counter() - t_file) / 60
        print(f"\nFile completed in {elapsed:.2f} min. Log → {log_path}")
        print("=" * 80)

    finally:
        sys.stdout = original_stdout
        log_file.close()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        'input_files', nargs='*',
        help='One or more .pkl or .csv dataset files to process. '
             'Defaults to all *.pkl files found in data/.',
    )
    p.add_argument(
        '--output-root', default='results/multi_treatment',
        help='Root directory for results. Default: results/multi_treatment',
    )
    p.add_argument(
        '--min-presence', type=int, default=50,
        help='Min articles per topic to include its sentiment (default: 50).',
    )
    p.add_argument(
        '--n-bootstrap', type=int, default=2000,
        help='Bootstrap iterations for CIs (default: 500).',
    )
    p.add_argument(
        '--no-pca', action='store_true',
        help='Skip PCA on confounders (topic presence matrix).',
    )
    p.add_argument(
        '--max-topics', type=int, default=10,
        help='Max number of topics to analyse (by article count). '
             'Only applies to topic-level analysis, not communities. '
             'Use 0 for no limit (default: 10).',
    )
    p.add_argument(
        '--random-state', type=int, default=42,
        help='Random seed (default: 42).',
    )
    p.add_argument(
        '--n-jobs', type=int, default=1,
        help='Parallel workers for the bootstrap loop and the main-fit '
             'random forests (default: 1). Use -1 for all cores.',
    )
    p.add_argument(
        '--communities', default=None, metavar='PATH',
        help='Path to communities JSON (e.g. communities.json). '
             'If provided, runs community-level analysis: each community '
             'is one treatment (presence = any topic present, '
             'sentiment = mean of present topics). '
             'Output saved under <output_dir>/community/.',
    )
    p.add_argument(
        '--communities-only', action='store_true',
        help='Skip topic-level DML; run only the community-level analysis. '
             'Requires --communities.',
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.communities_only and not args.communities:
        parser.error("--communities-only requires --communities PATH")

    communities = None
    if args.communities:
        with open(args.communities) as f:
            communities = json.load(f)
        print(f"Loaded {len(communities)} communities from {args.communities}")

    # Default to all .pkl files in data/ when none are specified
    input_files = args.input_files
    if not input_files:
        input_files = sorted(str(p) for p in Path('data').glob('*.pkl'))
        if not input_files:
            print("No .pkl files found in data/  — pass explicit file paths or add files to data/")
            return
        print(f"No input files specified — found {len(input_files)} in data/:")
        for f in input_files:
            print(f"  {f}")

    total_start = time.perf_counter()
    n_files = len(input_files)

    print("=" * 80)
    print("MULTI-TREATMENT DML SWEEP — BATCH RUN")
    print(f"Files      : {n_files}")
    print(f"Bootstrap  : {args.n_bootstrap}")
    print(f"Output root: {args.output_root}")
    print("=" * 80)

    for i, path in enumerate(input_files, 1):
        print(f"\n[{i}/{n_files}] {path}")
        try:
            run_sweep_for_file(
                input_path=path,
                output_root=args.output_root,
                min_presence=args.min_presence,
                n_bootstrap=args.n_bootstrap,
                use_pca=not args.no_pca,
                random_state=args.random_state,
                communities=communities,
                communities_only=args.communities_only,
                n_jobs=args.n_jobs,
                max_topics=args.max_topics or None,
            )
        except Exception as e:
            print(f"  ERROR processing {path}: {e}")
            import traceback
            traceback.print_exc()

    elapsed = (time.perf_counter() - total_start) / 60
    print(f"\nAll {n_files} file(s) completed in {elapsed:.2f} min.")
    print("=" * 80)


if __name__ == '__main__':
    main()
