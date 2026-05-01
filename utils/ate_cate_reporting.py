"""
Results reporting and visualization for ATE/CATE analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings


def format_ate_table(results: Dict, topic_name: str) -> pd.DataFrame:
    """
    Format ATE results into a clean table.

    Args:
        results: Results from run_stratified_analysis
        topic_name: Name of the treatment topic

    Returns:
        Formatted DataFrame
    """
    # Overall results
    overall_df = results['overall']['ate_results'].copy()
    overall_df['stratum'] = 'Overall'

    # Stratified results (by ideology)
    ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}
    stratified_dfs = []

    for ideology_val, strat_results in results['stratified'].items():
        if strat_results is not None:
            strat_df = strat_results['ate_results'].copy()
            strat_df['stratum'] = f"{ideology_labels[ideology_val]} ideology"
            stratified_dfs.append(strat_df)

    # Combine
    if stratified_dfs:
        all_results = pd.concat([overall_df] + stratified_dfs, ignore_index=True)
    else:
        all_results = overall_df

    # Reorder columns
    cols = ['stratum', 'method', 'ate', 'ate_lower', 'ate_upper', 'n_samples', 'note']
    all_results = all_results[[c for c in cols if c in all_results.columns]]

    # Add topic name
    all_results.insert(0, 'topic', topic_name)

    return all_results


def print_ate_summary(results: Dict, topic_name: str):
    """
    Print a formatted summary of ATE results.

    Args:
        results: Results from run_stratified_analysis
        topic_name: Name of the treatment topic
    """
    print(f"\n{'='*80}")
    print(f"ATE ANALYSIS SUMMARY: {topic_name}")
    print('='*80)

    ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}

    # Overall
    print(f"\n{'-'*80}")
    print("OVERALL (All ideology groups)")
    print('-'*80)
    overall_df = results['overall']['ate_results']
    print(overall_df.to_string(index=False))

    # Stratified by ideology
    for ideology_val in [0, 1, 2]:
        strat_results = results['stratified'].get(ideology_val)
        if strat_results is not None:
            print(f"\n{'-'*80}")
            print(f"{ideology_labels[ideology_val].upper()} IDEOLOGY")
            print(f"N = {strat_results['n_samples']}")
            print('-'*80)
            print(strat_results['ate_results'].to_string(index=False))
        else:
            print(f"\n{'-'*80}")
            print(f"{ideology_labels[ideology_val].upper()} IDEOLOGY")
            print('-'*80)
            print("Insufficient data for analysis")

    # CATE summary
    print(f"\n{'-'*80}")
    print("CONDITIONAL AVERAGE TREATMENT EFFECT (CATE)")
    print('-'*80)
    cate_results = results['overall']['cate_results']
    print(f"Method: {cate_results['method']}")
    print(f"ATE (from CATE model): {cate_results['ate']:.4f}")
    if cate_results['ate_lower'] is not None:
        print(f"95% CI: [{cate_results['ate_lower']:.4f}, {cate_results['ate_upper']:.4f}]")
    print(f"\nIndividual treatment effect distribution:")
    cate = cate_results['cate']
    print(f"  Mean:   {np.mean(cate):7.4f}")
    print(f"  Std:    {np.std(cate):7.4f}")
    print(f"  Min:    {np.min(cate):7.4f}")
    print(f"  25%:    {np.percentile(cate, 25):7.4f}")
    print(f"  Median: {np.median(cate):7.4f}")
    print(f"  75%:    {np.percentile(cate, 75):7.4f}")
    print(f"  Max:    {np.max(cate):7.4f}")


def plot_ate_comparison(results_dict: Dict[str, Dict], save_path: Optional[str] = None):
    """
    Plot ATE comparison across topics and methods.

    Args:
        results_dict: Dictionary mapping topic names to results
        save_path: Optional path to save figure
    """
    # Collect all results
    all_results = []
    for topic_name, results in results_dict.items():
        df = format_ate_table(results, topic_name)
        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Filter to overall results only for main comparison
    overall_df = combined_df[combined_df['stratum'] == 'Overall'].copy()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: ATE by method and topic
    ax1 = axes[0]
    topics = overall_df['topic'].unique()
    methods = overall_df['method'].unique()
    x = np.arange(len(topics))
    width = 0.2

    for i, method in enumerate(methods):
        method_data = overall_df[overall_df['method'] == method]
        ates = [method_data[method_data['topic'] == topic]['ate'].values[0]
                if len(method_data[method_data['topic'] == topic]) > 0 else 0
                for topic in topics]
        ax1.bar(x + i * width, ates, width, label=method)

    ax1.set_xlabel('Topic')
    ax1.set_ylabel('Average Treatment Effect (ATE)')
    ax1.set_title('ATE Comparison Across Topics and Methods')
    ax1.set_xticks(x + width * (len(methods) - 1) / 2)
    ax1.set_xticklabels(topics, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: ATE by ideology (stratified)
    ax2 = axes[1]
    stratified_df = combined_df[combined_df['stratum'] != 'Overall'].copy()

    if len(stratified_df) > 0:
        # Use DML method for consistency
        dml_df = stratified_df[stratified_df['method'] == 'DML']

        if len(dml_df) > 0:
            pivot_df = dml_df.pivot_table(
                index='stratum',
                columns='topic',
                values='ate',
                aggfunc='mean'
            )

            ideology_order = ['Left ideology', 'Center ideology', 'Right ideology']
            pivot_df = pivot_df.reindex(ideology_order)

            pivot_df.plot(kind='bar', ax=ax2)
            ax2.set_xlabel('Baseline Ideology')
            ax2.set_ylabel('Average Treatment Effect (ATE)')
            ax2.set_title('ATE by Baseline Ideology (DML Method)')
            ax2.legend(title='Topic')
            ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        else:
            ax2.text(0.5, 0.5, 'No stratified DML results available',
                    ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'No stratified results available',
                ha='center', va='center', transform=ax2.transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


def plot_cate_distribution(results: Dict, topic_name: str, save_path: Optional[str] = None):
    """
    Plot CATE (heterogeneous treatment effects) distribution.

    Args:
        results: Results from run_stratified_analysis
        topic_name: Name of the treatment topic
        save_path: Optional path to save figure
    """
    cate_results = results['overall']['cate_results']
    cate = cate_results['cate']
    data = results['overall']['data']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: CATE distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(cate, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(np.mean(cate), color='red', linestyle='--', label=f'Mean: {np.mean(cate):.3f}')
    ax1.axvline(np.median(cate), color='blue', linestyle='--', label=f'Median: {np.median(cate):.3f}')
    ax1.set_xlabel('Individual Treatment Effect')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'CATE Distribution: {topic_name}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Treatment (sentiment) value distribution
    ax2 = axes[0, 1]
    sentiment_continuous_full = data['T'].ravel()
    ax2.hist(sentiment_continuous_full, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Sentiment Value (Treatment)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Treatment Distribution')
    ax2.grid(alpha=0.3)

    # Plot 3: CATE by ideology
    ax3 = axes[1, 0]
    ideology = data['Y']
    ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}

    cate_by_ideology = {
        label: cate[ideology == val]
        for val, label in ideology_labels.items()
    }

    box_data = [cate_by_ideology[label] for label in ['Left', 'Center', 'Right']
                if len(cate_by_ideology[label]) > 0]
    box_labels = [label for label in ['Left', 'Center', 'Right']
                  if len(cate_by_ideology[label]) > 0]

    ax3.boxplot(box_data, labels=box_labels)
    ax3.set_xlabel('Ideology')
    ax3.set_ylabel('Individual Treatment Effect')
    ax3.set_title('CATE by Ideology')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Scatter plot CATE vs sentiment value
    ax4 = axes[1, 1]
    sentiment_continuous = data['T'].ravel()
    ax4.scatter(sentiment_continuous, cate, alpha=0.4, s=20)
    ax4.set_xlabel('Sentiment Value (Treatment)')
    ax4.set_ylabel('Individual Treatment Effect (CATE)')
    ax4.set_title('CATE vs Sentiment Value')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    return fig


def export_results_to_csv(results_dict: Dict[str, Dict], output_dir: str = '.'):
    """
    Export all results to CSV files.

    Args:
        results_dict: Dictionary mapping topic names to results
        output_dir: Directory to save CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Combine all ATE results
    all_ate_results = []
    for topic_name, results in results_dict.items():
        df = format_ate_table(results, topic_name)
        all_ate_results.append(df)

    combined_ate = pd.concat(all_ate_results, ignore_index=True)

    # Save ATE results
    ate_path = os.path.join(output_dir, 'ate_results.csv')
    combined_ate.to_csv(ate_path, index=False)
    print(f"ATE results saved to: {ate_path}")

    # Save CATE summaries
    cate_summaries = []
    for topic_name, results in results_dict.items():
        cate_results = results['overall']['cate_results']
        cate = cate_results['cate']

        summary = {
            'topic': topic_name,
            'method': cate_results['method'],
            'ate': cate_results['ate'],
            'ate_lower': cate_results.get('ate_lower'),
            'ate_upper': cate_results.get('ate_upper'),
            'cate_mean': np.mean(cate),
            'cate_std': np.std(cate),
            'cate_min': np.min(cate),
            'cate_25pct': np.percentile(cate, 25),
            'cate_median': np.median(cate),
            'cate_75pct': np.percentile(cate, 75),
            'cate_max': np.max(cate),
            'n_samples': cate_results['n_samples']
        }
        cate_summaries.append(summary)

    cate_df = pd.DataFrame(cate_summaries)
    cate_path = os.path.join(output_dir, 'cate_summary.csv')
    cate_df.to_csv(cate_path, index=False)
    print(f"CATE summary saved to: {cate_path}")

    return {
        'ate_results': ate_path,
        'cate_summary': cate_path
    }


def compare_topics(results_dict: Dict[str, Dict]):
    """
    Print a comparison of ATE across topics.

    Args:
        results_dict: Dictionary mapping topic names to results
    """
    print(f"\n{'='*80}")
    print("CROSS-TOPIC COMPARISON")
    print('='*80)

    # Collect DML results (most robust)
    comparison_data = []

    for topic_name, results in results_dict.items():
        overall_df = results['overall']['ate_results']
        dml_row = overall_df[overall_df['method'] == 'DML']

        if len(dml_row) > 0:
            comparison_data.append({
                'Topic': topic_name,
                'ATE (DML)': dml_row['ate'].values[0],
                'Lower 95%': dml_row['ate_lower'].values[0] if dml_row['ate_lower'].notna().values[0] else None,
                'Upper 95%': dml_row['ate_upper'].values[0] if dml_row['ate_upper'].notna().values[0] else None,
                'N': dml_row['n_samples'].values[0]
            })

        # Add stratified results (by ideology)
        ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}
        for ideology_val, strat_results in results['stratified'].items():
            if strat_results is not None:
                strat_df = strat_results['ate_results']
                dml_row = strat_df[strat_df['method'] == 'DML']

                if len(dml_row) > 0:
                    comparison_data.append({
                        'Topic': f"{topic_name} ({ideology_labels[ideology_val]} ideology)",
                        'ATE (DML)': dml_row['ate'].values[0],
                        'Lower 95%': dml_row['ate_lower'].values[0] if dml_row['ate_lower'].notna().values[0] else None,
                        'Upper 95%': dml_row['ate_upper'].values[0] if dml_row['ate_upper'].notna().values[0] else None,
                        'N': dml_row['n_samples'].values[0]
                    })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print('='*80)
    print("- ATE represents the average effect of a 1-unit change in sentiment on ideology")
    print("- Positive ATE: Higher sentiment → Higher ideology value (more Right-leaning)")
    print("- Negative ATE: Higher sentiment → Lower ideology value (more Left-leaning)")
    print("- Ideology scale: 0=Left, 1=Center, 2=Right")
    print('='*80)


def examine_cate_by_ideology(results: Dict, topic_name: str, save_path: Optional[str] = None):
    """
    Examine CATE heterogeneity by baseline ideology.

    This is the proper alternative to stratification for examining how treatment
    effects differ across ideology groups. Unlike stratification (which requires
    outcome variation within each stratum), CATE analysis estimates individual
    treatment effects and then examines their distribution across subgroups.

    Args:
        results: Results from run_stratified_analysis
        topic_name: Name of the treatment topic
        save_path: Optional path to save figure

    Returns:
        DataFrame with CATE statistics by ideology
    """
    print(f"\n{'='*80}")
    print(f"CATE HETEROGENEITY BY BASELINE IDEOLOGY: {topic_name}")
    print('='*80)

    cate_results = results['overall']['cate_results']
    data = results['overall']['data']

    if cate_results is None:
        print("CATE results not available (no covariates for CATE estimation)")
        return None

    cate = cate_results['cate']
    ideology = data['Y']

    ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}

    # Calculate CATE statistics by ideology group
    stats_by_ideology = []

    for ideology_val in [0, 1, 2]:
        mask = ideology == ideology_val
        cate_group = cate[mask]

        if len(cate_group) > 0:
            stats = {
                'Ideology': ideology_labels[ideology_val],
                'N': len(cate_group),
                'Mean CATE': np.mean(cate_group),
                'Median CATE': np.median(cate_group),
                'Std CATE': np.std(cate_group),
                'Min CATE': np.min(cate_group),
                'Max CATE': np.max(cate_group),
                '25th percentile': np.percentile(cate_group, 25),
                '75th percentile': np.percentile(cate_group, 75)
            }
            stats_by_ideology.append(stats)

    stats_df = pd.DataFrame(stats_by_ideology)

    # Print summary
    print("\nCATEs (Individual Treatment Effects) by Baseline Ideology:")
    print(stats_df.to_string(index=False))

    # Statistical comparison
    print(f"\n{'-'*80}")
    print("STATISTICAL COMPARISON:")
    print('-'*80)

    # Compare Left vs Right
    left_cate = cate[ideology == 0]
    right_cate = cate[ideology == 2]

    if len(left_cate) > 0 and len(right_cate) > 0:
        from scipy import stats as scipy_stats

        # Two-sample t-test
        t_stat, p_value = scipy_stats.ttest_ind(left_cate, right_cate)

        print(f"\nLeft vs Right ideology:")
        print(f"  Mean CATE difference: {np.mean(left_cate) - np.mean(right_cate):.4f}")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")

        if p_value < 0.05:
            print(f"  Result: Significant difference (p < 0.05)")
            if np.mean(left_cate) > np.mean(right_cate):
                print(f"  Interpretation: Left-leaning outlets show STRONGER treatment effects")
            else:
                print(f"  Interpretation: Right-leaning outlets show STRONGER treatment effects")
        else:
            print(f"  Result: No significant difference (p >= 0.05)")
            print(f"  Interpretation: Treatment effects are similar across ideologies")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Box plots
    ax1 = axes[0]
    box_data = [cate[ideology == val] for val in [0, 1, 2] if len(cate[ideology == val]) > 0]
    box_labels = [ideology_labels[val] for val in [0, 1, 2] if len(cate[ideology == val]) > 0]

    ax1.boxplot(box_data, labels=box_labels)
    ax1.set_xlabel('Baseline Ideology')
    ax1.set_ylabel('Individual Treatment Effect (CATE)')
    ax1.set_title(f'CATE Distribution by Ideology: {topic_name}')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Violin plots
    ax2 = axes[1]
    positions = []
    violin_data = []
    violin_labels = []

    for i, val in enumerate([0, 1, 2]):
        if len(cate[ideology == val]) > 0:
            positions.append(i)
            violin_data.append(cate[ideology == val])
            violin_labels.append(ideology_labels[val])

    parts = ax2.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(violin_labels)
    ax2.set_xlabel('Baseline Ideology')
    ax2.set_ylabel('Individual Treatment Effect (CATE)')
    ax2.set_title(f'CATE Density by Ideology: {topic_name}')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    print(f"\n{'='*80}")
    print("INTERPRETATION GUIDE:")
    print('='*80)
    print("- CATE represents the estimated treatment effect for each individual article")
    print("- Mean CATE by ideology shows whether treatment effects differ across groups")
    print("- Positive CATE: Positive sentiment → shift toward Right ideology")
    print("- Negative CATE: Positive sentiment → shift toward Left ideology")
    print("- Large differences suggest heterogeneous treatment effects by ideology")
    print('='*80)

    return stats_df, fig


def print_mediation_summary(results: Dict, topic_name: str):
    """
    Print formatted summary of mediation analysis results.

    Args:
        results: Results from run_stratified_analysis with estimate_mediation=True
        topic_name: Name of the treatment topic
    """
    print(f"\n{'='*80}")
    print(f"MEDIATION ANALYSIS: {topic_name}")
    print('='*80)

    mediation_results = results['overall'].get('mediation_results')

    if mediation_results is None or len(mediation_results) == 0:
        print("\nNo mediation results available.")
        print("Ensure estimate_mediation=True and mediators are included in analysis.")
        return

    # Print results for each contrast
    for contrast_label, result in mediation_results.items():
        if result is None:
            print(f"\n{contrast_label}: Analysis failed")
            continue

        print(f"\n{'-'*80}")
        print(f"{contrast_label}")
        print(f"{'-'*80}")

        te = result['te']
        nde = result['nde']
        nie = result['nie']
        te_ci = result['te_ci']
        nde_ci = result['nde_ci']
        nie_ci = result['nie_ci']
        prop_mediated = result['prop_mediated']
        prop_reliable = result.get('prop_mediated_reliable', True)

        # Determine significance
        te_sig = '**' if (te_ci[0] > 0 and te_ci[1] > 0) or (te_ci[0] < 0 and te_ci[1] < 0) else ''
        nde_sig = '**' if (nde_ci[0] > 0 and nde_ci[1] > 0) or (nde_ci[0] < 0 and nde_ci[1] < 0) else ''
        nie_sig = '**' if (nie_ci[0] > 0 and nie_ci[1] > 0) or (nie_ci[0] < 0 and nie_ci[1] < 0) else ''

        print(f"  Total Effect (TE):           {te:7.4f} [{te_ci[0]:7.4f}, {te_ci[1]:7.4f}] {te_sig}")
        print(f"  Natural Direct Effect (NDE): {nde:7.4f} [{nde_ci[0]:7.4f}, {nde_ci[1]:7.4f}] {nde_sig}")
        print(f"  Natural Indirect Effect (NIE): {nie:7.4f} [{nie_ci[0]:7.4f}, {nie_ci[1]:7.4f}] {nie_sig}")
        if prop_reliable:
            print(f"  Proportion Mediated:         {prop_mediated:6.1%}")
        else:
            print(f"  Proportion Mediated:         N/A (TE \u2248 0 or CI includes zero)")
        print(f"  Number of Mediators:         {result['n_mediators']}")
        print(f"  Decomposition Check:         |TE - (NDE + NIE)| = {result['decomposition_error']:.4f}")

    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print('='*80)
    print("- TE = Total effect of sentiment on ideology")
    print("- NDE = Direct effect (NOT through other sentiments)")
    print("- NIE = Indirect effect (THROUGH other sentiments)")
    print("- Proportion Mediated = NIE / TE")
    print("- ** = 95% CI excludes zero (statistically significant)")
    print('='*80)


def format_mediation_table(results: Dict, topic_name: str) -> pd.DataFrame:
    """
    Format mediation results into a DataFrame for export.

    Args:
        results: Results from run_stratified_analysis with estimate_mediation=True
        topic_name: Name of the treatment topic

    Returns:
        DataFrame with mediation results by contrast
    """
    mediation_results = results['overall'].get('mediation_results')

    if mediation_results is None or len(mediation_results) == 0:
        return pd.DataFrame()

    rows = []
    for contrast_label, result in mediation_results.items():
        if result is None:
            continue

        prop_reliable = result.get('prop_mediated_reliable', True)
        row = {
            'topic': topic_name,
            'contrast': contrast_label,
            'te': result['te'],
            'te_ci_lower': result['te_ci'][0],
            'te_ci_upper': result['te_ci'][1],
            'nde': result['nde'],
            'nde_ci_lower': result['nde_ci'][0],
            'nde_ci_upper': result['nde_ci'][1],
            'nie': result['nie'],
            'nie_ci_lower': result['nie_ci'][0],
            'nie_ci_upper': result['nie_ci'][1],
            # Store raw value; set to None when unreliable so downstream
            # tools don't accidentally use a -145% / -218% ratio.
            'prop_mediated': result['prop_mediated'] if prop_reliable else None,
            'prop_mediated_reliable': prop_reliable,
            'n_mediators': result['n_mediators'],
            'n_samples': result['n_samples'],
            'decomposition_error': result['decomposition_error']
        }
        rows.append(row)

    return pd.DataFrame(rows)


def export_mediation_results(results_dict: Dict[str, Dict], output_dir: str = '.'):
    """
    Export mediation results to CSV files.

    Args:
        results_dict: Dictionary mapping topic names to results
        output_dir: Directory to save CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Combine all mediation results
    all_mediation_results = []
    for topic_name, results in results_dict.items():
        df = format_mediation_table(results, topic_name)
        if len(df) > 0:
            all_mediation_results.append(df)

    if len(all_mediation_results) > 0:
        combined_mediation = pd.concat(all_mediation_results, ignore_index=True)

        # Save mediation results
        mediation_path = os.path.join(output_dir, 'mediation_summary.csv')
        combined_mediation.to_csv(mediation_path, index=False)
        print(f"\nMediation results saved to: {mediation_path}")

        return {'mediation_summary': mediation_path}
    else:
        print("\nNo mediation results to export.")
        return {}


def plot_mediation_results(
    results: Dict,
    topic_name: str,
    save_dir: Optional[str] = None
) -> list:
    """
    Generate mediation result plots and optionally save to disk.

    Produces two figures:
      1. Forest plot — NDE and NIE point estimates with 95 % CI error bars for
         every contrast present in the results.
      2. CATE by ideology — box plot + mean±SE bar chart of individual treatment
         effects grouped by outlet ideology (Left / Center / Right).

    Args:
        results: Results dict from run_stratified_analysis (must include
            'mediation_results' and 'cate_results' in the 'overall' key).
        topic_name: Label used in plot titles and file names.
        save_dir: If provided, save PNG files here and print their paths.

    Returns:
        List of (matplotlib.Figure, filename_str) tuples for every figure
        that was generated.
    """
    import os
    from matplotlib.lines import Line2D

    figs = []

    mediation_results = results['overall'].get('mediation_results') or {}
    valid_contrasts = {k: v for k, v in mediation_results.items() if v is not None}

    # ------------------------------------------------------------------ #
    # Plot 1 — NDE / NIE forest plot                                      #
    # ------------------------------------------------------------------ #
    if valid_contrasts:
        n_contrasts = len(valid_contrasts)
        fig, ax = plt.subplots(figsize=(9, max(3, 2.5 * n_contrasts)))

        # Two rows per contrast (NDE then NIE), with a gap between contrasts
        nde_color = '#2166ac'
        nie_color = '#d6604d'

        y_ticks = []
        y_tick_labels = []

        for idx, (label, result) in enumerate(valid_contrasts.items()):
            base_y = idx * 2.5
            for comp_offset, (comp, ci_key, color, comp_label) in enumerate([
                ('nde', 'nde_ci', nde_color, 'NDE'),
                ('nie', 'nie_ci', nie_color, 'NIE'),
            ]):
                y = base_y + comp_offset * 0.9
                val = result[comp]
                ci_lo, ci_hi = result[ci_key]
                ax.errorbar(
                    val, y,
                    xerr=[[val - ci_lo], [ci_hi - val]],
                    fmt='o', color=color, capsize=5, markersize=8, linewidth=2
                )
                y_ticks.append(y)
                y_tick_labels.append(f"{label} — {comp_label}")

        ax.axvline(0, color='grey', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_tick_labels, fontsize=9)
        ax.set_xlabel('Effect Size')
        ax.set_title(f'Mediation Effects (NDE / NIE) — {topic_name}')
        ax.grid(axis='x', alpha=0.3)

        legend_elements = [
            Line2D([0], [0], marker='o', color=nde_color, label='Natural Direct Effect (NDE)', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='o', color=nie_color, label='Natural Indirect Effect (NIE)', markersize=8, linestyle='None'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        plt.tight_layout()

        fname = 'mediation_forest_plot.png'
        if save_dir:
            path = os.path.join(save_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Mediation forest plot saved to: {path}")
        figs.append((fig, fname))

    # ------------------------------------------------------------------ #
    # Plot 2 — CATE distribution by outlet ideology                       #
    # ------------------------------------------------------------------ #
    cate_results = results['overall'].get('cate_results')
    cate_heterogeneity = results['overall'].get('cate_heterogeneity') or {}
    data = results['overall'].get('data')

    if cate_results is not None and data is not None:
        cate = cate_results['cate']
        ideology = data['Y']
        ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Left panel — box plot per ideology group
        ax1 = axes[0]
        box_data = [cate[ideology == val] for val in [0, 1, 2] if len(cate[ideology == val]) > 0]
        box_labels = [ideology_labels[val] for val in [0, 1, 2] if len(cate[ideology == val]) > 0]
        if box_data:
            ax1.boxplot(box_data, labels=box_labels, patch_artist=True,
                        boxprops=dict(facecolor='lightsteelblue', color='steelblue'),
                        medianprops=dict(color='navy'))
        ax1.axhline(0, color='grey', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Outlet Ideology')
        ax1.set_ylabel('Individual Treatment Effect (CATE)')
        ax1.set_title(f'CATE Distribution by Ideology\n{topic_name}')
        ax1.grid(axis='y', alpha=0.3)

        # Right panel — mean ± SE 95% CI bar chart
        ax2 = axes[1]
        valid_labels = [
            lbl for lbl in ['Left', 'Center', 'Right']
            if lbl in cate_heterogeneity and not cate_heterogeneity[lbl].get('note')
        ]
        if valid_labels:
            x = np.arange(len(valid_labels))
            means = [cate_heterogeneity[lbl]['mean_cate'] for lbl in valid_labels]
            se_lo = [
                cate_heterogeneity[lbl]['mean_cate'] - cate_heterogeneity[lbl]['se_ci_lower']
                for lbl in valid_labels
            ]
            se_hi = [
                cate_heterogeneity[lbl]['se_ci_upper'] - cate_heterogeneity[lbl]['mean_cate']
                for lbl in valid_labels
            ]
            ax2.bar(x, means, yerr=[se_lo, se_hi], capsize=6,
                    color='steelblue', alpha=0.7, error_kw={'linewidth': 2, 'capthick': 2})
            ax2.set_xticks(x)
            ax2.set_xticklabels(valid_labels)
            ax2.axhline(0, color='grey', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Outlet Ideology')
            ax2.set_ylabel('Mean CATE ± SE 95% CI')
            ax2.set_title(f'Mean CATE by Ideology\n{topic_name}')
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        fname = 'cate_by_ideology.png'
        if save_dir:
            path = os.path.join(save_dir, fname)
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"CATE by ideology plot saved to: {path}")
        figs.append((fig, fname))

    plt.close('all')
    return figs
