"""
Modular ATE and CATE Analysis Framework

This module provides functions for estimating Average Treatment Effects (ATE) and
Conditional Average Treatment Effects (CATE) for sentiment variables on ideology.

Causal Framework:
- Treatment (F_T): Sentiment toward a specific topic (continuous score in [-100, 100])
- Outcome (Y): Political ideology (0=Left, 1=Center, 2=Right)
- Mediators (F_M): Sentiments toward other topics
- Confounders (T): Topic presence indicators

The analysis estimates the total causal effect of sentiment on ideology,
including effects mediated through other sentiment variables.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import warnings
from dataclasses import dataclass

from joblib import Parallel, delayed

# EconML imports
from econml.dml import LinearDML, CausalForestDML
from econml.metalearners import SLearner

# Sklearn imports for base learners
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor



class InsufficientTopicPresenceError(Exception):
    """Raised when requested mediators are all removed by the min_topic_presence filter."""
    pass


class InsufficientSamplesError(Exception):
    """Raised when a topic has fewer articles than the required minimum sample count."""
    pass


@dataclass
class TreatmentConfig:
    """Configuration for a treatment variable analysis."""
    topic_name: str
    topic_col: Optional[str] = None
    sentiment_col: Optional[str] = None
    min_samples: int = 50  # Minimum samples required for analysis

    def __post_init__(self):
        """Validate configuration and set defaults."""
        if self.topic_col is None:
            self.topic_col = f'topic {self.topic_name}'
        elif not self.topic_col.startswith('topic '):
            self.topic_col = f'topic {self.topic_name}'

        if self.sentiment_col is None:
            self.sentiment_col = f'sentiment {self.topic_name}'
        elif not self.sentiment_col.startswith('sentiment '):
            self.sentiment_col = f'sentiment {self.topic_name}'


class DataPreprocessor:
    """Handles data preprocessing for causal analysis."""

    def __init__(
        self,
        df: pd.DataFrame,
        verbose: bool = False,
        apply_pca_to_confounders: bool = False,
        pca_variance_threshold: float = 0.95,
        pca_max_components: Optional[int] = None,
        min_topic_presence: int = 5,
    ):
        """
        Initialize preprocessor.

        Args:
            df: Full dataset
            verbose: Print detailed logs
            apply_pca_to_confounders: If True, apply PCA to the confounder matrix W
                to reduce dimensionality before passing it to estimators. Only active
                when return_mediation_split=True, so W is available separately from M.
                M is never reduced because its columns are individual topic sentiments
                with direct causal meaning as mediators.
            pca_variance_threshold: Fraction of variance the retained principal
                components must collectively explain (e.g. 0.95 = 95%). PCA selects
                the minimum number of components that reaches this threshold.
                Ignored when apply_pca_to_confounders=False.
            pca_max_components: Hard upper bound on retained PCA components,
                regardless of how many the variance threshold selects. Useful when
                N is close to the original number of features. None = no cap.
            min_topic_presence: Minimum number of articles in the filtered subsample
                in which a topic must appear before its indicator is kept in W and
                its sentiment is kept in M. Only applied when
                apply_pca_to_confounders=True. Enforces confounder-mediator
                consistency: sentiment X is only admissible as a mediator if topic X
                is sufficiently represented in W, so that PCA can actually control
                for its presence on the mediator-outcome path.
        """
        self.df = df.copy()
        self.verbose = verbose
        self.apply_pca_to_confounders = apply_pca_to_confounders
        self.pca_variance_threshold = pca_variance_threshold
        self.pca_max_components = pca_max_components
        self.min_topic_presence = min_topic_presence

    def prepare_treatment_data(
        self,
        config: TreatmentConfig,
        include_other_sentiments: bool = True,
        include_topic_indicators: bool = True,
        return_mediation_split: bool = False
    ) -> Dict:
        """
        Prepare data for a specific treatment analysis.

        Args:
            config: Treatment configuration
            include_other_sentiments: Include sentiments for other topics as mediators
            include_topic_indicators: Include topic presence as confounders
            return_mediation_split: If True, return separate M (mediators) and W (confounders) arrays

        Returns:
            Dictionary with prepared data arrays and metadata.
            If return_mediation_split=True, includes separate M, W, mediator_cols, confounder_cols
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Preparing data for treatment: {config.topic_name}")
            print('='*70)

        # Filter to rows where topic is present
        df_filtered = self.df[self.df[config.topic_col] == True].copy()

        if self.verbose:
            print(f"Total articles in dataset: {len(self.df)}")
            print(f"Articles with topic '{config.topic_name}': {len(df_filtered)}")

        if len(df_filtered) < config.min_samples:
            raise InsufficientSamplesError(
                f"Insufficient samples for topic '{config.topic_name}': "
                f"{len(df_filtered)} < {config.min_samples}"
            )

        # Extract treatment: raw continuous sentiment score (−100 … +100).
        T = df_filtered[config.sentiment_col].values.astype(np.float64)

        # Extract outcome (ideology)
        Y = df_filtered['int_bias'].values.astype(np.float64)

        # Build feature matrices
        # Separate confounders and mediators for mediation analysis
        confounder_cols = []
        mediator_cols = []

        # Add topic indicators (confounders)
        if include_topic_indicators:
            topic_cols = [col for col in self.df.columns if col.startswith('topic ')]
            # Exclude the treatment topic itself
            topic_cols = [col for col in topic_cols if col != config.topic_col]
            confounder_cols.extend(topic_cols)

        # Add other sentiments (mediators)
        if include_other_sentiments:
            sentiment_cols = [col for col in self.df.columns if col.startswith('sentiment ')]
            # Exclude the treatment sentiment itself
            sentiment_cols = [col for col in sentiment_cols if col != config.sentiment_col]
            mediator_cols.extend(sentiment_cols)

        # --- Confounder-mediator consistency filter (active only when PCA is used) ---
        # Confounders (W) = "topic X" presence indicators.
        # Mediators   (M) = "sentiment X" scores.
        # By construction, sentiment X == 0 whenever topic X is absent, so the
        # causal effect of sentiment X on ideology can only be identified through
        # articles where topic X is actually present.  After PCA compresses W, a
        # topic that appears in very few articles contributes negligibly to any
        # retained component — meaning PCA no longer effectively controls for its
        # presence.  Keeping sentiment X as a mediator in that situation violates
        # the sequential ignorability assumption (there is an uncontrolled
        # confounder of the sentiment X → ideology path).
        # Fix: drop both "topic X" from W and "sentiment X" from M for any topic
        # that appears in fewer than min_topic_presence articles in this subsample,
        # before PCA is fitted.
        n_confounder_cols_raw = len(confounder_cols)
        n_mediator_cols_raw = len(mediator_cols)
        if self.apply_pca_to_confounders and confounder_cols:
            topic_presence = df_filtered[confounder_cols].sum()
            active_confounder_cols = [
                col for col in confounder_cols
                if topic_presence[col] >= self.min_topic_presence
            ]
            # Naming convention: "topic X" <-> "sentiment X".
            # Only keep a sentiment column if its paired topic indicator survived.
            active_topic_names = {col[len('topic '):] for col in active_confounder_cols}
            active_mediator_cols = [
                col for col in mediator_cols
                if col.startswith('sentiment ') and col[len('sentiment '):] in active_topic_names
            ]
            n_dropped_conf = n_confounder_cols_raw - len(active_confounder_cols)
            n_dropped_med = n_mediator_cols_raw - len(active_mediator_cols)
            if self.verbose:
                print(f"\nConfounder-mediator consistency filter "
                      f"(min topic presence: {self.min_topic_presence}):")
                print(f"  Confounders: {n_confounder_cols_raw} → {len(active_confounder_cols)} "
                      f"({n_dropped_conf} rare topics removed)")
                print(f"  Mediators  : {n_mediator_cols_raw} → {len(active_mediator_cols)} "
                      f"({n_dropped_med} corresponding sentiments removed)")
            confounder_cols = active_confounder_cols
            mediator_cols = active_mediator_cols

        # Track confounder count after consistency filter but before PCA
        n_confounder_cols_original = len(confounder_cols)
        pca_confounders = None  # Will hold the fitted PCA object if PCA is applied

        # Build separate matrices for mediation analysis
        if return_mediation_split:
            # fillna(0): a NaN in a topic-presence indicator means the topic was
            # not tagged for this article, which is semantically equivalent to
            # absent (0).  NaN in a sentiment column means no sentiment was
            # computed because the topic was not present — neutral (0) is the
            # correct imputation, matching how sentiment is defined to be 0 when
            # the topic indicator is False.
            W = df_filtered[confounder_cols].fillna(0).values.astype(np.float64) if confounder_cols else np.empty((len(df_filtered), 0))
            M = df_filtered[mediator_cols].fillna(0).values.astype(np.float64) if mediator_cols else np.empty((len(df_filtered), 0))

            # --- PCA dimensionality reduction on confounders (W) ---
            # M is left untouched: each column is a specific topic's sentiment and
            # has direct causal meaning as a mediator.  Reducing M would destroy
            # the individual mediator estimates needed for NIE decomposition.
            if self.apply_pca_to_confounders and W.shape[1] > 1:
                # Step 1: probe fit with float n_components to discover how many
                # components are needed to reach the variance threshold.
                # sklearn selects the *minimum* integer k such that
                # sum(explained_variance_ratio_[:k]) >= pca_variance_threshold.
                pca_probe = PCA(n_components=self.pca_variance_threshold)
                pca_probe.fit(W)
                n_components = pca_probe.n_components_

                # Step 2: apply the optional hard cap.  Important when N is close
                # to the original feature count: the variance threshold alone may
                # select nearly as many components as there were original features.
                if self.pca_max_components is not None:
                    n_components = min(n_components, self.pca_max_components)

                # Step 3: fit and transform W using the resolved integer n_components.
                pca_confounders = PCA(n_components=n_components)
                W = pca_confounders.fit_transform(W)

                if self.verbose:
                    var_retained = pca_confounders.explained_variance_ratio_.sum()
                    print(f"\nPCA applied to confounders (W):")
                    print(f"  Original dimensions : {n_confounder_cols_original}")
                    print(f"  Reduced dimensions  : {n_components}")
                    print(f"  Variance retained   : {var_retained:.3f} "
                          f"(threshold: {self.pca_variance_threshold:.2f})")
                    if self.pca_max_components is not None:
                        print(f"  Max components cap  : {self.pca_max_components}")

            # Combined matrix for backward compatibility
            X = np.hstack([W, M]) if (W.shape[1] > 0 or M.shape[1] > 0) else np.ones((len(df_filtered), 1))
        else:
            # Original behavior: combine all features
            feature_cols = confounder_cols + mediator_cols
            # fillna(0): same imputation as the mediation split path above —
            # absent topic → indicator 0; topic not present → sentiment 0.
            X = df_filtered[feature_cols].fillna(0).values.astype(np.float64) if feature_cols else np.ones((len(df_filtered), 1))
            W = None
            M = None

        # Check if we have any covariates
        if X.shape[1] == 1 and not (W is not None and (W.shape[1] > 0 or M.shape[1] > 0)):
            warnings.warn(
                "No covariates (confounders or mediators) included in analysis. "
                "This will produce correlation estimates, not causal effects. "
                "Consider setting include_other_sentiments=True or include_topic_indicators=True."
            )
            if self.verbose:
                print("\nWarning: No covariates provided. Using constant feature.")

        if self.verbose:
            print(f"\nData shapes:")
            print(f"  Treatment (T): {T.shape}")
            print(f"  Outcome (Y): {Y.shape}")
            print(f"  Covariates (X): {X.shape}")
            print(f"\nTreatment (sentiment) distribution:")
            p10, p25, p50, p75, p90 = np.percentile(T, [10, 25, 50, 75, 90])
            print(f"  Mean ± SD : {T.mean():.2f} ± {T.std():.2f}")
            print(f"  Range     : [{T.min():.1f}, {T.max():.1f}]")
            print(f"  Percentiles (P10/Q1/med/Q3/P90): "
                  f"{p10:.1f} / {p25:.1f} / {p50:.1f} / {p75:.1f} / {p90:.1f}")

            print(f"\nOutcome (ideology) distribution:")
            ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}
            for ideology in [0, 1, 2]:
                count = (Y == ideology).sum()
                pct = count / len(Y) * 100
                print(f"  {ideology_labels[ideology]:6s} ({ideology}): {count:4d} ({pct:5.1f}%)")

        # Prepare return dictionary
        result = {
            'T': T,
            'Y': Y,
            'X': X,
            'df_filtered': df_filtered,
            'feature_cols': confounder_cols + mediator_cols,
            'config': config,
            'n_samples': len(df_filtered),
            'pca_confounders': pca_confounders,               # fitted PCA object, or None if PCA not applied
            'n_confounder_cols_original': n_confounder_cols_original  # original W width before any reduction
        }

        # Add mediation-specific fields if requested
        if return_mediation_split:
            assert W is not None and M is not None  # guaranteed by the branch above; narrows type for checker
            result['M'] = M
            result['W'] = W
            result['mediator_cols'] = mediator_cols
            result['confounder_cols'] = confounder_cols

            if self.verbose:
                print(f"\nMediation split:")
                if pca_confounders is not None:
                    print(f"  Confounders (W): {W.shape} "
                          f"(PCA-reduced from {n_confounder_cols_original})")
                else:
                    print(f"  Confounders (W): {W.shape}")
                print(f"  Mediators (M): {M.shape}")
                print(f"  Number of mediators: {len(mediator_cols)}")
                if pca_confounders is not None:
                    print(f"  Number of confounders: {n_confounder_cols_original} "
                          f"→ {W.shape[1]} (after PCA)")
                else:
                    print(f"  Number of confounders: {len(confounder_cols)}")

        return result

    def stratify_by_ideology(
        self,
        data: Dict,
        ideology_value: int
    ) -> Dict:
        """
        Stratify prepared data by ideology (outcome).

        This allows estimation of treatment effects separately for each
        ideology group to examine effect heterogeneity.

        Args:
            data: Output from prepare_treatment_data
            ideology_value: 0 (Left), 1 (Center), or 2 (Right)

        Returns:
            Stratified data dictionary
        """
        mask = data['Y'] == ideology_value

        if self.verbose:
            ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}
            print(f"\nStratifying to ideology={ideology_labels[ideology_value]} ({ideology_value})")
            print(f"  Samples: {int(mask.sum())} / {len(mask)}")

        return {
            'T': data['T'][mask],
            'Y': data['Y'][mask],
            'X': data['X'][mask],
            'df_filtered': data['df_filtered'][mask],
            'feature_cols': data['feature_cols'],
            'config': data['config'],
            'n_samples': int(mask.sum()),
            'stratified_by': f"ideology_{ideology_value}"
        }


class ATEEstimator:
    """Estimates Average Treatment Effects using multiple methods."""

    def __init__(self, verbose: bool = False, random_state: int = 42):
        """
        Initialize ATE estimator.

        Args:
            verbose: Print detailed logs
            random_state: Random seed for reproducibility
        """
        self.verbose = verbose
        self.random_state = random_state

    def _get_model_y(self):
        """Get model for outcome (ideology is ordinal 0,1,2)."""
        # Treat ideology as continuous/ordinal (Left=0, Center=1, Right=2)
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=self.random_state
        )

    def _get_model_t(self):
        """Get model for treatment (sentiment is continuous)."""
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=self.random_state
        )

    def estimate_dml(self, data: Dict) -> Dict:
        """
        Estimate ATE using Double Machine Learning (DML).

        Args:
            data: Prepared data from DataPreprocessor

        Returns:
            Results dictionary with ATE estimate and statistics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("Estimating ATE using Double Machine Learning (DML)")
            print('='*70)

        T, Y, X = data['T'], data['Y'], data['X']

        # LinearDML for continuous treatment and ordinal outcome
        est = LinearDML(
            model_y=self._get_model_y(),
            model_t=self._get_model_t(),
            random_state=self.random_state
        )

        est.fit(Y, T, X=X, W=None)

        # Get ATE (effect of unit change in sentiment)
        ate = est.ate(X=X)
        ate_interval = est.ate_interval(X=X, alpha=0.05)

        if self.verbose:
            print(f"ATE: {ate:.4f}")
            print(f"95% CI: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")

        return {
            'method': 'DML',
            'ate': ate,
            'ate_lower': ate_interval[0],
            'ate_upper': ate_interval[1],
            'estimator': est,
            'n_samples': data['n_samples']
        }

    def estimate_slearner(self, data: Dict) -> Dict:
        """
        Estimate ATE using S-Learner meta-learner.

        Args:
            data: Prepared data from DataPreprocessor

        Returns:
            Results dictionary with ATE estimate and statistics
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("Estimating ATE using S-Learner")
            print('='*70)

        T, Y, X = data['T'], data['Y'], data['X']

        est = SLearner(
            overall_model=RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=self.random_state
            )
        )

        est.fit(Y, T, X=X)

        # Get ATE by computing average effect across all samples
        ate = est.ate(X=X)

        if self.verbose:
            print(f"ATE: {ate:.4f}")

        return {
            'method': 'S-Learner',
            'ate': ate,
            'ate_lower': None,
            'ate_upper': None,
            'estimator': est,
            'n_samples': data['n_samples']
        }

    def estimate_all(self, data: Dict) -> pd.DataFrame:
        """
        Estimate ATE using all available methods.

        Args:
            data: Prepared data from DataPreprocessor

        Returns:
            DataFrame with results from all methods
        """
        results = []

        # Run all estimators
        for estimator_func in [
            self.estimate_dml,
            self.estimate_slearner,
        ]:
            try:
                result = estimator_func(data)
                results.append(result)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: {estimator_func.__name__} failed: {str(e)}")

        # Convert to DataFrame
        df_results = pd.DataFrame([
            {
                'method': r['method'],
                'ate': r['ate'],
                'ate_lower': r.get('ate_lower'),
                'ate_upper': r.get('ate_upper'),
                'n_samples': r['n_samples'],
                'note': r.get('note', '')
            }
            for r in results
        ])

        return df_results, results


class CATEEstimator:
    """Estimates Conditional Average Treatment Effects."""

    def __init__(self, verbose: bool = False, random_state: int = 42):
        """
        Initialize CATE estimator.

        Args:
            verbose: Print detailed logs
            random_state: Random seed for reproducibility
        """
        self.verbose = verbose
        self.random_state = random_state

    def estimate_causal_forest(self, data: Dict) -> Dict:
        """
        Estimate CATE using Causal Forest (via CausalForestDML).

        Args:
            data: Prepared data from DataPreprocessor

        Returns:
            Results dictionary with CATE estimates
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("Estimating CATE using Causal Forest")
            print('='*70)

        T, Y, X = data['T'], data['Y'], data['X']

        # Use Causal Forest DML (treating ideology as ordinal)
        est = CausalForestDML(
            model_y=RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state
            ),
            model_t=RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state
            ),
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=self.random_state
        )

        est.fit(Y, T, X=X, W=None)

        # Get individual treatment effects
        cate = est.effect(X=X)

        # Summary statistics
        ate = est.ate(X=X)
        ate_interval = est.ate_interval(X=X, alpha=0.05)

        if self.verbose:
            print(f"ATE (from CATE model): {ate:.4f}")
            print(f"95% CI: [{ate_interval[0]:.4f}, {ate_interval[1]:.4f}]")
            print(f"\nCATE distribution:")
            print(f"  Mean: {np.mean(cate):.4f}")
            print(f"  Std: {np.std(cate):.4f}")
            print(f"  Min: {np.min(cate):.4f}")
            print(f"  Max: {np.max(cate):.4f}")
            print(f"  Median: {np.median(cate):.4f}")

        return {
            'method': 'Causal Forest',
            'cate': cate,
            'ate': ate,
            'ate_lower': ate_interval[0],
            'ate_upper': ate_interval[1],
            'estimator': est,
            'n_samples': data['n_samples']
        }

    def analyze_heterogeneity_by_ideology(
        self,
        data: Dict,
        cate_results: Dict
    ) -> Dict:
        """
        Stratify individual CATE predictions by outlet ideology (Left/Center/Right).

        This is NOT the same as refitting the estimator within each ideology stratum
        (which would fail because ideology is the outcome variable and would have no
        variation within each stratum).  Instead:
          1. The Causal Forest is already fitted on the full sample.
          2. Its per-article predictions are grouped by each article's observed
             ideology label.
          3. Summary statistics are computed within each group.

        Interpretation: if Left-coded articles have a more negative mean CATE than
        Right-coded articles, it means the forest predicts a stronger (more
        left-shifting) treatment effect for articles whose covariate profile
        resembles left-leaning outlets.  This reveals whether treatment effect
        heterogeneity correlates with the outlet's existing ideological position.

        Two confidence interval types are reported:
          se_ci:     ±1.96 * std(CATE_group) / sqrt(n_group). Treats each
                     individual CATE estimate as if it were observed (no model
                     error). Valid when the forest is well-specified.
          forest_ci: Mean of CausalForestDML's per-article 95% CIs within each
                     group (infinitesimal jackknife). Accounts for individual-level
                     estimation uncertainty; tends to be wider than se_ci.

        Args:
            data: Output from DataPreprocessor (must contain 'Y' and 'X').
            cate_results: Output from estimate_causal_forest (must contain 'cate'
                and 'estimator').

        Returns:
            Dict keyed by ideology label ('Left', 'Center', 'Right'), each value
            a dict of summary statistics.
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("CATE HETEROGENEITY ANALYSIS BY OUTLET IDEOLOGY")
            print('='*70)
            print("Grouping individual CATE predictions by observed ideology label.")
            print("(Forest was fitted on the full sample; only predictions are grouped.)")

        Y = data['Y']
        X = data['X']
        cate = cate_results['cate']
        est = cate_results['estimator']
        ideology_labels = {0: 'Left', 1: 'Center', 2: 'Right'}

        # Forest-based pointwise CIs via infinitesimal jackknife (CausalForestDML).
        # Averaging these within ideology groups gives approximate group-level
        # intervals that capture individual-level estimation uncertainty.
        try:
            cate_lower_pw, cate_upper_pw = est.effect_interval(X, alpha=0.05)
            has_forest_ci = True
        except Exception as e:
            if self.verbose:
                print(f"  (Forest pointwise CIs unavailable: {e})")
            cate_lower_pw = cate_upper_pw = None
            has_forest_ci = False

        results = {}
        for val, label in ideology_labels.items():
            mask = Y == val
            n = int(mask.sum())

            if n < 5:
                results[label] = {
                    'n': n,
                    'note': 'Too few samples for reliable estimate'
                }
                continue

            cate_group = cate[mask]
            mean_cate = float(np.mean(cate_group))
            std_cate = float(np.std(cate_group, ddof=1))
            se = std_cate / np.sqrt(n)

            entry: Dict = {
                'n': n,
                'mean_cate': mean_cate,
                'std_cate': std_cate,
                'se_ci_lower': mean_cate - 1.96 * se,
                'se_ci_upper': mean_cate + 1.96 * se,
            }

            if has_forest_ci:
                assert cate_lower_pw is not None and cate_upper_pw is not None
                entry['forest_ci_lower'] = float(np.mean(cate_lower_pw[mask]))
                entry['forest_ci_upper'] = float(np.mean(cate_upper_pw[mask]))

            results[label] = entry

        if self.verbose:
            col1, col2, col3, col4 = 10, 5, 10, 22
            hdr = (f"\n  {'Ideology':<{col1}} {'N':>{col2}}  "
                   f"{'Mean CATE':>{col3}}  {'SE-based 95% CI':>{col4}}")
            if has_forest_ci:
                hdr += f"  {'Forest 95% CI (avg)':>24}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 3))

            for label in ['Left', 'Center', 'Right']:
                r = results.get(label, {})
                if r.get('note'):
                    print(f"  {label:<{col1}} {r['n']:>{col2}}  (too few samples)")
                    continue
                se_ci = f"[{r['se_ci_lower']:+.4f}, {r['se_ci_upper']:+.4f}]"
                row = (f"  {label:<{col1}} {r['n']:>{col2}}  "
                       f"{r['mean_cate']:>+{col3}.4f}  {se_ci:>{col4}}")
                if has_forest_ci:
                    f_ci = f"[{r['forest_ci_lower']:+.4f}, {r['forest_ci_upper']:+.4f}]"
                    row += f"  {f_ci:>24}"
                print(row)

            overall_mean = float(np.mean(cate))
            print(f"\n  Overall (all articles): mean CATE = {overall_mean:+.4f}")
            valid_means = [r['mean_cate'] for r in results.values() if 'mean_cate' in r]
            if len(valid_means) >= 2:
                het_range = max(valid_means) - min(valid_means)
                print(f"  Heterogeneity range (max − min group mean): {het_range:.4f}")

        return results


class MediationEstimator:
    """
    Estimates Natural Direct and Indirect Effects for mediation analysis.

    Decomposes total effect into:
    - NDE (Natural Direct Effect): Effect NOT through mediators
    - NIE (Natural Indirect Effect): Effect THROUGH mediators

    Uses DML-based estimation with cross-fitting for valid inference.
    """

    def __init__(
        self,
        verbose: bool = False,
        random_state: int = 42,
        n_splits: int = 2,
        n_bootstrap: int = 100
    ):
        """
        Initialize mediation estimator.

        Args:
            verbose: Print detailed logs
            random_state: Random seed for reproducibility
            n_splits: Number of cross-fitting splits for DML
            n_bootstrap: Number of bootstrap iterations for CIs
        """
        self.verbose = verbose
        self.random_state = random_state
        self.n_splits = n_splits
        self.n_bootstrap = n_bootstrap

    def estimate_mediation_dml(
        self,
        data: Dict,
        treatment_baseline: int = 0,
        treatment_target: int = 1
    ) -> Dict:
        """
        Estimate NDE and NIE using DML-based mediation analysis.

        This method estimates mediation effects for a specific treatment contrast
        (e.g., Neutral → Positive). For multiple contrasts, call this method
        multiple times with different baseline/target values.

        Args:
            data: Output from DataPreprocessor with M, W separated
            treatment_baseline: Reference treatment level (e.g., 0=Neutral)
            treatment_target: Target treatment level (e.g., 1=Positive)

        Returns:
            Dictionary with:
                - te: Total Effect
                - nde: Natural Direct Effect
                - nie: Natural Indirect Effect
                - te_ci, nde_ci, nie_ci: 95% confidence intervals
                - prop_mediated: NIE/TE proportion
                - n_mediators: Number of mediators
                - mediator_effects: Individual mediator contributions
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Mediation Analysis: Treatment {treatment_baseline} → {treatment_target}")
            print('='*70)

        # Extract data
        T = data['T']
        Y = data['Y']
        M = data.get('M')
        W = data.get('W')

        # Check if mediation split was performed
        if M is None or W is None:
            raise ValueError(
                "Mediation analysis requires separate M and W matrices. "
                "Set return_mediation_split=True when preparing data."
            )

        if M.shape[1] == 0:
            if self.verbose:
                print("Warning: No mediators available. Cannot perform mediation analysis.")
            return None

        n_samples = len(T)
        n_mediators = M.shape[1]

        if self.verbose:
            print(f"\nSample size: {n_samples}")
            print(f"Number of mediators: {n_mediators}")
            print(f"Number of confounders: {W.shape[1]}")
            print(f"Treatment contrast: {treatment_baseline} → {treatment_target}")

        # Step 1: Estimate Total Effect using existing DML
        from econml.dml import LinearDML

        # Combine M and W for total effect estimation
        X_total = np.hstack([W, M]) if W.shape[1] > 0 else M

        te_estimator = LinearDML(
            model_y=RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state
            ),
            model_t=RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state
            ),
            random_state=self.random_state
        )

        te_estimator.fit(Y, T, X=X_total, W=None)

        # Get total effect for this contrast
        te = te_estimator.effect(X_total, T0=treatment_baseline, T1=treatment_target).mean()
        te_interval = te_estimator.effect_interval(
            X_total, T0=treatment_baseline, T1=treatment_target, alpha=0.05
        )
        te_ci_lower = te_interval[0].mean()
        te_ci_upper = te_interval[1].mean()

        if self.verbose:
            print(f"\nTotal Effect (TE): {te:.4f} [{te_ci_lower:.4f}, {te_ci_upper:.4f}]")

        # Step 2: Fit mediator models (T → M) with cross-fitting
        from sklearn.model_selection import KFold

        mediator_models = []
        M_counterfactual_baseline = np.zeros_like(M)
        M_counterfactual_target = np.zeros_like(M)

        if self.verbose:
            print(f"\nFitting {n_mediators} mediator models with cross-fitting...")

        for j in range(n_mediators):
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state + j)

            M_pred_baseline = np.zeros(n_samples)
            M_pred_target = np.zeros(n_samples)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(M)):
                # Fit mediator model on training set
                model_m = RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_leaf=10,
                    random_state=self.random_state + j + fold_idx
                )

                # Features for mediator model: T + W
                if W.shape[1] > 0:
                    X_mediator_train = np.column_stack([T[train_idx].reshape(-1, 1), W[train_idx]])
                else:
                    X_mediator_train = T[train_idx].reshape(-1, 1)

                model_m.fit(X_mediator_train, M[train_idx, j])

                # Predict mediator values at baseline and target treatment levels
                # For test set
                T_baseline = np.full(len(test_idx), treatment_baseline)
                T_target_arr = np.full(len(test_idx), treatment_target)

                if W.shape[1] > 0:
                    X_baseline = np.column_stack([T_baseline.reshape(-1, 1), W[test_idx]])
                    X_target = np.column_stack([T_target_arr.reshape(-1, 1), W[test_idx]])
                else:
                    X_baseline = T_baseline.reshape(-1, 1)
                    X_target = T_target_arr.reshape(-1, 1)

                M_pred_baseline[test_idx] = model_m.predict(X_baseline)
                M_pred_target[test_idx] = model_m.predict(X_target)

            M_counterfactual_baseline[:, j] = M_pred_baseline
            M_counterfactual_target[:, j] = M_pred_target
            mediator_models.append(model_m)  # Save last model for reference

        if self.verbose:
            print("Mediator models fitted.")

        # Step 3: Estimate NDE - effect when mediators held at baseline
        # NDE = E[Y(T=target, M=M(baseline))] - E[Y(T=baseline, M=M(baseline))]

        # Create datasets with counterfactual mediators
        X_nde_target = np.hstack([W, M_counterfactual_baseline]) if W.shape[1] > 0 else M_counterfactual_baseline
        X_nde_baseline = np.hstack([W, M_counterfactual_baseline]) if W.shape[1] > 0 else M_counterfactual_baseline

        # Use DML to estimate effect with mediators fixed
        nde_estimator = LinearDML(
            model_y=RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state + 2000
            ),
            model_t=RandomForestRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                random_state=self.random_state + 2000
            ),
            random_state=self.random_state + 2000
        )

        # Fit with counterfactual mediators (mediators at baseline level)
        nde_estimator.fit(Y, T, X=X_nde_baseline, W=None)
        nde = nde_estimator.effect(X_nde_baseline, T0=treatment_baseline, T1=treatment_target).mean()

        if self.verbose:
            print(f"Natural Direct Effect (NDE): {nde:.4f}")

        # Step 4: Estimate NIE - effect through mediators
        # NIE = TE - NDE
        nie = te - nde

        if self.verbose:
            print(f"Natural Indirect Effect (NIE): {nie:.4f}")

        # Step 5: Bootstrap confidence intervals
        if self.verbose:
            print(f"\nBootstrapping confidence intervals ({self.n_bootstrap} iterations)...")

        nde_boots = []
        nie_boots = []

        np.random.seed(self.random_state)

        for b in range(self.n_bootstrap):
            if self.verbose and (b + 1) % 10 == 0:
                print(f"  Bootstrap iteration {b + 1}/{self.n_bootstrap}")
            # Resample with replacement
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)

            T_boot = T[boot_idx]
            Y_boot = Y[boot_idx]
            M_boot = M[boot_idx]
            W_boot = W[boot_idx] if W.shape[1] > 0 else np.empty((len(boot_idx), 0))

            try:
                # Fit mediator models on bootstrap sample
                M_cf_baseline_boot = np.zeros_like(M_boot)

                for j in range(n_mediators):
                    model_m_boot = RandomForestRegressor(
                        n_estimators=50, max_depth=5, min_samples_leaf=10,
                        random_state=self.random_state + b + j
                    )

                    if W_boot.shape[1] > 0:
                        X_m_boot = np.column_stack([T_boot.reshape(-1, 1), W_boot])
                    else:
                        X_m_boot = T_boot.reshape(-1, 1)

                    model_m_boot.fit(X_m_boot, M_boot[:, j])

                    # Predict at baseline
                    T_baseline_boot = np.full(len(boot_idx), treatment_baseline)
                    if W_boot.shape[1] > 0:
                        X_baseline_boot = np.column_stack([T_baseline_boot.reshape(-1, 1), W_boot])
                    else:
                        X_baseline_boot = T_baseline_boot.reshape(-1, 1)

                    M_cf_baseline_boot[:, j] = model_m_boot.predict(X_baseline_boot)

                # Estimate NDE on bootstrap sample
                X_nde_boot = np.hstack([W_boot, M_cf_baseline_boot]) if W_boot.shape[1] > 0 else M_cf_baseline_boot

                nde_est_boot = LinearDML(
                    model_y=RandomForestRegressor(
                        n_estimators=50, max_depth=5, min_samples_leaf=10,
                        random_state=self.random_state + b + 2000
                    ),
                    model_t=RandomForestRegressor(
                        n_estimators=50, max_depth=5, min_samples_leaf=10,
                        random_state=self.random_state + b + 2000
                    ),
                    random_state=self.random_state + b + 2000
                )

                nde_est_boot.fit(Y_boot, T_boot, X=X_nde_boot, W=None)
                nde_boot = nde_est_boot.effect(X_nde_boot, T0=treatment_baseline, T1=treatment_target).mean()

                # Estimate TE on bootstrap sample
                X_total_boot = np.hstack([W_boot, M_boot]) if W_boot.shape[1] > 0 else M_boot
                te_est_boot = LinearDML(
                    model_y=RandomForestRegressor(
                        n_estimators=50, max_depth=5, min_samples_leaf=10,
                        random_state=self.random_state + b + 2000
                    ),
                    model_t=RandomForestRegressor(
                        n_estimators=50, max_depth=5, min_samples_leaf=10,
                        random_state=self.random_state + b + 2000
                    ),
                    random_state=self.random_state + b + 2000
                )

                te_est_boot.fit(Y_boot, T_boot, X=X_total_boot, W=None)
                te_boot = te_est_boot.effect(X_total_boot, T0=treatment_baseline, T1=treatment_target).mean()

                nie_boot = te_boot - nde_boot

                nde_boots.append(nde_boot)
                nie_boots.append(nie_boot)

            except Exception as e:
                if self.verbose and b < 5:  # Only print first few errors
                    print(f"  Bootstrap iteration {b} failed: {str(e)[:50]}")
                continue

        # Calculate confidence intervals from bootstrap samples
        if len(nde_boots) >= 10:  # Need at least 10 successful boots
            nde_ci_lower = np.percentile(nde_boots, 2.5)
            nde_ci_upper = np.percentile(nde_boots, 97.5)
            nie_ci_lower = np.percentile(nie_boots, 2.5)
            nie_ci_upper = np.percentile(nie_boots, 97.5)

            if self.verbose:
                print(f"Bootstrap completed: {len(nde_boots)} successful iterations")
        else:
            if self.verbose:
                print(f"Warning: Only {len(nde_boots)} successful bootstrap iterations. Using approximation.")
            # Fallback: use wider CIs
            nde_ci_lower = nde - 2 * np.abs(nde)
            nde_ci_upper = nde + 2 * np.abs(nde)
            nie_ci_lower = nie - 2 * np.abs(nie)
            nie_ci_upper = nie + 2 * np.abs(nie)

        # Calculate proportion mediated
        prop_mediated = nie / te if abs(te) > 1e-6 else 0.0

        # Flag whether prop_mediated is numerically meaningful.  The ratio
        # NIE/TE blows up when TE is near zero, producing values like -145% or
        # -218% that are arithmetically correct but substantively uninterpretable.
        # We suppress the value in reports when either:
        #   (a) |TE| < 0.02  (negligibly small total effect), or
        #   (b) the TE 95% CI straddles zero (no evidence of any effect)
        te_ci_includes_zero = (te_ci_lower <= 0 <= te_ci_upper)
        prop_mediated_reliable = abs(te) >= 0.02 and not te_ci_includes_zero

        # Check decomposition
        decomposition_error = abs(te - (nde + nie))

        if self.verbose:
            print(f"\nResults:")
            print(f"  TE:  {te:.4f} [{te_ci_lower:.4f}, {te_ci_upper:.4f}]")
            print(f"  NDE: {nde:.4f} [{nde_ci_lower:.4f}, {nde_ci_upper:.4f}]")
            print(f"  NIE: {nie:.4f} [{nie_ci_lower:.4f}, {nie_ci_upper:.4f}]")
            if prop_mediated_reliable:
                print(f"  Proportion mediated: {prop_mediated:.1%}")
            else:
                print(f"  Proportion mediated: N/A (TE \u2248 0 or CI includes zero)")
            print(f"  Decomposition check: |TE - (NDE + NIE)| = {decomposition_error:.4f}")

        return {
            'method': 'Mediation-DML',
            'contrast': f"{treatment_baseline} → {treatment_target}",
            'te': te,
            'nde': nde,
            'nie': nie,
            'te_ci': (te_ci_lower, te_ci_upper),
            'nde_ci': (nde_ci_lower, nde_ci_upper),
            'nie_ci': (nie_ci_lower, nie_ci_upper),
            'prop_mediated': prop_mediated,
            'prop_mediated_reliable': prop_mediated_reliable,
            'n_mediators': n_mediators,
            'n_samples': n_samples,
            'decomposition_error': decomposition_error,
            'n_bootstrap_success': len(nde_boots)
        }


def run_stratified_analysis(
    df: pd.DataFrame,
    config: TreatmentConfig,
    include_other_sentiments: bool = True,
    include_topic_indicators: bool = True,
    estimate_mediation: bool = False,
    mediation_method: str = 'dml',
    mediator_subset: Optional[List[str]] = None,
    verbose: bool = False,
    random_state: int = 42,
    n_bootstrap: int = 100,
    apply_pca_to_confounders: bool = False,
    pca_variance_threshold: float = 0.95,
    pca_max_components: Optional[int] = None,
    min_topic_presence: int = 5,
) -> Dict:
    """
    Run complete ATE/CATE analysis, optionally with mediation analysis.

    Args:
        df: Full dataset
        config: Treatment configuration
        include_other_sentiments: Include other sentiments as mediators
        include_topic_indicators: Include topic indicators as confounders
        estimate_mediation: If True, estimate NIE/NDE mediation effects
        mediation_method: Method for mediation analysis ('dml' or 'sequential')
        mediator_subset: Optional list of specific mediator column names to focus on
        verbose: Print detailed logs
        random_state: Random seed
        apply_pca_to_confounders: Reduce the confounder matrix W via PCA before
            passing it to all estimators. Has no effect unless estimate_mediation=True
            (W is only available as a separate matrix in mediation mode).
        pca_variance_threshold: Fraction of variance the PCA components must explain.
        pca_max_components: Hard cap on the number of PCA components retained.
        min_topic_presence: Minimum article count for a topic to be retained in W
            and its sentiment retained in M. Enforces confounder-mediator consistency
            when apply_pca_to_confounders=True.

    Returns:
        Dictionary with results for overall and stratified analyses.
        If estimate_mediation=True, includes 'mediation_results' with NIE/NDE decomposition.
    """
    preprocessor = DataPreprocessor(
        df,
        verbose=verbose,
        apply_pca_to_confounders=apply_pca_to_confounders,
        pca_variance_threshold=pca_variance_threshold,
        pca_max_components=pca_max_components,
        min_topic_presence=min_topic_presence,
    )
    ate_estimator = ATEEstimator(verbose=verbose, random_state=random_state)
    cate_estimator = CATEEstimator(verbose=verbose, random_state=random_state)

    # Prepare overall data (with mediation split if requested)
    data_overall = preprocessor.prepare_treatment_data(
        config,
        include_other_sentiments=include_other_sentiments,
        include_topic_indicators=include_topic_indicators,
        return_mediation_split=estimate_mediation
    )

    # Filter mediators to subset if specified
    if estimate_mediation and mediator_subset is not None and data_overall.get('M') is not None:
        mediator_cols = data_overall['mediator_cols']
        # Find indices of requested mediators
        mediator_indices = [i for i, col in enumerate(mediator_cols) if col in mediator_subset]

        if len(mediator_indices) > 0:
            data_overall['M'] = data_overall['M'][:, mediator_indices]
            data_overall['mediator_cols'] = [mediator_cols[i] for i in mediator_indices]
            # Rebuild X matrix with filtered mediators
            W = data_overall['W']
            M_filtered = data_overall['M']
            data_overall['X'] = np.hstack([W, M_filtered]) if W.shape[1] > 0 else M_filtered
            data_overall['feature_cols'] = data_overall['confounder_cols'] + data_overall['mediator_cols']

            if verbose:
                print(f"\nFiltered to {len(mediator_indices)} mediators from subset:")
                for col in data_overall['mediator_cols']:
                    print(f"  - {col}")
        else:
            raise InsufficientTopicPresenceError(
                f"None of the requested mediators {mediator_subset} survived the "
                f"min_topic_presence={min_topic_presence} filter. "
                f"Available mediators after filter: {mediator_cols}"
            )

    # Overall ATE
    if verbose:
        print("\n" + "="*70)
        print("OVERALL ANALYSIS (All sentiment levels)")
        print("="*70)

    ate_results_overall, ate_objects_overall = ate_estimator.estimate_all(data_overall)

    # Only estimate CATE if we have features
    cate_heterogeneity_overall = None
    if data_overall['X'].shape[1] > 0:
        cate_results_overall = cate_estimator.estimate_causal_forest(data_overall)
        cate_heterogeneity_overall = cate_estimator.analyze_heterogeneity_by_ideology(
            data_overall, cate_results_overall
        )
    else:
        if verbose:
            print("\nWarning: No features available for CATE estimation (need covariates)")
        cate_results_overall = None

    # Mediation analysis (if requested)
    mediation_results_overall = None
    if estimate_mediation:
        if verbose:
            print("\n" + "="*70)
            print("MEDIATION ANALYSIS (NIE/NDE Decomposition)")
            print("="*70)

        if data_overall.get('M') is not None and data_overall['M'].shape[1] > 0:
            mediation_estimator = MediationEstimator(
                verbose=verbose,
                random_state=random_state,
                n_splits=2,
                n_bootstrap=n_bootstrap
            )

            # Data-driven percentile contrasts on the raw continuous score
            # (IQR and P10→P90).
            T_vals = data_overall['T']
            p10, p25, p75, p90 = np.percentile(T_vals, [10, 25, 75, 90])
            contrasts = [
                (p25, p75, f'Q25 → Q75 ({p25:.1f} → {p75:.1f})'),
                (p10, p90, f'P10 → P90 ({p10:.1f} → {p90:.1f})'),
            ]

            mediation_results_overall = {}

            for baseline, target, label in contrasts:
                if verbose:
                    print(f"\n{'-'*70}")
                    print(f"Contrast: {label}")
                    print(f"{'-'*70}")

                try:
                    result = mediation_estimator.estimate_mediation_dml(
                        data_overall,
                        treatment_baseline=baseline,
                        treatment_target=target
                    )

                    if result is not None:
                        mediation_results_overall[label] = result
                except Exception as e:
                    if verbose:
                        print(f"Error estimating mediation for {label}: {str(e)}")
                    mediation_results_overall[label] = None

        else:
            if verbose:
                print("\nWarning: No mediators available for mediation analysis.")
                print("Set include_other_sentiments=True and ensure mediator data exists.")

    return {
        'overall': {
            'ate_results': ate_results_overall,
            'ate_objects': ate_objects_overall,
            'cate_results': cate_results_overall,
            'cate_heterogeneity': cate_heterogeneity_overall,
            'mediation_results': mediation_results_overall,
            'data': data_overall
        },
        'config': config
    }


def run_multi_treatment_dml(
    df: pd.DataFrame,
    outcome_col: str = 'int_bias',
    min_topic_presence: int = 50,
    apply_pca_to_confounders: bool = True,
    pca_variance_threshold: float = 0.95,
    pca_max_components: int = 50,
    normalize_treatment: bool = True,
    n_bootstrap: int = 200,
    random_state: int = 42,
    verbose: bool = True,
    alpha: float = 0.05,
    n_jobs: int = 1,
    max_topics: int | None = None,
) -> Dict:
    """
    Multi-treatment DML: estimate the causal effect of every topic sentiment
    on ideology simultaneously, controlling for all other topic sentiments and
    topic-presence indicators.

    Uses full-dataset N (not filtered per topic) for maximum power. Each
    sentiment column is used as-is — the presence indicators in W absorb the
    "topic absent ↔ sentiment = 0" variation, so the DML isolates within-
    presence sentiment variation.

    Args:
        df: Full dataset with 'topic <name>', 'sentiment <name>', and
            ``outcome_col`` columns.
        outcome_col: Column name for the outcome variable (default 'int_bias').
        min_topic_presence: Minimum number of articles in which a topic must
            appear to include its sentiment as a treatment column.
        apply_pca_to_confounders: Reduce W (topic presence matrix) via PCA.
        pca_variance_threshold: Variance fraction retained after PCA.
        pca_max_components: Hard cap on PCA components.
        normalize_treatment: Standardise each T column to zero mean, unit
            variance before fitting so ATEs are on a common scale. ATE is then
            "effect of a 1-SD increase in the raw sentiment score".
        n_bootstrap: Number of bootstrap resamples for confidence intervals.
        random_state: Random seed.
        verbose: Print progress information.
        alpha: Significance level for confidence intervals.

    Returns:
        Dict with keys:
          'ate_table'          — pd.DataFrame sorted by |ATE|, columns:
                                 topic, ate, ci_lower, ci_upper, significant,
                                 n_topic_articles, sentiment_col
          'n_samples'          — number of rows in df
          'n_treatments'       — number of treatment columns used
          'active_sentiments'  — list of sentiment column names included
          'normalize_treatment'  — whether T was standardised
          'T_mean', 'T_std'    — pre-normalisation statistics (for back-transform)
    """
    # ------------------------------------------------------------------ #
    # 1. Select topics with sufficient presence                           #
    # ------------------------------------------------------------------ #
    topic_cols = sorted(c for c in df.columns if c.startswith('topic '))
    topic_presence = df[topic_cols].sum()

    active_topic_cols = [
        c for c in topic_cols
        if topic_presence[c] >= min_topic_presence
        and c.replace('topic ', 'sentiment ') in df.columns
    ]

    # Limit to top-N topics by article count
    if max_topics is not None and len(active_topic_cols) > max_topics:
        active_topic_cols = sorted(
            active_topic_cols, key=lambda c: topic_presence[c], reverse=True
        )[:max_topics]
        if verbose:
            print(f"Limiting to top {max_topics} topics by article count")

    active_sentiment_cols = [c.replace('topic ', 'sentiment ') for c in active_topic_cols]

    if verbose:
        print(f"Topics with ≥{min_topic_presence} articles: {len(active_topic_cols)} / {len(topic_cols)}")
        print(f"N articles (full dataset): {len(df)}")
        print(f"Treatment mode: continuous (raw score)")

    if len(active_sentiment_cols) < 2:
        raise ValueError(
            f"Need at least 2 treatment columns; only {len(active_sentiment_cols)} topics "
            f"have ≥{min_topic_presence} articles."
        )

    # ------------------------------------------------------------------ #
    # 2. Build Y, T, W matrices                                           #
    # ------------------------------------------------------------------ #
    # Drop rows where outcome is NaN (e.g. GPT data may lack int_bias for
    # some articles).  Filter df so T and W stay aligned with Y.
    valid_mask = df[outcome_col].notna()
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"  WARNING: {n_dropped} NaN value(s) in '{outcome_col}' — dropping those rows "
              f"({len(df) - n_dropped} of {len(df)} remain)")
        df = df[valid_mask].reset_index(drop=True)

    Y = df[outcome_col].values.astype(np.float64)

    # fillna(0) on sentiment columns: a NaN score means the topic was not
    # present in the article, so its sentiment is undefined.  Imputing 0
    # (neutral) is consistent with the causal model — sentiment is defined
    # as 0 whenever the associated topic indicator is False.  The topic
    # presence indicators in W absorb the "topic absent" variation, so the
    # DML residualises out that source of correlation before using T.
    T_raw = df[active_sentiment_cols].fillna(0).values.astype(np.float64)

    T_mean = T_raw.mean(axis=0)
    T_std = T_raw.std(axis=0)
    T_std[T_std < 1e-8] = 1.0          # avoid divide-by-zero for constant columns
    T = (T_raw - T_mean) / T_std if normalize_treatment else T_raw.copy()

    # fillna(0): a NaN in a topic-presence indicator means the topic was not
    # tagged for this article — semantically equivalent to absent (0).
    W = df[active_topic_cols].fillna(0).values.astype(np.float64)

    # ------------------------------------------------------------------ #
    # 3. PCA on W                                                         #
    # ------------------------------------------------------------------ #
    if apply_pca_to_confounders and W.shape[1] > 1:
        pca_probe = PCA(n_components=pca_variance_threshold)
        pca_probe.fit(W)
        n_comp = min(pca_probe.n_components_, pca_max_components)
        pca = PCA(n_components=n_comp)
        W = pca.fit_transform(W)
        if verbose:
            var = pca.explained_variance_ratio_.sum()
            print(f"PCA on W: {len(active_topic_cols)} → {n_comp} components ({var:.3f} variance)")

    n_samples, k = T.shape

    # ------------------------------------------------------------------ #
    # 4. Fit multi-treatment LinearDML (point estimates)                  #
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\nFitting LinearDML with {k} treatments …")

    est = LinearDML(
        model_y=RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            random_state=random_state, n_jobs=n_jobs,
        ),
        model_t=RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=10,
            random_state=random_state + 1, n_jobs=n_jobs,
        ),
        random_state=random_state
    )
    est.fit(Y, T, W=W)

    # Point-estimate ATE for each treatment: effect of +1 unit (= 1 SD if normalised)
    T0_base = np.zeros((n_samples, k))
    ates = np.zeros(k)
    for i in range(k):
        T1_i = np.zeros((n_samples, k))
        T1_i[:, i] = 1.0
        ates[i] = est.effect(T0=T0_base, T1=T1_i).mean()

    # ------------------------------------------------------------------ #
    # 5. Bootstrap confidence intervals                                    #
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"Bootstrapping CIs ({n_bootstrap} iterations) …")

    rng = np.random.default_rng(random_state)
    # Pre-draw per-iteration seeds so workers stay deterministic and independent.
    boot_seeds = rng.integers(0, 1_000_000, size=(n_bootstrap, 4))

    def _one_boot(seeds):
        b_rng = np.random.default_rng(int(seeds[0]))
        idx = b_rng.choice(n_samples, size=n_samples, replace=True)
        T_b, Y_b, W_b = T[idx], Y[idx], W[idx]
        try:
            est_b = LinearDML(
                model_y=RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_leaf=10,
                    random_state=int(seeds[1]), n_jobs=1,
                ),
                model_t=RandomForestRegressor(
                    n_estimators=100, max_depth=5, min_samples_leaf=10,
                    random_state=int(seeds[2]), n_jobs=1,
                ),
                random_state=int(seeds[3]),
            )
            est_b.fit(Y_b, T_b, W=W_b)
            T0_b = np.zeros((n_samples, k))
            row = np.zeros(k)
            for i in range(k):
                T1_i = np.zeros((n_samples, k))
                T1_i[:, i] = 1.0
                row[i] = est_b.effect(T0=T0_b, T1=T1_i).mean()
            return row
        except Exception:
            return np.full(k, np.nan)

    if n_jobs == 1:
        boot_list = []
        for b in range(n_bootstrap):
            if verbose and (b + 1) % 50 == 0:
                print(f"  Bootstrap {b + 1}/{n_bootstrap}")
            boot_list.append(_one_boot(boot_seeds[b]))
    else:
        if verbose:
            print(f"  Parallel bootstrap with n_jobs={n_jobs}")
        boot_list = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_one_boot)(boot_seeds[b]) for b in range(n_bootstrap)
        )
    boot_ates = np.vstack(boot_list)

    valid_boots = boot_ates[~np.isnan(boot_ates).any(axis=1)]
    if verbose:
        print(f"  Valid bootstrap iterations: {len(valid_boots)}/{n_bootstrap}")

    if len(valid_boots) >= 10:
        ci_lower = np.percentile(valid_boots, 100 * alpha / 2, axis=0)
        ci_upper = np.percentile(valid_boots, 100 * (1 - alpha / 2), axis=0)
    else:
        warnings.warn("Too few valid bootstrap iterations; CIs unavailable.")
        ci_lower = np.full(k, np.nan)
        ci_upper = np.full(k, np.nan)

    # ------------------------------------------------------------------ #
    # 6. Assemble results table                                           #
    # ------------------------------------------------------------------ #
    topic_names = [c.replace('sentiment ', '') for c in active_sentiment_cols]
    rows = []
    for i, name in enumerate(topic_names):
        significant = bool(ci_lower[i] > 0 or ci_upper[i] < 0) if not np.isnan(ci_lower[i]) else False
        rows.append({
            'topic': name,
            'sentiment_col': active_sentiment_cols[i],
            'ate': float(ates[i]),
            'ci_lower': float(ci_lower[i]),
            'ci_upper': float(ci_upper[i]),
            'significant': significant,
            'n_topic_articles': int(topic_presence[f'topic {name}']),
        })

    ate_table = (
        pd.DataFrame(rows)
        .assign(abs_ate=lambda d: d['ate'].abs())
        .sort_values('abs_ate', ascending=False)
        .drop(columns='abs_ate')
        .reset_index(drop=True)
    )

    return {
        'ate_table': ate_table,
        'estimator': est,
        'n_samples': n_samples,
        'n_treatments': k,
        'active_sentiments': active_sentiment_cols,
        'normalize_treatment': normalize_treatment,
        'T_mean': T_mean,
        'T_std': T_std,
    }
