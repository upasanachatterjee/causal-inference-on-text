# Causal Inference for Sentiment → Ideology

Causal inference research code from a thesis project investigating how
topic-specific sentiment in news articles (`F_T`) causally affects the political
ideology classification of the article (`Y`), using the AllSides dataset.

The codebase supports:
- **ATE / CATE** estimation with bootstrap confidence intervals
- **Mediation analysis** (NDE / NIE) via DML
- **Multi-treatment DML** — simultaneous causal effects of every topic
  sentiment on ideology
- **Community-level analysis** — group topics into communities and treat each
  community as a single (treatment, mediator) unit

Treatment is always the **raw continuous sentiment score** (range −100…+100),
standardised to unit variance where appropriate. ATEs are interpreted as the
effect of a 1-SD increase in sentiment on the ideology score.

## Causal framework

- `X` (collider): article text — never condition on it
- `T` (confounder): boolean topic-presence indicators
- `F` (sentiment): continuous score per topic; 0 if topic tag absent
- `Y` (outcome): ideology label, `int_bias` column (0=Left, 1=Center, 2=Right)
- **Treatment** `F_T`: sentiment toward the topic of interest
- **Mediators** `F_M1, F_M2, ...`: sentiments toward the *other* topics
- **Confounders**: topic-presence indicators (PCA-reduced to 95% variance by default)

## Setup

```bash
pip install -r requirements.txt
```

Datasets are expected under `data/`

## Example scripts

All entry points live in `scripts/`. Each writes a per-case directory with a
log file, CSV results, and forest / mediation plots.

### `scripts/mediation_analysis.py`

Single-topic mediation analysis. Decomposes the total effect of a treatment
topic's sentiment on ideology into NDE (direct) + NIE (mediated through other
sentiments).

```bash
# default: hardcoded topic pairs (Donald Trump ↔ Politics) on every data/*.pkl
python scripts/mediation_analysis.py

# specific dataset
python scripts/mediation_analysis.py data/human_random_test_clean.pkl

# community-level mediation: community 1 as treatment, community 2 as mediator
python scripts/mediation_analysis.py data/human_random_test_clean.pkl \
    --communities communities_full_dataset_min_weight_2.json \
    --community-cases 1:2 \
    --communities-only \
    --output-root results/community_mediation \
    --min-topic-presence 50 \
    --pca-variance 0.95 \
    --n-bootstrap 2000
```

Useful flags: `--min-topic-presence`, `--pca-variance`, `--n-bootstrap`,
`--communities PATH`, `--community-cases T:M …`, `--communities-only`.

### `scripts/multi_treatment_dml_sweep.py`

Estimates the ATE of every topic sentiment simultaneously using a multivariate
LinearDML.

```bash
# all data/*.pkl with defaults
python scripts/multi_treatment_dml_sweep.py

# explicit dataset, community-level analysis included
python scripts/multi_treatment_dml_sweep.py data/human_random_test_clean.pkl \
    --output-root results/multi_treatment \
    --communities communities_full_dataset_min_weight_2.json \
    --min-presence 50 \
    --n-bootstrap 2000 \
    --max-topics 20
```

## Recreating the community mediation sweep

The full sweep (community mediation across many pairs and datasets) corresponds
to the following loop over community pairs `(T, M)`. For each pair, run
`mediation_analysis.py` with `--community-cases T:M --communities-only`:

```bash
DATASETS=(data/human_random_test_clean.pkl data/best_random_bias_test_clean.pkl)
COMMUNITIES=communities_full_dataset_min_weight_2.json
PAIRS=("1:2" "2:1" "1:0" "0:1" "0:2" "2:0" "4:1" "1:4" "4:2" "2:4")
RESULTS_ROOT=results_full_dataset_min_weight_2

for DATASET in "${DATASETS[@]}"; do
    TAG=$(basename "${DATASET%.*}")
    for PAIR in "${PAIRS[@]}"; do
        T="${PAIR%%:*}"; M="${PAIR##*:}"
        OUT="${RESULTS_ROOT}/community_mediation_analysis_min_pres_50/${TAG}/results_analysis_min_pres_50_with_communities_${T}_${M}"
        python scripts/mediation_analysis.py "$DATASET" \
            --output-root "$OUT" \
            --min-topic-presence 50 \
            --pca-variance 0.95 \
            --communities "$COMMUNITIES" \
            --community-cases "${T}:${M}" \
            --communities-only \
            --n-bootstrap 2000
    done
done
```

Optional follow-up runs:

```bash
# Multi-treatment DML, topic + community level
python scripts/multi_treatment_dml_sweep.py "$DATASET" \
    --output-root "${RESULTS_ROOT}/multi_treatment_min_pres_50/${TAG}" \
    --min-presence 50 --n-bootstrap 2000 --max-topics 20 \
    --communities "$COMMUNITIES"

# Topic-level mediation: Donald Trump → Politics
python scripts/mediation_analysis.py "$DATASET" \
    --output-root "${RESULTS_ROOT}/mediation_analysis/continuous_pca_95_min_comm_50/${TAG}" \
    --min-topic-presence 50 --pca-variance 0.95 --n-bootstrap 2000
```

`n_bootstrap=2000` is heavy (~30–90 min per run depending on N); use
`--n-bootstrap 100` while iterating, then bump it for final results. The
example scripts are independent — run them in any order or in parallel across
datasets and pairs.
