# Climate, dairy productivity and methane emissions

Data and analysis code for the methods in *Supplementary Materials*.

## Structure

```text
.
├── data/
│   ├── Historical_milk_climate_methane.xlsx
│   └── Other_exposures.xlsx
├── data_sources.py                  # Data sources
├── data_processing.py               # Aggregation and transformations
├── historical_associations.R        # Long-term and single-variable models
├── annual_panel_counterfactual.py   # Annual panel and counterfactuals
├── bayesian_models.R                # Domain models and sensitivity analyses
├── shapley.R                        # Six-domain Shapley decomposition
├── structural_equations.R           # SEM and sensitivity analyses
├── predictive_sensitivity.py        # Grouped prediction and SHAP
├── climate_processing.py            # Climate aggregation and calibration
├── future_projections.py            # Climate, productivity and demand scenarios
├── common.py                        # Input and output helpers
├── panel.py                         # Fixed effects and clustered inference
├── hierarchical.stan                # Bayesian model
├── requirements.txt
└── README.md
```

## Data

Each workbook includes variable definitions and source information. The three files contain historical observations, production-context variables, and future climate and population inputs, respectively. Population data cover 26 states and the Federal District; the projection function sums these to obtain the national demand multiplier specified in the methods.

## Use

Install Python dependencies with `pip install -r requirements.txt`. R analyses use R 4.3.2, CmdStanR 0.9.0, CmdStan 2.40.0, posterior 1.6.1 and lavaan 0.6-19.

Python scripts expose functions for data preparation and analysis; read Excel sheets with `common.read_table(path, sheet_name=...)`. Annual-panel command-line runs take a JSON configuration specifying `annual_panel` (prepared CSV) and `output`, plus `--seed` and `--fractions`. R routines use prepared CSV inputs:

```sh
Rscript historical_associations.R amc_means.csv results/historical
Rscript bayesian_models.R context.csv results/domains domains 1234
Rscript shapley.R context.csv results/shapley 1234
```

For Bayesian sensitivity analyses, replace `domains` with `loso`, `drop` or `supplementary`. SEM functions are loaded with `source("structural_equations.R")`. Supply transformed variables, the model specification and an explicit random seed to the corresponding analysis functions.
