# Climate, dairy productivity and methane emissions

Data and analysis code for the methods in *Supplementary Materials*.

## Structure

```text
.
├── data/
│   ├── Historical_milk_climate_methane.xlsx
│   ├── Other_exposures.xlsx
│   └── Future_climate_population.xlsx
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
