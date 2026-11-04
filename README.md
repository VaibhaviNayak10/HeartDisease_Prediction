# Heart Disease Prediction: Applied Probability and Statistics Project

## Project objective
Analyze the UCI Heart Disease data from four sources (Cleveland, Hungarian,
Switzerland and VA Long Beach), quantify statistical patterns and probabilities,
and compare machine-learning classifiers for the binary outcome `num > 0`.

**Educational analysis only. This is not a clinical diagnostic system.**

## Dataset
Place the four original UCI files in `Dataset/`:
- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

Each file contains 13 patient attributes and the original target `num`.
The original target is retained. For binary classification, `num == 0` is
coded as no disease and `num > 0` as disease present. The source/hospital label
is retained for subgroup analysis and is not used as a model predictor.

## Analyses included
1. Data integration, missingness and zero-value audit.
2. Descriptive statistics: mean, median, variance, standard deviation, quartiles
   and 95% confidence intervals.
3. Empirical marginal, joint and conditional probabilities.
4. Bayes' theorem using observed priors and likelihoods.
5. Inferential statistics:
   - Chi-square tests for hospital/feature association with disease status.
   - Welch independent-samples t-tests for numerical features.
   - Mann-Whitney U tests as rank-based companion analyses.
   - Holm adjustment across the family of exploratory tests.
6. Visualization: target distribution, hospital prevalence with Wilson intervals,
   feature histograms, boxplots, correlation heatmap and Bayesian prior/posterior.
7. Machine learning: Logistic Regression, Random Forest, Decision Tree and KNN.
8. Evaluation: accuracy, precision, recall, specificity, F1, ROC-AUC and
   confusion matrices. Existing paired bootstrap AUC comparison and McNemar
   test compare Logistic Regression with Random Forest.

Generated tables are saved in `results/statistical_results/`; plots are saved
in `results/plots/`; fitted Logistic Regression and Random Forest artifacts
are saved in `saved_models/`.

## Important methodological notes
- Imputation and scaling for predictive modelling are fitted on training data
  only. The held-out test set is not used to fit the preprocessing steps.
- The hospital/source field is excluded from model predictors to reduce the
  risk of learning site identity instead of clinical signal.
- For descriptive/inferential analysis, missing values are not silently
  imputed. Each test uses available observations for the relevant variables.
- `trestbps == 0` and `chol == 0` are treated as implausible measurement codes
  and handled as missing for analyses/model preprocessing. Other zero values
  are not automatically treated as missing.
- The UCI data are historical and drawn from selected clinical populations.
  The combined disease proportion is not a population prevalence estimate.
  Hospital-wise differences can reflect referral, selection, measurement and
  case-mix differences, not necessarily differences in underlying risk.
- A high test score on this small, heterogeneous dataset does not establish
  clinical readiness. External validation is required before real-world use.
- The additional Decision Tree and KNN results are evaluated on the same
  preprocessed hold-out split for comparability. Model selection should ideally
  use cross-validation on training data, leaving the test set for one final
  evaluation.

## Folder structure
```text
Dataset/                     Original four UCI data files
src/                         Loading, preprocessing, analysis and modelling
saved_models/                Serialized fitted model artifacts
results/plots/                Generated figures
results/statistical_results/ Generated CSV summaries and interpretations
run_project.py                Full reproducible pipeline
requirements.txt              Python dependencies
```
