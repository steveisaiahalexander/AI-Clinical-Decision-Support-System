# Probability Calibration and Abstention

## Data boundary

The 19,974-row held-out test set is not used by the calibration workflow. The
script reads only X_train.pkl and y_train.pkl, then makes five deterministic,
stratified partitions: 60% for disposable validation-model fits and 10% each
for temperature fitting, method selection, abstention-threshold selection,
and final development validation. Temporary validation models are not saved.
The locked XGBoost and CatBoost artifacts, their hyperparameters, and the
0.675 / 0.325 ensemble weights are not changed.

Because the deployed models were fitted on all of X_train, the validation
copies are trained on a smaller subset. The resulting temperature and policy
are therefore a development estimate transferred to the locked full-data
ensemble, not an independent evaluation of that exact fitted artifact.

## Calibration method

The candidate is single-parameter temperature scaling of the weighted ensemble
probabilities. For class probabilities p, it computes softmax(log(p) / T). The
temperature is fitted by minimizing multiclass negative log loss on the
calibration-fit partition. This one-parameter method preserves class ranking
and is less flexible than per-class calibration for the 30-class problem.

The method-selection partition compares raw and temperature-scaled outputs.
Temperature scaling is selected only when it lowers log loss on that partition;
otherwise inference retains the raw ensemble probabilities. The separate final
development-validation partition reports multiclass Brier score, log loss,
equal-width top-label ECE, and reliability bins for both outputs.

ECE is the sample-weighted absolute difference between accuracy and mean
confidence within 15 equal-width bins of the maximum class probability. It is
top-label ECE, not a classwise calibration guarantee.

## Current development results

The independent method-selection subset favored temperature scaling:
temperature 0.673404; raw versus scaled log loss 0.9480 versus 0.8630,
multiclass Brier 0.4306 versus 0.4039, and top-label ECE 0.1351 versus 0.0109.
On the separate 7,990-row development-validation partition, raw versus selected
output scored 0.4316 versus 0.4045 Brier, 0.9611 versus 0.8796 log loss, and
0.1348 versus 0.0171 top-label ECE.

The threshold-selection subset chose minimum top probability 0.1130, minimum
top-two margin 0.5554, and maximum normalized entropy 0.5092. Its selected
coverage was 50.01%. On the separate development-validation partition, coverage
was 48.76%, abstention rate 51.24%, accuracy among accepted cases 89.43%, and
top-1 accuracy before abstention 70.18%.

These metrics describe temporary models fitted on 60% of the training split.
Calibration settings are transferred to the locked models trained on all of
X_train, so these values are development estimates and may not match performance
of those exact artifacts.

## Uncertainty and abstention

For normalized class probabilities p:

- Top probability is max(p).
- Top-two margin is p(1) - p(2) after sorting probabilities descending.
- Entropy is -sum(p * log(p)).
- Normalized entropy is entropy divided by log(number of classes), in [0, 1].

The policy abstains when any selected condition fails: minimum top probability,
minimum top-two margin, or maximum normalized entropy. Thresholds are selected
on their own development partition by maximizing observed top-1 accuracy among
threshold combinations with at least 50% coverage. Results on the separate
development-validation partition report coverage, abstention rate, and
accuracy when accepted.

These thresholds are research interface settings, not clinically validated
cutoffs. Model probability, uncertainty, and abstention are not a diagnosis.
The calibration and abstention estimates also require future validation on a
separate, representative clinical dataset before clinical use.

## Reproduction

To reproduce validation-only calibration, run:

    .venv\Scripts\python.exe scripts\calibrate_final_ensemble.py --allow-validation-refits --device auto

The script writes configs/ensemble_calibration.json,
outputs/reports/ensemble_calibration_validation_metrics.json, and
outputs/figures/calibration_reliability.png. It never reads X_test.pkl or
y_test.pkl.
