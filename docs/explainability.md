# Explainability

## Current attribution

The results UI displays selected-feature Shapley values for the predicted class's
**weighted ensemble probability**. For a selected symptom set `S`, the reference
input is the all-zero symptom vector. For each coalition `A` within `S`, only
the symptoms in `A` are present; all other features are zero. The coalition
value is:

```text
v(A) = 0.675 * P_XGBoost(class | A) + 0.325 * P_CatBoost(class | A)
```

Each component probability is selected using that model's own `classes_` order.
The feature value is the standard Shapley allocation over all subsets:

```text
phi_i = sum over A subset S\{i} of
        |A|! (|S|-|A|-1)! / |S|! * (v(A union {i}) - v(A))
```

The component attributions use the same coalition game and reference. By
linearity of Shapley values, `phi_i = 0.675 * phi_i_XGBoost + 0.325 *
phi_i_CatBoost`. With exact enumeration this is an exact attribution of the
ensemble probability, not a juxtaposition of unrelated model scores. The
permutation path is shared across both components, so their weighted estimates
also equal the ensemble estimate feature-by-feature. Values are probability
changes (the UI formats them as percentage points), and their sum is the
ensemble prediction minus the all-zero reference probability.

All coalitions are enumerated when at most 12 symptoms are selected. Larger
inputs use 128 deterministic sampled permutations. The sampled allocation is
approximate, but the per-permutation changes telescope, so its attributions
still sum to the full prediction-minus-reference difference.

This is model feature attribution under an explicit zero-symptom reference. It
does not show that a symptom caused a disease, describe clinical mechanisms, or
provide a clinical explanation. The zero-symptom reference is a computational
baseline, not a claim about a realistic patient.

## Verification of the previous component explanations

The previous XGBoost implementation used `shap.TreeExplainer` with
`feature_perturbation="tree_path_dependent"` and `model_output="raw"`. Its
multiclass output was indexed at the predicted class and was in XGBoost raw
margin space, not probability space. For the Headache/Nausea/Fatigue example,
the selected class was index 29; the selected feature values plus the expected
value reproduced the XGBoost output margin (2.2324786 versus 2.2324781).

The previous CatBoost implementation used native
`get_feature_importance(type="ShapValues")`. Its multiclass output had shape
`(1, 30, 175)`: one row, 30 class scores, and 174 feature values plus one
expected value. It selected predicted class 29 and dropped that class's final
expected-value slot. The selected contributions plus that slot reproduced
CatBoost's raw score (approximately zero). They were valid raw-score values,
but not probabilities and not directly comparable to the XGBoost values.

For the same three symptoms, CatBoost's class-29 probability moved from
6.10% at the all-zero reference to 22.64%, while its class-29 raw score remained
approximately zero. In a multiclass model the probability depends on all class
scores through softmax, so a nearly unchanged score for one class does not mean
that its probability or the ensemble probability was unchanged. The current
endpoint now computes all component and ensemble values in probability space.

## Model and evaluation integrity

This explainability change is inference-only. It does not retrain, tune, or
modify either saved model, the ensemble weights, or final test evaluation. No
held-out test data is used by the explanation method or its synthetic tests.
