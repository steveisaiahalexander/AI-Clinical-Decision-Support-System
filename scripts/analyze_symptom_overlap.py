from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Training.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 80)
    print("SYMPTOM PATTERN / DISEASE OVERLAP ANALYSIS")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("\n[1/6] Loading dataset...")

    df = pd.read_csv(RAW_DATA)

    print(f"Dataset shape: {df.shape}")

    target = "disease"

    feature_columns = [
        column for column in df.columns
        if column != target
    ]

    X = df[feature_columns]
    y = df[target]

    print(f"Features: {len(feature_columns)}")
    print(f"Diseases: {y.nunique()}")
    print(f"Samples: {len(df)}")

    # ---------------------------------------------------------
    # 2. Create symptom signatures
    # ---------------------------------------------------------

    print("\n[2/6] Creating symptom signatures...")

    # Convert each patient's 174 symptoms into a single
    # binary string representing the complete symptom pattern.
    #
    # Example:
    # 001000100010...
    #
    # Identical signatures mean identical symptom vectors.

    signatures = X.astype(np.int8).astype(str).agg(
        "".join,
        axis=1,
    )

    df_analysis = pd.DataFrame({
        "signature": signatures,
        "disease": y.values,
    })

    unique_signatures = signatures.nunique()

    print(f"Total samples: {len(signatures)}")
    print(f"Unique symptom signatures: {unique_signatures}")

    duplicate_signature_count = (
        len(signatures) - unique_signatures
    )

    print(
        f"Samples belonging to repeated symptom signatures: "
        f"{duplicate_signature_count}"
    )

    # ---------------------------------------------------------
    # 3. Check identical symptoms -> multiple diseases
    # ---------------------------------------------------------

    print("\n[3/6] Checking conflicting symptom signatures...")

    signature_groups = (
        df_analysis
        .groupby("signature")["disease"]
        .agg(
            sample_count="size",
            unique_diseases="nunique",
            diseases=lambda x: sorted(x.unique().tolist()),
        )
        .reset_index()
    )

    conflicting = signature_groups[
        signature_groups["unique_diseases"] > 1
    ].copy()

    print(
        f"Conflicting symptom signatures: "
        f"{len(conflicting)}"
    )

    conflicting_samples = int(
        conflicting["sample_count"].sum()
    )

    print(
        f"Samples involved in conflicting signatures: "
        f"{conflicting_samples}"
    )

    # ---------------------------------------------------------
    # 4. Most ambiguous symptom patterns
    # ---------------------------------------------------------

    print("\n[4/6] Finding most ambiguous patterns...")

    if len(conflicting) > 0:

        conflicting["disease_count"] = (
            conflicting["diseases"].apply(len)
        )

        conflicting = conflicting.sort_values(
            [
                "unique_diseases",
                "sample_count",
            ],
            ascending=False,
        )

        top_conflicts = conflicting.head(20).copy()

        print("\nTop conflicting symptom patterns:")

        for _, row in top_conflicts.iterrows():

            print(
                f"\nSamples: {row['sample_count']}"
                f"\nDiseases: {row['unique_diseases']}"
                f"\nLabels: {row['diseases']}"
            )

    else:

        top_conflicts = pd.DataFrame()

        print(
            "\nNo identical symptom pattern was found "
            "with multiple disease labels."
        )

    # ---------------------------------------------------------
    # 5. Disease-level symptom overlap
    # ---------------------------------------------------------

    print("\n[5/6] Calculating disease-level symptom overlap...")

    # Average symptom vector for each disease.
    #
    # This gives us the proportion of patients in each disease
    # group exhibiting each symptom.

    disease_profiles = (
        df.groupby(target)[feature_columns]
        .mean()
    )

    diseases = disease_profiles.index.tolist()

    overlap_records = []

    for i in range(len(diseases)):

        for j in range(i + 1, len(diseases)):

            disease_a = diseases[i]
            disease_b = diseases[j]

            profile_a = disease_profiles.loc[disease_a].values
            profile_b = disease_profiles.loc[disease_b].values

            # Mean absolute difference between symptom
            # probabilities.
            mean_difference = np.mean(
                np.abs(profile_a - profile_b)
            )

            # Number of symptoms where both diseases have
            # relatively similar prevalence.
            similar_symptoms = np.sum(
                np.abs(profile_a - profile_b) < 0.10
            )

            overlap_records.append({
                "disease_a": disease_a,
                "disease_b": disease_b,
                "mean_profile_difference": float(
                    mean_difference
                ),
                "similar_symptoms_within_10_percent": int(
                    similar_symptoms
                ),
            })

    overlap_df = pd.DataFrame(overlap_records)

    overlap_df = overlap_df.sort_values(
        "mean_profile_difference"
    )

    print("\nMost similar disease profiles:")

    print(
        overlap_df.head(20).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 6. Save reports
    # ---------------------------------------------------------

    print("\n[6/6] Saving reports...")

    # Conflicting signatures
    conflicting_file = (
        REPORT_DIR /
        "conflicting_symptom_signatures.csv"
    )

    if len(conflicting) > 0:

        conflicting.to_csv(
            conflicting_file,
            index=False,
        )

    # Disease profile overlap
    overlap_file = (
        REPORT_DIR /
        "disease_profile_overlap.csv"
    )

    overlap_df.to_csv(
        overlap_file,
        index=False,
    )

    # Disease profiles
    profile_file = (
        REPORT_DIR /
        "disease_symptom_profiles.csv"
    )

    disease_profiles.to_csv(
        profile_file
    )

    # Summary
    summary = {
        "total_samples": int(len(df)),
        "total_features": int(len(feature_columns)),
        "total_diseases": int(y.nunique()),
        "unique_symptom_signatures": int(
            unique_signatures
        ),
        "repeated_signature_samples": int(
            duplicate_signature_count
        ),
        "conflicting_signature_count": int(
            len(conflicting)
        ),
        "conflicting_signature_samples": int(
            conflicting_samples
        ),
        "unique_signature_ratio": float(
            unique_signatures / len(df)
        ),
    }

    summary_file = (
        REPORT_DIR /
        "symptom_overlap_summary.json"
    )

    with open(summary_file, "w") as f:
        json.dump(
            summary,
            f,
            indent=4,
        )

    print("\nSaved:")
    print(conflicting_file)
    print(overlap_file)
    print(profile_file)
    print(summary_file)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()