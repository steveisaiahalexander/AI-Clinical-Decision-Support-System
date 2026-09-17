from src.data.dataset import DatasetLoader
from src.core.constants import TARGET_COLUMN


# =====================================================
# LOAD DATASET
# =====================================================

train = DatasetLoader.load_training_dataset()


# =====================================================
# CLASS DISTRIBUTION
# =====================================================

distribution = (
    train[TARGET_COLUMN]
    .value_counts()
    .sort_values(ascending=True)
)


# =====================================================
# DISPLAY CLASS DISTRIBUTION
# =====================================================

print()

print("=" * 70)
print("DISEASE CLASS DISTRIBUTION")
print("=" * 70)

print(
    distribution.to_string()
)


# =====================================================
# SUMMARY
# =====================================================

print()

print("=" * 70)
print("CLASS SUPPORT SUMMARY")
print("=" * 70)

print(
    f"Total classes : {len(distribution)}"
)

print(
    f"Total samples : {len(train)}"
)


# =====================================================
# DIFFERENT SUPPORT THRESHOLDS
# =====================================================

thresholds = [2, 5, 10, 20, 50, 100]


print()

print("=" * 70)
print("MINIMUM CLASS SUPPORT ANALYSIS")
print("=" * 70)

for threshold in thresholds:

    classes_below = (
        distribution[
            distribution < threshold
        ]
    )

    classes_at_or_above = (
        distribution[
            distribution >= threshold
        ]
    )

    print()

    print(
        f"Threshold >= {threshold}"
    )

    print(
        f"Usable classes : "
        f"{len(classes_at_or_above)}"
    )

    print(
        f"Excluded classes: "
        f"{len(classes_below)}"
    )

    print(
        f"Samples retained: "
        f"{classes_at_or_above.sum()}"
    )

    print(
        f"Samples excluded: "
        f"{classes_below.sum()}"
    )


# =====================================================
# RARE CLASSES
# =====================================================

rare_threshold = 20

rare_classes = (
    distribution[
        distribution < rare_threshold
    ]
)


print()

print("=" * 70)
print(
    f"RARE CLASSES (< {rare_threshold} SAMPLES)"
)
print("=" * 70)

if rare_classes.empty:

    print("No rare classes found.")

else:

    print(
        rare_classes.to_string()
    )


# =====================================================
# WELL-SUPPORTED CLASSES
# =====================================================

supported_classes = (
    distribution[
        distribution >= rare_threshold
    ]
)


print()

print("=" * 70)
print(
    f"SUPPORTED CLASSES (>= {rare_threshold} SAMPLES)"
)
print("=" * 70)

print(
    supported_classes.to_string()
)


# =====================================================
# FINAL SUMMARY
# =====================================================

print()

print("=" * 70)
print("FINAL CLASS ANALYSIS")
print("=" * 70)

print(
    "Original classes :",
    len(distribution)
)

print(
    "Rare classes     :",
    len(rare_classes)
)

print(
    "Supported classes:",
    len(supported_classes)
)

print(
    "Rare samples     :",
    rare_classes.sum()
)

print(
    "Supported samples:",
    supported_classes.sum()
)