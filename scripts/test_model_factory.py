from src.models.model_factory import ModelFactory


print("=" * 60)
print("TESTING MODEL FACTORY")
print("=" * 60)

models = ModelFactory.get_all_models()

print()

for name, model in models.items():

    print(
        f"{name:<25} -> "
        f"{type(model).__name__}"
    )

print()

print("=" * 60)
print(f"TOTAL MODELS: {len(models)}")
print("=" * 60)