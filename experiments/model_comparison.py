from pathlib import Path
import csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "model_comparison.csv"
)


results = [
    {
        "Model": "Transformer Baseline",
        "Best_Validation_Loss": 5.1510,
        "BLEU": 1.3252,
        "chrF": 18.5463,
        "Repetition_Rate": 0.00
    },
    {
        "Model": "V1 Global Morphology",
        "Best_Validation_Loss": 5.1397,
        "BLEU": 0.9179,
        "chrF": 14.8867,
        "Repetition_Rate": 0.00
    },
    {
        "Model": "V2 Morphology Sequence",
        "Best_Validation_Loss": 5.0067,
        "BLEU": 2.6487,
        "chrF": 20.2512,
        "Repetition_Rate": 0.00
    }
]


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Model",
            "Best_Validation_Loss",
            "BLEU",
            "chrF",
            "Repetition_Rate"
        ]
    )

    writer.writeheader()

    writer.writerows(results)


print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    f"{'Model':35}"
    f"{'Val Loss':>12}"
    f"{'BLEU':>12}"
    f"{'chrF':>12}"
    f"{'Repeat %':>12}"
)

print("-" * 83)

for result in results:

    print(
        f"{result['Model']:35}"
        f"{result['Best_Validation_Loss']:>12.4f}"
        f"{result['BLEU']:>12.4f}"
        f"{result['chrF']:>12.4f}"
        f"{result['Repetition_Rate']:>12.2f}"
    )


baseline = results[0]
v2 = results[2]

bleu_improvement = (
    (v2["BLEU"] - baseline["BLEU"])
    / baseline["BLEU"]
) * 100

chrf_improvement = (
    (v2["chrF"] - baseline["chrF"])
    / baseline["chrF"]
) * 100

loss_improvement = (
    (baseline["Best_Validation_Loss"]
     - v2["Best_Validation_Loss"])
    / baseline["Best_Validation_Loss"]
) * 100


print("\n")
print("=" * 70)
print("V2 vs BASELINE")
print("=" * 70)

print(
    f"Validation loss change: "
    f"{loss_improvement:.2f}%"
)

print(
    f"BLEU change: "
    f"{bleu_improvement:.2f}%"
)

print(
    f"chrF change: "
    f"{chrf_improvement:.2f}%"
)

print("\nSaved:")
print(OUTPUT_FILE)