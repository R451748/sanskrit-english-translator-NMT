import pandas as pd

TRAIN = "dataset/processed/train.csv"
VAL = "dataset/processed/validation.csv"
TEST = "dataset/processed/test.csv"

train = pd.read_csv(TRAIN)
val = pd.read_csv(VAL)
test = pd.read_csv(TEST)

train_sanskrit = set(train["sanskrit"])
val_sanskrit = set(val["sanskrit"])
test_sanskrit = set(test["sanskrit"])

train_val = train_sanskrit & val_sanskrit
train_test = train_sanskrit & test_sanskrit
val_test = val_sanskrit & test_sanskrit

print("=" * 70)
print("DATASET LEAKAGE ANALYSIS")
print("=" * 70)

print("Train rows:", len(train))
print("Validation rows:", len(val))
print("Test rows:", len(test))

print("\nSanskrit sentence overlap:")
print("Train ∩ Validation:", len(train_val))
print("Train ∩ Test:", len(train_test))
print("Validation ∩ Test:", len(val_test))

print("\nExamples of overlapping sentences:")

if train_val:
    print("\nTrain / Validation:")
    for x in list(train_val)[:5]:
        print(x)

if train_test:
    print("\nTrain / Test:")
    for x in list(train_test)[:5]:
        print(x)

if val_test:
    print("\nValidation / Test:")
    for x in list(val_test)[:5]:
        print(x)

print("\n" + "=" * 70)