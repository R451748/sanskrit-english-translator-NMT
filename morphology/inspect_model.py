from transformers import AutoTokenizer, AutoConfig

MODEL_NAME = "chronbmm/sanskrit5-multitask"

print("=" * 70)
print("SANSKRIT5 MULTITASK MODEL INSPECTION")
print("=" * 70)

print("\nLoading configuration...")

config = AutoConfig.from_pretrained(MODEL_NAME)

print("\nMODEL CONFIGURATION")
print("-" * 70)

print(config)

print("\n" + "=" * 70)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("\nTOKENIZER SPECIAL TOKENS")
print("-" * 70)

print("PAD:", tokenizer.pad_token)
print("UNK:", tokenizer.unk_token)
print("BOS:", tokenizer.bos_token)
print("EOS:", tokenizer.eos_token)

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)