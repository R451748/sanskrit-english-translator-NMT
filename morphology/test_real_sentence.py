from tense_detector import detect_tense
from tense_checker import check_tense


morphology = """
rāmaḥ_rāma_SNM
sītā_sītā_SNM
lakṣmaṇena_lakṣmaṇa_SIM
vana_vana_SANe
nivasati_nivas_SPr3In
"""


translation = (
    "Rama went to the forest with Sita and Lakshmana "
    "and lived there for many years."
)


print("=" * 70)
print("REAL SENTENCE TENSE CHECK")
print("=" * 70)

print("Expected Sanskrit tense:",
      detect_tense(morphology))

result = check_tense(
    morphology,
    translation
)

print("English detected tense:", result["detected"])
print("Mismatch:", result["mismatch"])