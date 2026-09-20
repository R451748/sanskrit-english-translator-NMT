import re


# Confirmed from your actual Sanskrit morphology model
TENSE_MAP = {
    "SPr3In": "present",
    "SPs3In": "past",
    "SFu3In": "future",
    "SPr2Im": "imperative",
    "SPr3O": "optative",
}


def extract_tense(morphology_output):
    """Extract the first recognized Sanskrit verbal tense/mood."""

    for token in morphology_output.split():

        parts = token.split("_")

        if len(parts) >= 3:

            tag = parts[-1].strip()

            if tag in TENSE_MAP:
                return TENSE_MAP[tag]

    return "unknown"


def detect_english_tense(translation):
    """
    Conservative English tense detector.

    Returns:
        present
        past
        future
        imperative
        optative
        unknown
    """

    text = translation.lower().strip()

    # --------------------------------------------------
    # FUTURE
    # --------------------------------------------------

    if re.search(
        r"\b(will|shall|will be|shall be)\b",
        text
    ):
        return "future"

    # --------------------------------------------------
    # PAST
    # --------------------------------------------------

    past_words = {
        # be
        "was",
        "were",

        # have
        "had",

        # common irregular verbs
        "went",
        "came",
        "said",
        "saw",
        "gave",
        "taught",
        "spoke",
        "took",
        "made",
        "did",
        "knew",
        "found",
        "became",
        "began",
        "heard",
        "lived",

        # common regular past forms
        "walked",
        "visited",
        "stayed",
        "went",
        "studied",
        "learned",
        "learned",
        "performed",
        "completed",
        "followed",
        "returned",
        "reached",
        "entered",
        "left",
    }

    words = set(re.findall(r"\b[a-z]+\b", text))

    if words.intersection(past_words):
        return "past"

    # --------------------------------------------------
    # PRESENT
    # --------------------------------------------------

    present_words = {
        # be
        "is",
        "are",
        "am",

        # have
        "has",
        "have",

        # common verbs
        "go",
        "goes",
        "come",
        "comes",
        "say",
        "says",
        "see",
        "sees",
        "give",
        "gives",
        "teach",
        "teaches",
        "speak",
        "speaks",
        "live",
        "lives",
        "take",
        "takes",
        "make",
        "makes",
        "do",
        "does",
        "know",
        "knows",
        "find",
        "finds",
        "become",
        "becomes",
        "begin",
        "begins",
        "hear",
        "hears",
        "walk",
        "walks",
        "visit",
        "visits",
        "stay",
        "stays",
        "study",
        "studies",
        "learn",
        "learns",
        "perform",
        "performs",
        "complete",
        "completes",
        "follow",
        "follows",
        "return",
        "returns",
        "reach",
        "reaches",
        "enter",
        "enters",
        "leave",
        "leaves",
    }

    if words.intersection(present_words):
        return "present"

    # --------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------

    return "unknown"


def check_tense(morphology_output, translation):

    expected = extract_tense(morphology_output)

    actual = detect_english_tense(translation)

    # Do not flag anything when either side is uncertain.
    if expected == "unknown" or actual == "unknown":

        return {
            "expected": expected,
            "detected": actual,
            "mismatch": False
        }

    return {
        "expected": expected,
        "detected": actual,
        "mismatch": expected != actual
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    tests = [

        # Correct present
        (
            "rāmaḥ_rāma_SNM gacchati_gam_SPr3In",
            "Rama goes to the forest."
        ),

        # Incorrect present -> past
        (
            "rāmaḥ_rāma_SNM gacchati_gam_SPr3In",
            "Rama went to the forest."
        ),

        # Correct past
        (
            "rāmaḥ_rāma_SNM agacchat_gam_SPs3In",
            "Rama went to the forest."
        ),

        # Correct future
        (
            "rāmaḥ_rāma_SNM gamiṣyati_gam_SFu3In",
            "Rama will go to the forest."
        ),

        # Present with third-person verb
        (
            "rāmaḥ_rāma_SNM nivasati_nivas_SPr3In",
            "Rama lives there for many years."
        ),

        # Present incorrectly translated as past
        (
            "rāmaḥ_rāma_SNM nivasati_nivas_SPr3In",
            "Rama lived there for many years."
        ),

        # Present - teaches
        (
            "kṛṣṇaḥ_kṛṣṇa_SNM upadiśati_upadiś_SPr3In",
            "Krishna teaches Arjuna."
        ),

        # Past - taught
        (
            "kṛṣṇaḥ_kṛṣṇa_SNM upadiśat_upadiś_SPs3In",
            "Krishna taught Arjuna."
        ),
    ]


    print("=" * 70)
    print("IMPROVED TENSE CONSISTENCY TEST")
    print("=" * 70)

    for morphology, translation in tests:

        result = check_tense(
            morphology,
            translation
        )

        print("\nTranslation:", translation)
        print("Expected   :", result["expected"])
        print("Detected   :", result["detected"])
        print("Mismatch   :", result["mismatch"])

    print("\n" + "=" * 70)