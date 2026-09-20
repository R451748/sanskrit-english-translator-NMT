import re


# Sanskrit morphology tags observed from your local morphology model.
# We only act when the tag is clearly identifiable.
TENSE_MAP = {
    # Present
    "SPr1In": "present",
    "SPr2In": "present",
    "SPr3In": "present",
    "SPr1Ip": "present",
    "SPr2Ip": "present",
    "SPr3Ip": "present",
    "SPr1Is": "present",
    "SPr2Is": "present",
    "SPr3Is": "present",

    # Imperfect / past
    "SIm1In": "past",
    "SIm2In": "past",
    "SIm3In": "past",
    "SIm1Ip": "past",
    "SIm2Ip": "past",
    "SIm3Ip": "past",

    # Perfect
    "SPrf1In": "past",
    "SPrf2In": "past",
    "SPrf3In": "past",

    # Aorist
    "SAor1In": "past",
    "SAor2In": "past",
    "SAor3In": "past",

    # Future
    "SFu1In": "future",
    "SFu2In": "future",
    "SFu3In": "future",
    "SFu1Ip": "future",
    "SFu2Ip": "future",
    "SFu3Ip": "future",

    # Conditional
    "SCo1In": "conditional",
    "SCo2In": "conditional",
    "SCo3In": "conditional",

    # Imperative
    "SImv1In": "imperative",
    "SImv2In": "imperative",
    "SImv3In": "imperative",

    # Optative / potential
    "SOp1In": "optative",
    "SOp2In": "optative",
    "SOp3In": "optative",
}


def extract_tags(morphology_text):
    """Extract morphology tags from SLM output."""

    tags = []

    for token in morphology_text.split():

        parts = token.split("_")

        if len(parts) >= 3:

            tag = parts[-1].strip()

            if tag in TENSE_MAP:
                tags.append(tag)

    return tags


def detect_tense(morphology_text):
    """
    Detect the dominant tense/mood.

    Returns:
        present
        past
        future
        conditional
        imperative
        optative
        unknown
    """

    tags = extract_tags(morphology_text)

    if not tags:
        return "unknown"

    categories = {}

    for tag in tags:
        category = TENSE_MAP[tag]
        categories[category] = categories.get(category, 0) + 1

    return max(categories, key=categories.get)


if __name__ == "__main__":

    examples = {

        "Present":
        "rāmaḥ_rāma_SNM vana_vana_SANe gacchati_gam_SPr3In",

        "Past":
        "rāmaḥ_rāma_SNM vana_vana_SANe agacchat_gam_SIm3In",

    }

    for name, morphology in examples.items():

        print(
            name,
            "=>",
            detect_tense(morphology)
        )