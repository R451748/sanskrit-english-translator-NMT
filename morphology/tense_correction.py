import re


def correct_tense(translation, expected_tense):
    """
    Conservative tense correction.

    Only corrects obvious present/past/future forms.
    If no safe correction is found, returns the original translation.
    """

    if not translation or expected_tense == "unknown":
        return translation

    text = translation

    # --------------------------------------------------
    # PRESENT -> PAST
    # --------------------------------------------------

    if expected_tense == "past":

        replacements = {
            r"\bgoes\b": "went",
            r"\bgo\b": "went",
            r"\blives\b": "lived",
            r"\blive\b": "lived",
            r"\bcomes\b": "came",
            r"\bcome\b": "came",
            r"\bsays\b": "said",
            r"\bsay\b": "said",
            r"\bteaches\b": "taught",
            r"\bteach\b": "taught",
            r"\bspeaks\b": "spoke",
            r"\bspeak\b": "spoke",
            r"\bsees\b": "saw",
            r"\bsee\b": "saw",
            r"\bgives\b": "gave",
            r"\bgive\b": "gave",
            r"\btakes\b": "took",
            r"\btake\b": "took",
            r"\bmakes\b": "made",
            r"\bmake\b": "made",
            r"\bis\b": "was",
            r"\bare\b": "were",
            r"\bam\b": "was",
            r"\bhas\b": "had",
            r"\bhave\b": "had",
        }

        for pattern, replacement in replacements.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                return re.sub(
                    pattern,
                    replacement,
                    text,
                    count=1,
                    flags=re.IGNORECASE
                )

    # --------------------------------------------------
    # PAST -> PRESENT
    # --------------------------------------------------

    if expected_tense == "present":

        replacements = {
            r"\bwent\b": "goes",
            r"\blived\b": "lives",
            r"\bcame\b": "comes",
            r"\bsaid\b": "says",
            r"\btaught\b": "teaches",
            r"\bspoke\b": "speaks",
            r"\bsaw\b": "sees",
            r"\bgave\b": "gives",
            r"\btook\b": "takes",
            r"\bmade\b": "makes",
            r"\bwas\b": "is",
            r"\bwere\b": "are",
            r"\bhad\b": "has",
        }

        for pattern, replacement in replacements.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                return re.sub(
                    pattern,
                    replacement,
                    text,
                    count=1,
                    flags=re.IGNORECASE
                )

    # --------------------------------------------------
    # FUTURE
    # --------------------------------------------------

    if expected_tense == "future":

        replacements = {
            r"\bgoes\b": "will go",
            r"\bgo\b": "will go",
            r"\bwent\b": "will go",
            r"\blives\b": "will live",
            r"\blived\b": "will live",
            r"\bcomes\b": "will come",
            r"\bcame\b": "will come",
            r"\bsays\b": "will say",
            r"\bsaid\b": "will say",
            r"\bteaches\b": "will teach",
            r"\btaught\b": "will teach",
        }

        for pattern, replacement in replacements.items():
            if re.search(pattern, text, flags=re.IGNORECASE):
                return re.sub(
                    pattern,
                    replacement,
                    text,
                    count=1,
                    flags=re.IGNORECASE
                )

    # Nothing safe to change
    return translation


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    tests = [
        ("Rama went to the forest.", "present"),
        ("Rama goes to the forest.", "past"),
        ("Rama went to the forest.", "past"),
        ("Rama goes to the forest.", "future"),
        ("Rama lives there for many years.", "present"),
        ("Krishna taught Arjuna.", "present"),
        ("Krishna teaches Arjuna.", "past"),
    ]

    print("=" * 70)
    print("TENSE CORRECTION TEST")
    print("=" * 70)

    for translation, tense in tests:

        corrected = correct_tense(
            translation,
            tense
        )

        print("\nOriginal :", translation)
        print("Expected :", tense)
        print("Corrected:", corrected)

    print("\n" + "=" * 70)