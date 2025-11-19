import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("presidio-flask-api")

cases = [
    # label, Estonian case name, English case name
    ("n", "nimetav", "nominative"),
    ("g", "omastav", "genitive"),
    ("p", "osastav", "partitive"),
    ("ill", "sisseütlev", "illative"),
    ("in", "seesütlev", "inessive"),
    ("el", "seestütlev", "elative"),
    ("all", "alaleütlev", "allative"),
    ("ad", "alalütlev", "adessive"),
    ("abl", "alaltütlev", "ablative"),
    ("tr", "saav", "translative"),
    ("ter", "rajav", "terminative"),
    ("es", "olev", "essive"),
    ("ab", "ilmaütlev", "abessive"),
    ("kom", "kaasaütlev", "comitative"),
]


def synthesize_word(word):
    """
    Synthesize all case forms for a single Estonian word

    Args:
        word: Single word string

    Returns:
        List of all case forms (singular + plural)
    """
    # Lazy import to avoid circular import issues
    from estnltk.vabamorf.morf import synthesize

    sing_rows = []
    plur_rows = []

    for case, case_name_est, case_name_eng in cases:
        sing_rows.extend(synthesize(word, "sg " + case, "S"))
        plur_rows.extend(synthesize(word, "pl " + case, "S"))

    all_forms = sing_rows + plur_rows
    # Remove duplicates and empty strings
    all_forms = [form for form in set(all_forms) if form.strip()]

    logger.info(f"Synthesized {len(all_forms)} forms for word '{word}'")
    return all_forms


def synthesize_all(words):
    """
    Synthesize all case forms for a list of Estonian words

    Args:
        words: List of word strings or single word string

    Returns:
        List containing original words plus all synthesized case forms
    """
    if not words:
        return []

    # Handle single word string
    if isinstance(words, str):
        words = [words]

    result = []

    for word in words:
        if not word or not isinstance(word, str):
            continue

        word = word.strip()
        if not word:
            continue

        # Add original word
        result.append(word)

        # Add all synthesized forms
        try:
            synthesized = synthesize_word(word)
            result.extend(synthesized)
        except Exception as e:
            logger.warning(f"Could not synthesize word '{word}': {e}")
            # Continue with just the original word

    # Remove duplicates while preserving order
    seen = set()
    unique_result = []
    for item in result:
        item_lower = item.lower()
        if item_lower not in seen:
            seen.add(item_lower)
            unique_result.append(item)

    logger.info(
        f"Total words after synthesis: {len(unique_result)} (from {len(words)} original)"
    )
    return unique_result
