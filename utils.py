from estnltk.vabamorf.morf import synthesize
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


def synthesize_word(word: str) -> list[str]:
    """
    Synthesize all case forms for a single Estonian word

    Vabamorf needs the right part of speech. Allow- and denylists are full of
    proper nouns - place names, company names, project codenames - and for a
    capitalised word Vabamorf returns nothing under "S" (common noun), which
    silently reduced this whole function to an identity. "H" is pärisnimi,
    the proper-noun class; "S" is kept as a fallback for ordinary words and for
    surnames, which "H" does not always cover.

    Args:
        word: Single word string

    Returns:
        List of all case forms (singular + plural)
    """
    pos_tags = ("H", "S") if word[:1].isupper() else ("S", "H")

    all_forms: list[str] = []
    for pos in pos_tags:
        for case, _, _ in cases:
            all_forms.extend(synthesize(word, "sg " + case, pos))
            all_forms.extend(synthesize(word, "pl " + case, pos))
        # Matching is case-insensitive downstream, so a lowercase fallback form
        # is still useful; only try the next tag if this one produced nothing.
        if all_forms:
            break

    # Remove duplicates and empty strings
    all_forms = [form for form in set(all_forms) if form.strip()]

    logger.info(f"Synthesized {len(all_forms)} forms for word '{word}'")
    return all_forms


def synthesize_phrase(phrase: str) -> list[str]:
    """
    Synthesize case forms for an allow/denylist entry of one or more words

    Estonian inflects the head of such a phrase, which is its last word, so
    "Maksu- ja Tolliamet" has to become "Maksu- ja Tolliametis",
    "Maksu- ja Tolliametisse" and so on rather than being left untouched.

    Args:
        phrase: One allow/denylist entry, possibly several words

    Returns:
        List of inflected variants of the whole phrase
    """
    parts = phrase.split()
    if len(parts) < 2:
        return synthesize_word(phrase)

    prefix = " ".join(parts[:-1])
    return [f"{prefix} {form}" for form in synthesize_word(parts[-1])]


def synthesize_all(words: list[str] | str) -> list[str]:
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
            synthesized = synthesize_phrase(word)
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
