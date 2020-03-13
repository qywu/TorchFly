import os
from typing import List


def merge_short_sents(sentences: List[str], min_length=20) -> List[str]:
    """ Sentences shorter than 20 will be merged together.
        One reason for doing this is to avoid the errors from Spacy.
    """
    new_sentences = []
    history = []

    for sent in sentences:
        if len(history) > 0:
            sent = " ".join(history) + " " + sent
        if len(sent) < 20:
            history.append(sent)
            continue
        else:
            history = []
        new_sentences.append(sent)

    return new_sentences