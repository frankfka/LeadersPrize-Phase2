from analyze.ne_reasoner.propositions.text_analysis import get_propositional_structures, has_pos, get_np1, \
    has_a_regex_match, is_complex, regex_vp


def rule_1(doc):
    """
    """
    propositional_structures = get_propositional_structures(doc)
    for prop in propositional_structures:
        np1 = get_np1(prop)
        is_proper_noun = has_pos(np1, 'PROPN')
        is_subject_pronoun = has_pos(np1, 'PRON')
        return is_proper_noun or is_subject_pronoun
    return False


def rule_2(doc):
    """
    2- A string of words is a proposition-expressing entity if it has a proposition predicate part
    if a sentence has a VP, and is complex. It is more likely to be a proposition
    """
    has_vp = has_a_regex_match(doc, regex_vp)
    complex = is_complex(doc)
    return has_vp and complex


def rule_3(doc):
    """(time-word in SENT) or (PP in SENT) or (location-word in SENT)
    """
    has_time_indicator = False  # Todo
    has_space_indicator = False  # Todo
    has_pp = has_pos(doc, 'ADP')
    return has_time_indicator or has_space_indicator or has_pp


def rule_4(doc):
    """VP in SENT
    """
    return has_a_regex_match(doc, regex_vp)


def rule_5(doc):
    """? NP1-VP-NP2 in SENT
    """
    propositional_structures = get_propositional_structures(doc)
    return len(propositional_structures) > 0


RULES = [rule_1, rule_2, rule_3, rule_4, rule_5]
