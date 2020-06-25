import en_core_web_sm
import textacy

nlp = None


def spacy_tokens(text):
    global nlp
    if nlp is None:
        nlp = en_core_web_sm.load()
    spacy_doc = nlp(text)
    for token in spacy_doc:
        print('\t\t'.join([token.text,
                           token.pos_,
                           token.tag_,
                           ]))


def print_matches(matches):
    print([match.text for match in matches])


def has_a_regex_match(doc, pattern):
    matches = list(textacy.extract.pos_regex_matches(doc, pattern))
    return len(matches) > 0


def verb_is_intransitive(verb):
    for child in verb.children:
        if child.dep_ in ["iobj", "pobj", "dobj", "dative"]:
            return False
    return True


def get_np_vp_nps(doc):
    pattern = fr'({regex_np})<PUNCT>?({regex_vp})<PUNCT>?({regex_np})'
    matches = list(textacy.extract.pos_regex_matches(doc, pattern))
    return matches


def get_np_vp_adjs(doc):
    pattern = fr'({regex_np})<PUNCT>?({regex_vp})<PUNCT>?({regex_adj})'
    matches = list(textacy.extract.pos_regex_matches(doc, pattern))
    return matches


def get_propositional_structures(doc):
    return get_np_vp_nps(doc) + get_np_vp_adjs(doc)


def get_np1(span):
    nps = textacy.extract.pos_regex_matches(span, regex_np)
    return next(nps)


def has_pos(span, pos):
    return pos in [token.pos_ for token in span]


def is_complex(doc):
    """Complex can be defined in terms of:
    -embeddings (sentence has more than one verb, an embedded verb (occurring after a complementizer
        such as that, which, etc.)
    -sentence includes parenthetical phrases such as guess what? I meant, I think , you know, etc.
    -sentence is a reported speech, with a structure such as We should deny access to all non-Canadians. Said JohnB"""

    verbs = textacy.extract.pos_regex_matches(doc, '<VERB>')
    has_multiple_verbs = len(list(verbs)) >= 2

    includes_parenthetical_phrase = False  # todo

    is_reported_speech = False  # todo

    return has_multiple_verbs or includes_parenthetical_phrase or is_reported_speech


regex_compound_noun = r'<NOUN>+'
regex_np = r"<DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN>|<PROPN>|<PRON> <PART>?)+"
regex_adj = r"(<ADV>* <ADJ> <CONJ>?)* (<ADV>* <ADJ>)"
non_gerund_verb = '(<AUX>|<VB>|<VBD>|<VBN>|<VBP>|<VBZ>)'
regex_vp = fr'(<VERB>* <ADV>* <PART>* <ADP>? {non_gerund_verb}+ <PART>* ({regex_adj})? (<SCONJ>|<ADP>)?)+'
regex_pp = r"<ADP> <DET>? <NUM>* (<ADJ> <PUNCT>? <CONJ>?)* (<NOUN> <PART>?)+"