from typing import List

import spacy
import spacy.symbols as sym
from spacy.matcher import Matcher

# https://spacy.io/api/annotation
subj_symbols = {sym.nsubj, sym.csubj, sym.nsubjpass, sym.csubjpass, sym.agent, sym.expl}
obj_symbols = {sym.dobj, sym.obj, sym.attr, sym.oprd, sym.pobj, sym.acomp}
prep_symbols = {sym.ADJ}
verb_symbols = {sym.AUX, sym.VERB}
ignore_pos = {sym.AUX, sym.CONJ, sym.CCONJ, sym.DET, sym.INTJ, sym.PART, sym.PUNCT, sym.SCONJ, sym.X, sym.SPACE}
prep_pattern = [
    {"POS": "VERB", "OP": "*"},
    {"POS": "AUX", "OP": "*"},
    {"POS": "ADV", "OP": "*"},
    {"POS": "AUX", "OP": "*"},
    {"POS": "PART", "OP": "*"},
    {"POS": "VERB", "OP": "+"},
    {"POS": "PART", "OP": "*"}
]


# Define an object to hold all the processed information
class TruthTuple:

    def __init__(self, agent, event, prep_obj, is_root):
        self.agent = agent
        self.event = event
        self.prep_obj = prep_obj
        self.is_root = is_root

    def __repr__(self):
        return f"( Agent: {self.agent}; Event: {self.event}, Object: {self.prep_obj}, Is Root: {self.is_root} )"


class TruthTupleExtractor:

    def __init__(self):
        # TODO: This requires dockerfile to download the right package
        self.nlp = spacy.load("en_core_web_sm")
        prep_matcher = Matcher(self.nlp.vocab)
        prep_matcher.add("PrepositionalPhrases", None, prep_pattern)
        self.preposition_matcher = prep_matcher

    def extract(self, sentence) -> List[TruthTuple]:
        """
        Extract truth tuples from a sentence
        """
        doc = self.nlp(sentence)
        extracted_tuples = []
        verb_phrases = self.__get_verb_phrases(doc)  # All verb phrases

        # Iterate through all tokens in the doc
        for token in doc:
            # Only match subjects, which must have a preposition attached upsteam
            if token.dep not in subj_symbols:
                continue
            # Get the root verb for the subject, skip if we can't find one
            possible_verb = token.head
            while possible_verb.head != possible_verb and possible_verb.pos not in verb_symbols:
                possible_verb = possible_verb.head
            if possible_verb.pos not in verb_symbols:
                continue
            # Get truth tuple objects
            subject = token
            verb = possible_verb
            verb_is_root = verb.head == verb.head.head  # A "root" verb has its head as itself
            prep_obj = self.__get_nearest_obj(doc, verb)

            # Convert root objects into chunks (ex. verb -> verb phrases, subj/obj -> noun phrases)
            verb = self.__get_chunked_phrase(verb, doc, verb_phrases, phrase_type='verb')
            subject = self.__get_chunked_phrase(subject, doc, verb_phrases, phrase_type='object')
            if prep_obj:
                prep_obj = self.__get_chunked_phrase(prep_obj, doc, verb_phrases, phrase_type='object')

            # Add the truth tuple to list
            extracted_tuples.append(
                TruthTuple(subject, verb, prep_obj, verb_is_root)
            )

        return extracted_tuples

    def __token_is_in_span(self, token, span):
        """
        Returns True if the token is part of the span
        """
        return token.i in range(span.start, span.end)

    def __get_chunked_phrase(self, token, doc, verb_phrases, phrase_type):
        """
        Returns the verb/noun chunk/entity describing the token, or just returns the token itself if none found
        - phrase_type: object -> returns named entities or noun chunks
        - phrase_type: verb -> returns verb phrases
        """
        if phrase_type == 'object':
            # Noun chunks have greater priority than named entities
            for noun_chunk in doc.noun_chunks:
                if self.__token_is_in_span(token, noun_chunk):
                    return noun_chunk
            for entity in doc.ents:
                if self.__token_is_in_span(token, entity):
                    return entity
        elif phrase_type == 'verb':
            for verb_phrase in verb_phrases:
                if self.__token_is_in_span(token, verb_phrase):
                    return verb_phrase
        return token

    def __get_verb_phrases(self, doc):
        """
        Matches all verb phrases within the doc
        """
        # Extract all verb spans
        verb_phrases = self.preposition_matcher(doc)
        verb_spans = []
        for _, start, end in verb_phrases:
            verb_spans.append(doc[start:end])

        # These match multiple (ex. "were registered", "were registered to"), deduplicate
        verb_spans.sort(key=lambda x: len(x))
        deduped_verb_spans = []
        for i, verb_span in enumerate(verb_spans):
            found_match = False
            for j in range(i + 1, len(verb_spans)):
                other_verb_span = verb_spans[j]
                if self.__token_is_in_span(verb_span[0], other_verb_span):
                    found_match = True
                    break
            if not found_match:
                deduped_verb_spans.append(verb_span)

        return deduped_verb_spans

    def __get_nearest_obj(self, doc, prep_token):
        """
        Returns the nearest object for the preposition in the semantic tree
        """
        # Use BFS to search for closest object
        stack = list(prep_token.children)
        while stack:
            possible_obj = stack.pop(0)
            if possible_obj.dep in obj_symbols:
                return possible_obj
            stack += possible_obj.children
        return None
