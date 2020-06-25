from __future__ import unicode_literals

import itertools

import sklearn.metrics
import textacy
import textacy.utils

from propositions.rules import RULES
from propositions.text_analysis import spacy_tokens, print_matches, regex_np, regex_vp


def disable_textacy_depreciated_notifications():
    """Replace's textacy's depreciated notification with a no-op."""

    def deprecated(message: str, *, action: str = "always"):
        """An empty function to replace the depreciated function"""
        pass

    textacy.utils.deprecated = deprecated


disable_textacy_depreciated_notifications()


def analyze_sentences(sentences):
    """Print analytics for each sentence"""
    for sentence in sentences:
        analyze_sentence(sentence)


def analyze_sentence(sentence):
    """Print analytics for the sentence."""
    doc = textacy.make_spacy_doc(sentence, lang='en_core_web_sm')

    print(doc.text)
    spacy_tokens(sentence)
    print("noun phrases")
    print_matches(textacy.extract.pos_regex_matches(doc, regex_np))
    print('verb phrases')
    print_matches(textacy.extract.pos_regex_matches(doc, regex_vp))

    rules = [rule(doc) for rule in RULES]
    print(f'NP1 is right type: {rules[0]}')
    print(f'Has VP and is complex: {rules[1]}')
    print(f'Has Time, Space, or PP: {rules[2]}')
    print(f'Has VP: {rules[3]}')
    print(f'Has propositional structure: {rules[4]}')
    print(f'Is not a question: {rules[5]}')
    print(f'Score = {sum(rules)}/{len(rules)}')
    print()


def get_y_true(examples, counter_examples):
    """Get a list of is_proposition bools for scoring."""
    return [True] * len(examples) + [False] * len(counter_examples)


def get_precision_recall_f1(y_pred, y_true):
    precision = sklearn.metrics.precision_score(y_pred=y_pred, y_true=y_true)
    recall = sklearn.metrics.recall_score(y_pred=y_pred, y_true=y_true)
    f1 = sklearn.metrics.f1_score(y_pred=y_pred, y_true=y_true)
    return precision, recall, f1


def precision_recall_f1_string(precision, recall, f1):
    return f'Precision: {precision:.2f}\tRecall: {recall:.2f} \tF1 score: {f1:.2f}'


EXAMPLES = [
    "John is a nice man",
    "You are very impatient",
    "She said that the boy is tall",
    "Apple is looking at buying U.K. startup for $1 billion",
    "It is a nice day.",
    "When it rains, it pours.",
    "It pours when it rains.",
    "Skiing is easy.",
    "I don't want to go to Toronto."]

COUNTER_EXAMPLES = [
    "Welcome to the University of Auckland!",
    "How can I stop tailgating?",
    "When the car ahead reaches an object, make sure you count to four before you reach the object.",
    "Please help me navigate to the city.",
    "Run away from Toronto.",
    "A big brown cow, standing in a field.",
    "Help me if you can.",
    "Is John a nice man?"
]


def init_weights(examples=None, counter_examples=None):
    """Sets a variable on each function for its precision and recall on the passed data, so the rule can be weighted.

    This is basically just a way to avoid making them single function classes."""
    if examples is None:
        examples = EXAMPLES
    if counter_examples is None:
        counter_examples = COUNTER_EXAMPLES

    examples = [textacy.make_spacy_doc(example, lang='en_core_web_sm') for example in examples]
    counter_examples = [textacy.make_spacy_doc(example, lang='en_core_web_sm') for example in counter_examples]
    for i, rule in enumerate(RULES):
        y_pred = []
        for doc in itertools.chain(examples, counter_examples):
            y_pred.append(rule(doc))
        y_true = get_y_true(examples, counter_examples)
        precision, recall, f1 = get_precision_recall_f1(y_pred, y_true)
        # print(f'Rule {i + 1}\t{precision_recall_f1_string(precision, recall, f1)}')
        rule.precision = precision
        rule.recall = recall


def combined_classify(doc):
    proposition_score = 0

    for rule in RULES:
        if rule(doc):
            proposition_score += rule.precision
        else:
            proposition_score -= rule.recall

    return proposition_score > 0


def combined_score(examples, counter_examples):
    """Run the weighted rules to vote on classification."""
    examples = [textacy.make_spacy_doc(example, lang='en_core_web_sm') for example in examples]
    counter_examples = [textacy.make_spacy_doc(example, lang='en_core_web_sm') for example in counter_examples]
    y_pred = []
    for doc in itertools.chain(examples, counter_examples):
        y_pred.append(combined_classify(doc))
    y_true = get_y_true(examples, counter_examples)
    precision, recall, f1 = get_precision_recall_f1(y_pred, y_true)
    print(f'Combined Score \t{precision_recall_f1_string(precision, recall, f1)}')


if __name__ == '__main__':
    print("EXAMPLES")
    analyze_sentences(EXAMPLES)
    print()
    print("COUNTER EXAMPLES")
    analyze_sentences(COUNTER_EXAMPLES)

    init_weights(examples=EXAMPLES, counter_examples=COUNTER_EXAMPLES)
    combined_score(EXAMPLES, COUNTER_EXAMPLES)
