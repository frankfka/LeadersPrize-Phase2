import string
from typing import List

import nltk

import util.text_util as text_util


class TextPreprocessResult:

    def __init__(self, bert_sentences: List[str]):
        self.bert_sentences = bert_sentences


class TextPreprocessor:

    def process(self, text: str) -> TextPreprocessResult:
        bert_sentences = self.__get_cleaned_sentences(text)
        return TextPreprocessResult(bert_sentences)

    def __get_cleaned_sentences(self, text: str) -> List[str]:
        sentences = []
        for sentence in text_util.tokenize_by_sentence(text):
            # Do cleaning on entire sentence text
            sentence = text_util.expand_contractions(sentence)
            sentence = text_util.replace_symbols(sentence)
            sentence = text_util.clean_accent(sentence)
            sentence = text_util.convert_nums_to_words(sentence)
            # Get parts of speech so that we can remove certain items
            sent_words_with_pos = text_util.get_pos(
                    text_util.tokenize_by_word(sentence)
            )
            sent_words_with_pos = text_util.remove_pos(sent_words_with_pos)
            # Final cleaning - remove any punctuation
            sent_words = text_util.clean_tokenized(
                list(map(lambda x: x[0], sent_words_with_pos)),
                remove_stopwords=False,
                remove_punctuation=True
            )
            sentences.append(' '.join(sent_words))
        return sentences
