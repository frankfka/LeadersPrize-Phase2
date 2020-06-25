import re
import string
from typing import List

import preprocess.text_util as text_util


class TextPreprocessResult:

    def __init__(self, sentences: List[str]):
        self.sentences = sentences


class TextPreprocessor:

    def process_sentences(self, sentences: List[str]) -> List[str]:
        return [self.process_one_sentence(sent) for sent in sentences]

    def process_one_sentence(self, text: str) -> str:
        cleaned_sentence = text_util.expand_contractions(text)
        cleaned_sentence = text_util.replace_symbols(cleaned_sentence)
        cleaned_sentence = text_util.clean_accent(cleaned_sentence)
        cleaned_sentence = text_util.convert_num_to_words_v2(cleaned_sentence)
        return cleaned_sentence.strip()

    def __get_cleaned_sentences_v2(self, text: str) -> List[str]:
        """Feb 28 - simpler cleaning for experimentation"""
        sentences = []
        for sentence in text_util.tokenize_by_sentence(text):
            # Do cleaning on entire sentence text
            sentence = text_util.expand_contractions(sentence)
            sentence = text_util.replace_symbols(sentence)
            sentence = text_util.clean_accent(sentence)
            sentence = text_util.convert_num_to_words_v2(sentence)
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def __get_cleaned_sentences_v1(self, text: str) -> List[str]:
        """Jan - initial implementation"""
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
            # Final cleaning - keep only alphanumeric
            sentence = " ".join([item[0] for item in sent_words_with_pos])
            sentence = text_util.keep_alphanumeric(sentence).strip()
            if sentence:
                sentences.append(sentence)
        return sentences

    def __get_cleaned_sentences_phase1(self, text: str) -> List[str]:
        """Cleaning used in Phase 1"""
        sentences = []
        for sentence in text_util.tokenize_by_sentence(text):
            # Do cleaning on entire sentence text
            sentence = text_util.expand_contractions(sentence)
            sentence = text_util.replace_symbols(sentence)
            sentence = text_util.clean_accent(sentence)
            sentence = re.sub('[^0-9a-zA-Z.,]+', ' ', sentence)  # Strip non-alphanum except period and comma (useful in numbers)
            sentence = text_util.convert_nums_to_words(sentence)
            # Strip periods and commas, but we can't do a Regex because U.S. -> U S
            words_in_sentence = []
            # This will retain U.S. -> U.S., we could alternatively just cast U.S. -> US
            for word in text_util.tokenize_by_word(sentence):
                word = word.strip()
                if word not in set(string.punctuation):
                    words_in_sentence.append(word)
            sentence = ' '.join(words_in_sentence)
            if sentence:
                sentences.append(sentence)
        return sentences