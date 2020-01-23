import re
import string
import unicodedata

import contractions
from nltk.corpus import stopwords
from num2words import num2words
from nltk import word_tokenize, sent_tokenize, pos_tag, WordNetLemmatizer
from nltk.corpus import wordnet as wn


# Tokenize by Word
def tokenize_by_word(text):
    return word_tokenize(text)


# Tokenize by Sentence
def tokenize_by_sentence(text):
    return sent_tokenize(text)


# Strip non-alphanumeric from a sentence
def keep_alphanumeric(text):
    return re.sub('[^0-9a-zA-Z]+', ' ', text).strip()


# Cleans a word-tokenized document with given options
def clean_tokenized(
        tokenized,
        remove_stopwords=False,
        remove_punctuation=False,
        lowercase=False
):
    set_to_remove = set()  # Set of strings to remove from tokenized words list

    # Add to the set
    if remove_stopwords:
        stopwords_set = set(stopwords.words('english'))
        set_to_remove = set_to_remove.union(stopwords_set)
    if remove_punctuation:
        set_to_remove = set_to_remove.union(string.punctuation)

    # Process and return
    processed_tokens = []
    for tok in tokenized:
        tok = tok.strip()
        if tok.lower() not in set_to_remove and len(tok) > 0:
            if lowercase:
                processed_tokens.append(tok.lower())
            else:
                processed_tokens.append(tok)
    return processed_tokens


# Tags POS's a word-tokenized document
# Has option for lemmatization (ex. flying -> fly)
def analyze_pos(tokenized, lemmatize):
    # This dict converts NLTK POS tags to POS args for Wordnet lemmatizer
    from collections import defaultdict
    pos_to_lemma_arg = defaultdict(lambda: wn.NOUN)  # Default to noun
    pos_to_lemma_arg.update({
        'JJ': wn.ADJ,
        'VB': wn.VERB,
        'RB': wn.ADV
    })

    # Tag words with their part of speech
    tagged_tokens = pos_tag(tokenized)
    # Lemmatize with their POS if needed
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tagged_tokens = [
            (lemmatizer.lemmatize(w, pos_to_lemma_arg[pos]), pos)
            for (w, pos) in tagged_tokens
        ]

    # Returns a tuple (word, part_of_speech)
    return tagged_tokens


# If this is a number, returns the word representation (ex. 4.2 to four point two). Otherwise, return original word
# RETURN format: an array of words ['four', 'point', 'two'] or ['original_word']
def num_2_word(word):
    processed_word = re.sub('[^0-9a-zA-Z.]+', '', word)  # All non-alphanumeric (or period) to empty str
    if re.search('[a-zA-Z]', processed_word) is None and len(processed_word.strip()) > 0:
        # No characters in string, could be a number
        try:
            word_rep = num2words(processed_word)  # ex. forty-two
            word_rep = keep_alphanumeric(word_rep)  # ex. forty two
            return word_rep.split()
        except Exception:
            return word.split()
    return word.split()


# Converts numbers in a sentence to their word representation
def convert_nums_to_words(txt):
    tokens = tokenize_by_word(txt)
    return ' '.join([processed for token in tokens for processed in num_2_word(token)])


# Expands contractions: ex. We'll -> we will
def expand_contractions(txt):
    return contractions.fix(txt, slang=False)


# Replace accented characters with non-accented
def clean_accent(txt):
    txt = unicodedata.normalize('NFD', txt).encode('ascii', 'ignore').decode("utf-8")
    return str(txt)


# Replace certain characters with words
def replace_symbols(txt):
    txt = txt.replace("%", " percent ")
    txt = txt.replace("$", " dollar ")
    return txt


# Remove adjectives and adverbs
def remove_adv_adj(tokenized_with_pos):
    cleaned_toks = []
    pos_to_clean = ['JJ', 'RB', 'FW', 'UH']
    for item in tokenized_with_pos:
        if item[1] in pos_to_clean:
            # Don't remove useful words
            word = item[0].lower()
            if not ('no' in word or 'false' in word or 'fake' in word):
                continue
        cleaned_toks.append(item[0])
    return cleaned_toks
