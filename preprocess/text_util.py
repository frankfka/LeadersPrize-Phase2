import re
import string
import unicodedata

import contractions
from nltk.corpus import stopwords
from num2words import num2words
import numpy as np
from nltk import word_tokenize, sent_tokenize, pos_tag
from sklearn.metrics.pairwise import cosine_similarity

# Get cosine similarity between two vectors
def cos_sim(u, v):
    cos_similarities = cosine_similarity(u, v)  # n samples x n samples matrix
    return np.diagonal(cos_similarities)[0]


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


# Tags parts of speech, outputs list of tuples
def get_pos(word_tokens):
    return pos_tag(word_tokens)


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


def convert_num_to_words_v2(txt):
    new_txt = ""  # Create a new string to construct existing
    prev_end = 0
    matches = list(re.finditer(r"\.?([+-]?(?=.)((\d{1,3})+(,\d{3})*))(\.\d+)?", txt))
    matches.sort(key=lambda item: item.start(0))
    for match in matches:
        match_start, match_end, match_txt = match.start(0), match.end(0), match.group(0)
        # Add rest of the text up to this match
        new_txt += txt[prev_end:match_start]
        matched_num_txt = txt[match_start:match_end]
        # Attempt to map a number to text representation
        try:
            # Parse the number after getting rid of anything that's not a number, period, or dash
            matched_num_txt = num2words(re.sub(r"[^0-9.\-]+", "", matched_num_txt))  # ex. forty-two
            matched_num_txt = keep_alphanumeric(matched_num_txt)  # ex. forty two
        except Exception:
            # Not printing anything to keep console clean
            # print(f"Error converting {matched_num_txt} to text representation")
            pass
        # Add converted text
        new_txt += matched_num_txt
        # Update previous end index
        prev_end = match_end
    # Add the rest of the text, up to the last match index
    new_txt += txt[prev_end:]
    return new_txt


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


# Remove selected POS' from the input array, outputs an array of pos-tagged tokens without the selected parts of speech
def remove_pos(words_with_pos):
    pos_to_clean = [
        'POS',  # Possessives ex. 's
        'TO',  # ex. to go "to" the store
        'FW',  # Foreign words
        'UH'  # Interjections
    ]
    return list(filter(lambda item: item[1] not in pos_to_clean, words_with_pos))
