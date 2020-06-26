import collections
import csv

terms_cache = {}


def get_terms(column_string, cvm_path=None):
    if column_string in terms_cache.keys():
        return terms_cache[column_string]
    else:
        if cvm_path is None:
            cvm_path = r'../data/cvms/bows_premise_space_cvm.csv'

        terms = []
        with open(cvm_path, 'r') as cvm:
            csv_reader = csv.reader(cvm)
            header = next(csv_reader)
            column_index = header.index(column_string)
            for row in csv_reader:
                term = row[column_index]
                if term != 'NA':
                    terms.append(term)
        terms_cache[column_string] = terms
        return terms


def get_term_density_in_text(text, terms):

    total_used_characters = 0
    for term in terms:
        use_count = text.count(term)

        total_used_characters += use_count * len(term)
    density_score = float(total_used_characters) / len(text)
    return density_score


Median_IQR = collections.namedtuple("median_iqr", "median iqr")

terms_medians_iqrs = {
    "evidence": Median_IQR(median=0.02052, iqr=0.0125),
    "asserting": Median_IQR(median=0.00891, iqr=0.0119),
    "hedging": Median_IQR(median=0.01789, iqr=0.018845),
    "questioning": Median_IQR(median=0.01026, iqr=0.014605),
    "disagreeing": Median_IQR(median=0.0064, iqr=0.008025),
    "stancing": Median_IQR(median=0.07401, iqr=0.02432),
    "negative": Median_IQR(median=0.0807, iqr=0.032435),
    "fakeness": Median_IQR(median=0.00388, iqr=0.00461),
}


def unskew_term_predictions(column_name, prediction):
    median, iqr = terms_medians_iqrs[column_name]
    return _unskew_predictions(prediction, median, iqr)


def _unskew_predictions(prediction, current_median, current_interquartile_range):
    # draft
    desired_median = 0.5
    desired_interquartile_range = 0.5

    unskewed = (desired_median
                + (prediction - current_median) * desired_interquartile_range / current_interquartile_range)
    clipped = max(0, min(1, unskewed))
    return clipped
