from __future__ import annotations
import pandas as pd

import csv
import dataclasses
import typing as t
import numpy as np
import sklearn.metrics
from tqdm import tqdm

"""Truth attributes are measureable qualities expected to correlate with truth.
"""

TRUE = "TRUE"
NEUTRAL = "NEUTRAL"
FALSE = "FALSE"

TRUE_MAGNITUDE = 'true_magnitude'
NEUTRAL_MAGNITUDE = 'neutral_magnitude'
FALSE_MAGNITUDE = 'false_magnitude'


@dataclasses.dataclass
class BeliefAnalysis:
    """A claim and premise with their truth attributes."""
    claim: str
    premise: str

    entailment: float
    relevance: float
    asserting: float
    evidence: float

    paraphrase: float
    questioning: float
    disagreeing: float
    hedging: float
    neutral: float

    negative: float
    contradiction: float
    stancing: float
    fakeness: float

    @property
    def true_classification(self) -> str:
        close_tolerance = 0.08

        if np.isclose(self.true_magnitude, self.neutral_magnitude, rtol=close_tolerance):  # step 7iii
            return NEUTRAL
        elif np.isclose(self.false_magnitude, self.neutral_magnitude, rtol=close_tolerance):  # step 7iv
            return NEUTRAL
        elif np.isclose(self.true_magnitude, self.false_magnitude, rtol=close_tolerance):  # step 7v
            return NEUTRAL
        elif self.true_magnitude > self.false_magnitude:  # step 7vi
            return TRUE
        elif self.true_magnitude < self.false_magnitude:  # step 7vii
            return FALSE
        else:
            assert False

    @property
    # @functools.lru_cache() #todo this causes a hash error, but may be a good performance increase if fixed
    def true_magnitude(self) -> float:
        return np.linalg.norm(self.true_attribs) / np.linalg.norm([1] * len(self.true_attribs))

    @property
    # @functools.lru_cache()
    def neutral_magnitude(self) -> float:
        return np.linalg.norm(self.neutral_attribs) / np.linalg.norm([1] * len(self.neutral_attribs))

    @property
    # @functools.lru_cache()
    def false_magnitude(self) -> float:
        return np.linalg.norm(self.false_attribs) / np.linalg.norm([1] * len(self.false_attribs))

    @property
    def true_attribs(self):
        return np.array((self.entailment, self.relevance, self.asserting, self.evidence,
                         # self.support
                         ))

    @property
    def neutral_attribs(self):
        return np.array((self.paraphrase, self.questioning, self.disagreeing, self.hedging, self.neutral))

    @property
    def false_attribs(self):
        return np.array((self.negative, self.contradiction, self.stancing, self.fakeness,
                         # self.deny
                         ))

    @classmethod
    def from_csv_row(cls, row: t.List):
        """"""
        (claim, premise, relevance,
         entailment, contradiction, neutral,
         paraphrase,
         # comment, support, deny, query,
         evidence, asserting, hedging, questioning, disagreeing, stancing, negative, fakeness) = row

        return cls(claim=claim, premise=premise,
                   entailment=float(entailment), relevance=float(relevance), evidence=float(evidence),

                   asserting=float(asserting), paraphrase=float(paraphrase), hedging=float(hedging),
                   questioning=float(questioning), neutral=float(neutral),

                   disagreeing=float(disagreeing), contradiction=float(contradiction), stancing=float(stancing),
                   negative=float(negative), fakeness=float(fakeness),
                   )


def get_belief_analyses(premise: str, claim: str):
    """gld"""
    pass  # todo This will orchestrate several classifiers to obtain truth attributes


def get_all_toy_data():
    import glob
    toy_data_paths = glob.glob(f'../output/*_summary.csv')
    all_toy_data = []
    for path in toy_data_paths:
        all_toy_data += get_toy_data(path)
    return all_toy_data


def get_toy_data(path: str):
    """Gets truth attributes for precompiled training data."""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        attributes = [BeliefAnalysis.from_csv_row(row) for row in reader]
    return attributes


truth_attributes = ['entailment', 'relevance', 'evidence']
neutral_attributes = ['asserting', 'paraphrase', 'hedging', 'questioning', ]
false_attributes = ['disagreeing', 'contradiction', 'stancing', 'negative', 'fakeness']
summary_header = (['claim', 'premise',
                   'true_classification', 'true_magnitude', 'neutral_magnitude', 'false_magnitude', ]
                  + truth_attributes
                  + neutral_attributes
                  + false_attributes
                  )


def save_combined_summary(classifications, save_path):
    csv_path = f'../../{save_path}.csv'
    with open(csv_path, 'w+', encoding='utf-8', errors='ignore', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)
        for classification in classifications:
            row = [classification.__getattribute__(col) for col in summary_header]
            writer.writerow(row)

    df = pd.read_csv(csv_path)
    profile = ProfileReport(df)
    profile.to_file(f'../../{save_path}.html')


def get_verdicts(classifications_df):
    classification_by_claim = classifications_df.groupby('claim')

    magnitudes = [TRUE_MAGNITUDE, NEUTRAL_MAGNITUDE, FALSE_MAGNITUDE]

    verdict_df = classification_by_claim.mean()[magnitudes]

    greatest_magnitude = verdict_df[magnitudes].idxmax(axis=1)
    verdict_label = greatest_magnitude.map({TRUE_MAGNITUDE: TRUE,
                                            NEUTRAL_MAGNITUDE: NEUTRAL,

                                            FALSE_MAGNITUDE: FALSE})
    verdict_df['verdict'] = verdict_label
    return verdict_df, greatest_magnitude


def get_classifications_df(classifications):
    classifications_df = pd.DataFrame([{col: analysis.__getattribute__(col) for col in summary_header}
                                       for analysis in classifications])
    return classifications_df


def save_verdict(classifications, save_path):
    import pandas_profiling

    classifications_df = get_classifications_df(classifications)
    verdict_df, greatest_magnitude = get_verdicts(classifications_df)

    # add exemplars
    verdict_df['exemplar_0'] = None
    verdict_df['exemplar_1'] = None
    verdict_df['exemplar_2'] = None

    for claim in classifications_df.claim:
        classifications_for_claim = classifications_df[classifications_df.claim == claim]
        classifications_for_claim = classifications_for_claim.sort_values(greatest_magnitude[claim], ascending=False)
        top = classifications_for_claim.head(3)
        for i in range(3):
            try:
                premise = top.premise.iloc[i]
            except IndexError:
                premise = None
            verdict_df.loc[claim, f'exemplar_{i}'] = premise

    # add correlations
    for attribute_types, type_name in zip([truth_attributes, neutral_attributes, false_attributes],
                                          ['true', 'neutral', 'false']):
        for attribute in attribute_types:
            verdict_df[f'{attribute}_{type_name}_corr'] = 0
            for claim in verdict_df.index:
                correlations = classifications_df[classifications_df.claim == claim].corr().fillna(0)
                correlation = correlations[attribute][f'{type_name}_magnitude']
                verdict_df.loc[claim, f'{attribute}_{type_name}_corr'] = correlation

    cols = list(verdict_df)
    # move the column to head of list using index, pop and insert
    front = [cols.pop(cols.index('verdict')),
             cols.pop(cols.index('exemplar_0')),
             cols.pop(cols.index('exemplar_1')),
             cols.pop(cols.index('exemplar_2'))]
    cols = front + cols
    verdict_df = verdict_df.loc[:, cols]

    verdict_df.to_csv(f'../../{save_path}.csv')


def save_analytics():
    all_toy_data = get_all_toy_data()

    summary_path = f'output/combined_summary'
    save_combined_summary(all_toy_data, summary_path)

    verdict_path = f'output/verdict'
    save_verdict(all_toy_data, verdict_path)


def run_classifier_with_shared_premises(claims, article_names, articles):
    import predict_ensemble
    test_predict_ensemble.clear_attribute_files()
    classifier = predict_ensemble.EnsembleClassifier()

    test_predict_ensemble.run_ensemble_classifier(
        classifier,
        claims,
        (articles, article_names))


def run_classifier_with_separate_premises(claims, article_names, articles):
    import predict_ensemble
    test_predict_ensemble.clear_attribute_files()
    classifier = predict_ensemble.EnsembleClassifier()

    iterator = tqdm(enumerate(zip(claims, article_names, articles)), desc="Classifying claims", total=len(claims))

    for i, data_for_claim in iterator:
        claim, article_names_for_claim, articles_for_claim = data_for_claim
        test_predict_ensemble.run_ensemble_classifier(
            classifier,
            [claim],
            (article_names_for_claim, articles_for_claim),
            starting_index=i)


fact_check_rating_to_label = {
    2: TRUE,
    1: NEUTRAL,
    0: FALSE
}


def load_fact_check(start_claim: int, num_claims: int):
    data_path = '../data/factcheck/data.json'
    with open(data_path) as json_file:
        json = json_file.read()
    df = pd.read_json(json)

    return df.iloc[start_claim:start_claim+num_claims]


def get_fact_check_articles(articles_dict):
    article_names = list(articles_dict.keys())
    articles = []
    for file_name in article_names:
        try:
            with open(f'../data/factcheck/{file_name}', errors='ignore') as f:
                raw = f.read()
                parsed = predict_shared.parse_html(raw)
                articles.append(parsed)
        except FileNotFoundError:
            pass

    return article_names, articles


class Evaluator:
    def __init__(self, num_claims, start=0):
        self.start_claim = start
        self.num_claims = num_claims
        self.test_claims = load_fact_check(self.start_claim, self.num_claims)

    def build_truth_attributes_for_evaluation(self):
        claims = self.test_claims.claim
        article_names_and_articles = [get_fact_check_articles(articles)
                                      for articles in self.test_claims.related_articles]
        article_names, articles = zip(*article_names_and_articles)

        run_classifier_with_separate_premises(claims=claims,
                                              article_names=article_names,
                                              articles=articles)

    def get_predicted_verdicts(self):
        classifications = get_all_toy_data()
        classifications_df = get_classifications_df(classifications)
        verdict_df, _ = get_verdicts(classifications_df)
        return verdict_df.verdict

    def evaluate_classifier(self):
        verdict_predict = self.get_predicted_verdicts()
        verdict_true = self.test_claims.label.map(fact_check_rating_to_label)


        confusion_matrix = sklearn.metrics.confusion_matrix(y_pred=verdict_predict, y_true=verdict_true)

        self.plot_confusion_matrix(y_pred=verdict_predict, y_true=verdict_true, columns=[FALSE, NEUTRAL, TRUE])

        output = ""

        evaluation = pd.DataFrame({'prediction': list(verdict_predict), 'actual': verdict_true})
        evaluation.to_csv(f'../../output/evaluation_raw.csv', index=False)


        label_names = [FALSE, NEUTRAL, TRUE]

        conf_df = pd.DataFrame(confusion_matrix, index=label_names, columns=label_names).astype(int)


        output += f'Total predicted: \n{conf_df.sum()}\n\n'
        output += f'Total actual: \n{conf_df.T.sum()}\n\n'
        output += f'True positive: \n{conf_df.da.TP}\n\n'
        output += f'False positive: \n{conf_df.da.FP}\n\n'
        output += f'True negative: \n{conf_df.da.TN}\n\n'
        output += f'False negative: \n{conf_df.da.FN}\n\n'

        output += f'Accuracy: \n{conf_df.da.accuracy}\n'
        output += f'Micro Accuracy: {conf_df.da.micro_accuracy}\n\n'

        output += f'Precision: \n{conf_df.da.precision}\n'
        output += f'Micro Precision: {conf_df.da.micro_precision}\n\n'

        output += f'Recall: \n{conf_df.da.recall}\n'
        output += f'Micro Recall: {conf_df.da.micro_recall}\n\n'

        output += f'F1: \n{conf_df.da.f1}\n'
        output += f'Micro F1: {conf_df.da.micro_f1}\n\n'

        output += f'Error rate: \n{1 - conf_df.da.accuracy}\n\n'
        output += f'Specificity: \n{conf_df.da.specificity}\n'
        output += f'Micro Specificity: {conf_df.da.micro_specificity}\n\n'


        output += f'False positive rate: \n{conf_df.da.false_positive_rate}\n\n'

        da = conf_df.da
        mcc = ((da.TP * da.TN - da.FP * da.FN)
               / np.sqrt((da.TP + da.FP) * (da.TP + da.FN) * (da.TN + da.FP) * (da.TN + da.FN)))
        output += f'Mathews correlation coefficient: \n{mcc}\n\n'

        print(output)
        with open('../../output/evaluation.txt', 'w+') as f:
            f.write(output)
        print('done')

    def plot_confusion_matrix(self, y_pred, y_true, columns):
        import matplotlib.pyplot as plt
        import beliefs.pretty_print_confusion as ppc
        ppc.plot_confusion_matrix_from_data(predictions=y_pred, y_test=y_true, columns=columns)
        plt.savefig('../../output/confusion_matrix.png')


if __name__ == '__main__':
    evaluator = Evaluator(200, start=200)
    evaluator.build_truth_attributes_for_evaluation()
    evaluator.evaluate_classifier()
    save_analytics()
    print('done')
