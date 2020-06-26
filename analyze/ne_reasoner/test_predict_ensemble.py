import nltk
import pandas as pd

from analyze.ne_reasoner import predict_ensemble, predict_shared


def get_sentences(text):
    sentences = nltk.sent_tokenize(text)
    sentences = [sentence for sentence in sentences if len(sentence) >= 10]
    sentences = [sentence.strip() for sentence in sentences]
    return sentences


def safe_get_column(source, name):
    try:
        column = source[name]
    except KeyError:
        column = pd.Series(0, index=source.index)
    return column


def paraphrase_score_to_label(score):
    if score > 0:
        return 'Paraphrase'
    else:
        return 'Not Paraphrase'


def entailment_score_to_label(score):
    entailment_threshold = 0.33
    contradiction_threshold = -0.33

    if entailment_threshold < score:
        return 'Entailment'
    elif contradiction_threshold < score:
        return 'Neutral'
    else:
        return 'Contradiction'


#def rumor_scores_to_label(comment, support, deny, query):
    max_score = max(comment, support, deny, query)
    if max_score == comment:
        return 'Comment'
    elif max_score == support:
        return 'Support'
    elif max_score == deny:
        return 'Deny'
    elif max_score == query:
        return 'Query'
    else:
        assert False


OUTPUT_PATH = '../output'


def clear_attribute_files():
    import os
    import glob

    files = glob.glob(f'../output/*')
    for f in files:
        os.remove(f)


def run_ensemble_classifier(classifier, claims, articles_with_names, starting_index=0):
    for claim_index, claim in enumerate(claims):
        run_ensemble_classifier_for_claim(classifier, claim_index+starting_index, claim, articles_with_names)


def run_ensemble_classifier_for_claim(classifier, claim_index, claim, articles_with_names):
    claim_df = pd.DataFrame()
    for article_name, article in zip(articles_with_names[0], articles_with_names[1]):
        sentences = get_sentences(article)
        lines, predictions = classifier.predict_statement_in_contexts(statement=claim, contexts=sentences)

        try:
            relevance_density = len(lines) / len(sentences)
        except ZeroDivisionError:
            continue
        article_df = pd.DataFrame(predictions)

        article_df['article'] = article_name
        article_df['line'] = lines
        article_df['hypothesis'] = claim
        article_df['article_relevance'] = relevance_density
        sentences = [sentences[i] for i in lines]
        article_df['sentence'] = sentences
        claim_df = claim_df.append(article_df)

    if len(claim_df) == 0:
        print(f'{claim} has no relevant sentences. '
              f'Hypothesis is proposition: {predict_ensemble.is_proposition(claim)}')
    else:
        path = f'{OUTPUT_PATH}/{claim_index}_sentence_predictions.csv'
        claim_df = claim_df[['hypothesis',
                             'article',
                             'article_relevance',
                             'sentence',
                             'line',
                             'entailment',
                             'neutral',
                             'contradiction',
                             'paraphrase',
                             ] + classifier.term_column_names]
        claim_df.to_csv(path)

        article_mean_df = claim_df.groupby('article').mean()

        summary_df_dict = {
            'claim': claim,
            'premise': article_mean_df.index,
            'article_relevance': claim_df.groupby('article')['article_relevance'].first(),
        }

        columns = [
                      'entailment', 'contradiction', 'neutral',
                      'paraphrase',
                  ] + classifier.term_column_names
        for column in columns:
            df_column = {column: safe_get_column(article_mean_df, column)}
            summary_df_dict.update(df_column)

        summary_df = pd.DataFrame(summary_df_dict)

        summary_path = f'{OUTPUT_PATH}/{claim_index}_summary.csv'
        summary_df.to_csv(summary_path, index=False, float_format='%.5f')


if __name__ == '__main__':
    classifier = predict_ensemble.EnsembleClassifier()

    hypotheses = [
    'Hand washing causes spread of viral infections ',
]
    article_names, articles = predict_shared.load_articles(path='datainput')

    run_ensemble_classifier(classifier, hypotheses, (article_names, articles))

    print('done')
    # # rough test for sanity.
    # test_examples = [
    #     ('The cow is on the grass', 'The cow is on the grass'),
    #     ('The cow is on the grass', 'The cow is in the grassy field'),
    #     ('The cow is in the grassy field', 'The cow is on the grass'),
    #     ('The cow is on the grass', 'There is rain in the city'),
    #     ('There is rain in the city', 'There is rain in the city'),
    #     ('it is raining in around the buildings', 'There is rain in the city')
    # ]
    # predictions = []
    # for test_example in test_examples:
    #     prediction = classifier.predict_statement_in_context(statement=test_example[0],
    #                                                          context=test_example[1])
    #     predictions.append(prediction)
    #     print(prediction)
