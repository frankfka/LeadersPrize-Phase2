import pandas as pd

from experiments.util.experiment_util import get_query_generator, get_search_client, \
    get_text_preprocessor, get_html_preprocessor, get_word2vec_relevance_scorer, save_results, \
    get_infersent_relevance_scorer
from experiments.util.train_data_util import train_data_generator
from analyze.sentence_relevance_scorer import InfersentRelevanceScorer
from analyze.sentence_relevance_scorer.word2vec_relevance_scorer import Word2VecRelevanceScorer

INFERSENT_RELEVANCE_SCORER = "infersent"
WORD2VEC_RELEVANCE_SCORER = "word2vec"

# Run config
NUM_EXAMPLES = 50  # Number of examples to process, as this can get quite expensive
RELEVANCE_TYPE = INFERSENT_RELEVANCE_SCORER


def get_word2vec_relevances(claim, sentences, scorer: Word2VecRelevanceScorer):
    relevances = []
    for sentence in sentences:
        relevances.append(scorer.get_relevance(claim, sentence))
    return relevances


def get_infersent_relevances(claim, sentences, scorer: InfersentRelevanceScorer):
    return scorer.get_relevance(claim, sentences)


def main():
    # Services
    query_generator = get_query_generator()
    client = get_search_client()
    html_preprocessor = get_html_preprocessor()
    text_preprocessor = get_text_preprocessor()
    # Create the appropriate relevance scorer
    relevance_scorer = get_infersent_relevance_scorer() if RELEVANCE_TYPE == INFERSENT_RELEVANCE_SCORER \
        else get_word2vec_relevance_scorer()

    # Outputs
    ids = []
    original_claims = []
    processed_claims = []
    queries = []
    processed_sentences = []
    true_labels = []

    for idx, claim in enumerate(train_data_generator()):
        if idx == NUM_EXAMPLES:
            break
        print(idx)

        # TODO: No truth tuples here
        query = query_generator.get_query(claim)
        search_res = client.search(query)
        processed_claim = ' '.join(text_preprocessor.process(claim.claim).bert_sentences)  # Preprocessed claim

        ids.append(claim.id)
        original_claims.append(claim.claim)
        true_labels.append(claim.label)
        queries.append(query)
        processed_claims.append(processed_claim)
        # Continue in case of error
        if search_res.error:
            processed_sentences.append(f"Error: {search_res.error}")
            continue

        # Create master list of sentences
        sentences = []
        for article in search_res.results:
            html_processed = html_preprocessor.process(article.content).text
            text_processed = text_preprocessor.process(html_processed)
            sentences += text_processed.bert_sentences

        # Run relevance scores
        if RELEVANCE_TYPE == INFERSENT_RELEVANCE_SCORER:
            relevances = get_infersent_relevances(claim.claim, sentences, relevance_scorer)
        else:
            relevances = get_word2vec_relevances(claim.claim, sentences, relevance_scorer)

        # Combine the two results
        processed_sentences_with_relevance = list(zip(relevances, sentences))
        # Construct final string
        processed_sentences_with_relevance.sort(key=lambda item: item[0], reverse=True)
        process_result = ""
        for rel, sent in processed_sentences_with_relevance:
            if len(process_result) > 10000:
                # Some basic truncation to limit file size
                break
            process_result += f"|SEP| {rel}: {sent} \n"

        processed_sentences.append(process_result)

    # Export the result, with relevance scores and the processed text
    export_df = pd.DataFrame(data={"id": ids, "label": true_labels, "original": original_claims, "query": queries,
                                   "processed_claim": processed_claims, "processed_sentences": processed_sentences})
    save_results(export_df, "sentence_relevance_scorer", f"claim_to_{RELEVANCE_TYPE}_relevance")


if __name__ == '__main__':
    main()
