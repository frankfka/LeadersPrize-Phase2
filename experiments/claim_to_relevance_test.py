import pandas as pd

from experiments.experiment_util import train_data_generator, get_query_generator, get_search_client, \
    get_text_preprocessor, get_html_preprocessor, get_relevance_scorer, save_results


def main():
    # Don't process everything as this can get quite expensive
    num_items = 50

    # Services
    query_generator = get_query_generator()
    client = get_search_client()
    html_preprocessor = get_html_preprocessor()
    text_preprocessor = get_text_preprocessor()
    relevance_scorer = get_relevance_scorer()

    # Outputs
    ids = []
    original_claims = []
    processed_claims = []
    queries = []
    processed_sentences = []
    true_labels = []

    for idx, claim in enumerate(train_data_generator()):
        if idx == num_items:
            break
        print(idx)

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

        # Process article text from HTML
        processed_articles = map(lambda article: html_preprocessor.process(article.content).text, search_res.results)

        # Combine all the articles, find relevance, then sort by decreasing relevance
        processed_sentences_with_relevance = []
        for a in processed_articles:
            for processed_sentence in text_preprocessor.process(a).bert_sentences:
                relevance = relevance_scorer.get_relevance(processed_claim, processed_sentence)
                processed_sentences_with_relevance.append((relevance, processed_sentence))

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
    save_results(export_df, "relevance_scorer", "claim_to_relevance")


if __name__ == '__main__':
    main()
