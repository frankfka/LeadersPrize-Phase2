import os
import re

import pandas as pd

from experiments.util.experiment_util import get_query_generator, get_search_client, get_html_preprocessor, \
    save_results, get_lsa_relevance_scorer
from experiments.util.train_data_util import train_data_generator

# Run config
NUM_EXAMPLES = 10  # Number of examples to process, as this can get quite expensive


def main():
    """
    For each claim, run query generator and rank the results by relevance using the LSA document relevance scorer
    """
    # Services
    query_generator = get_query_generator()
    client = get_search_client()
    html_preprocessor = get_html_preprocessor()
    relevance_scorer = get_lsa_relevance_scorer()

    # Outputs
    ids = []
    original_claims = []
    ranked_articles_for_claims = []

    for idx, claim in train_data_generator("/Users/frankjia/Desktop/LeadersPrize/train/train.json"):
        if idx == NUM_EXAMPLES:
            break
        print(idx)

        query = query_generator.get_query(claim)
        search_res = client.search(query)

        ids.append(claim.id)
        original_claims.append(claim.claim)

        # Create master list of sentences
        article_texts = []
        article_urls = []
        for article in search_res.results:
            # Write the article for future checking
            url_filename = re.sub(r"([/:.])+", "_", article.url)  # Create a save-friendly filename
            filepath = f"output/{claim.id}/{url_filename}.html"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'a') as f:
                f.write(article.content)

            # Process the articles
            html_processed_text = html_preprocessor.process(article.content).text
            article_urls.append(article.url)
            article_texts.append(html_processed_text)

        # Both claim and article_texts are unpreprocessed - LSA class currently does the preprocessing
        # TODO: Add in claimant as well?
        relevances = relevance_scorer.analyze(claim.claim, article_texts)
        print(relevances)
        articles_with_relevances = list(zip(article_urls, relevances))
        articles_with_relevances.sort(key=lambda x: x[1], reverse=True)

        # Create an export string with the URL and the relevance:
        article_rank_result = ""
        for url, rel in articles_with_relevances:
            article_rank_result += f"( {rel}: {url} )"
        ranked_articles_for_claims.append(article_rank_result)

    # Export the result, with relevance scores and the processed text
    export_df = pd.DataFrame(data={"id": ids, "claim": original_claims, "ranked_articles": ranked_articles_for_claims})
    save_results(export_df, "document_relevance_scorer", "claim_to_ranked_articles")


if __name__ == '__main__':
    main()
