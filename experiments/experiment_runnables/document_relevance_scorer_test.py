import math
from typing import List

from core.models import PipelineArticle
from experiments.util.experiment_util import get_query_generator, get_search_client, get_html_preprocessor, \
    get_lsa_relevance_scorer, get_text_preprocessor
from experiments.util.train_data_util import train_data_generator


def test_document_relevance_scorer():
    """
    This runs the pipeline up to retrieving articles and ranking them by relevance.
    Determines, if there is shared articles between what is retrieved and what is given as related_articles,
    the ranking of the related article
    """
    num_examples = 1000  # Limit # of examples so this runs faster
    bqg = get_query_generator()
    client = get_search_client()
    html_preprocessor = get_html_preprocessor()
    text_prepreprocessor = get_text_preprocessor()
    article_relevance_scorer = get_lsa_relevance_scorer()

    total_searched = 0
    average_rankings = []

    for idx, claim in train_data_generator("/Users/frankjia/Desktop/LeadersPrize/train/train.json"):
        if idx == num_examples:
            break
        print(idx)
        # Execute query
        q = bqg.get_query(claim)
        searched_articles = client.search(q).results

        # Process articles from raw HTML to parsed text
        pipeline_articles: List[PipelineArticle] = []
        for raw_article in searched_articles:
            if raw_article and raw_article.content:
                pipeline_article = PipelineArticle(raw_article)
                # Extract data from HTML
                html_process_result = html_preprocessor.process(raw_article.content)
                pipeline_article.html_attributes = html_process_result.html_atts
                pipeline_article.raw_body_text = html_process_result.text
                pipeline_articles.append(pipeline_article)

        # Get Article Relevance
        preprocessed_claim_sentences = text_prepreprocessor.process(claim.claim + " " + claim.claimant).sentences
        preprocessed_claim = claim.claim + " " + claim.claimant
        if preprocessed_claim_sentences:
            preprocessed_claim = preprocessed_claim_sentences[0]
        pipeline_article_texts: List[str] = [p.raw_body_text for p in pipeline_articles]
        article_relevances = article_relevance_scorer.analyze(preprocessed_claim,
                                                              pipeline_article_texts)
        for article_relevance, pipeline_article in zip(article_relevances, pipeline_articles):
            # Sometimes we get nan from numpy operations
            pipeline_article.relevance = article_relevance if math.isfinite(article_relevance) else 0

        # Based on article relevance, only consider the top relevances
        pipeline_articles.sort(key=lambda article: article.relevance, reverse=True)

        sorted_urls = [article.url for article in pipeline_articles]
        claim_urls = [article.url for article in claim.related_articles]
        common_urls = list(set(sorted_urls).intersection(claim_urls))

        total_searched += 1
        if common_urls:
            # Determine index of shared url in the sorted urls
            index_sum = 0
            for url in common_urls:
                index_sum += sorted_urls.index(url)
            average_rankings.append(float(index_sum) / len(common_urls))

    print("RESULTS")
    print(total_searched)
    print(len(average_rankings))
    print(float(sum(average_rankings)) / len(average_rankings))

if __name__ == '__main__':
    test_document_relevance_scorer()