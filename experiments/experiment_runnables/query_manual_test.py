from core.models import LeadersPrizeClaim
from search_client.client import ArticleSearchClient


def does_query_generate_provided_articles(claim: LeadersPrizeClaim,
                                          query: str,
                                          search_client: ArticleSearchClient):
    """
    Determine whether the query given will generate matching articles
    - This returns the set of matching articles

    Can run by the following:
    from experiments.util.train_data_util import train_data_generator
    from experiments.util.experiment_util import get_search_client
    from experiments.experiment_runnables.query_manual_test import does_query_generate_provided_articles
    client = get_search_client()
    train_data_generator = train_data_generator("PATH")

    idx, claim = next(train_data_generator)
    query = "example query"
    does_query_generate_provided_articles(claim, query, client)
    """
    res = search_client.search(query)
    res_urls = [r.url for r in res.results]
    claim_urls = [c.url for c in claim.related_articles]
    print(res_urls)
    print(claim_urls)
    return list(set(res_urls).intersection(claim_urls))