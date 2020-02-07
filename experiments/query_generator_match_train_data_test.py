from experiments.experiment_util import train_data_generator, get_query_generator, get_search_client, \
    get_truth_tuple_extractor


def compare_query_results_with_train_data():
    """
    Compare the resulting URL's from searched articles with the ground truth articles given in training data
    """
    num_examples = 30  # Limit # of examples so this runs faster
    bqg = get_query_generator()
    truth_tup_extractor = get_truth_tuple_extractor()
    client = get_search_client()

    ids = []
    claims = []
    queries = []
    training_urls = []
    client_urls = []
    shared_urls_for_claim = []

    # TODO: Finish this off
    for idx, claim in enumerate(train_data_generator()):
        if idx == num_examples:
            break

        # Execute query
        claim_truth_tuples = truth_tup_extractor.extract(claim.claimant + " " + claim.claim)
        q = bqg.get_query(claim, truth_tuples=claim_truth_tuples)
        res = client.search(q)
        searched_urls = []
        # Get URLs from the result
        for r in res.results:
            searched_urls.append(r.url)
        # Get URL's from training data
        training_article_urls = []
        for item in claim.related_articles:
            training_article_urls.append(item.url)
        # Get shared items
        shared_urls = list(set(training_article_urls).intersection(searched_urls))

        ids.append(claim.id)
        claims.append(claim.claim)
        queries.append(q)
        training_urls.append(training_article_urls)
        client_urls.append(searched_urls)
        shared_urls_for_claim.append(shared_urls)

    # Get stats
    num_claims_with_shared_articles = sum([1 for item in shared_urls_for_claim if item])
    frac_shared = float(num_claims_with_shared_articles) / len(ids)
    print(f"{num_claims_with_shared_articles} claims searched with shared articles out of {len(ids)}: {frac_shared}")

    num_shared_articles = sum([len(item) for item in shared_urls_for_claim])
    total_train_articles = sum([len(item) for item in training_urls])
    print(f"{num_shared_articles} shared articles found in {total_train_articles} total: {float(num_shared_articles) / total_train_articles}")

    # TODO: Export dataframe

def main():
    compare_query_results_with_train_data()


if __name__ == '__main__':
    main()
