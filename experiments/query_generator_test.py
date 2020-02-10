import os

import pandas as pd

from experiments.util.experiment_util import save_results, get_query_generator, get_search_client, \
    get_timestamp, get_truth_tuple_extractor
from experiments.util.train_data_util import train_data_generator


def execute_queries_export_urls():
    """
    Output the queries as well as the resulting URL's from the search client
    Will also create the HTML files returned from the query
    """
    num_examples = 30  # Limit # of examples so this runs faster
    bqg = get_query_generator()
    client = get_search_client()
    timestamp = get_timestamp()

    ids = []
    original_claims = []
    data = []

    for idx, claim in enumerate(train_data_generator()):
        if idx == num_examples:
            break

        ids.append(claim.id)
        original_claims.append(claim.claim)
        # TODO: No truth tuples here
        q = bqg.get_query(claim)
        res = client.search(q)
        export_str = ""
        for i, r in enumerate(res.results):
            # Add to URLs
            export_str += f"{r.score}: {r.url} | "
            # Write Result
            filepath = f"output/basic_query_generator/query_html_{timestamp}/{claim.id}_{i}.html"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'a') as f:
                f.write(r.content)
        data.append(export_str)

    export_df = pd.DataFrame(data={"id": ids, "original": original_claims, "results": data})
    save_results(export_df, "basic_query_generator", "query_to_url", time_str=timestamp)


def create_and_export_queries():
    """
    Output the generated queries only
    """
    num_examples = 100
    bqg = get_query_generator()
    truth_tup_extractor = get_truth_tuple_extractor()

    ids = []
    original_claims = []
    processed = []

    for idx, claim in enumerate(train_data_generator()):
        if idx == num_examples:
            break
        ids.append(claim.id)
        original_claims.append(claim.claim)
        claim_truth_tuples = truth_tup_extractor.extract(claim.claimant + " " + claim.claim)
        processed.append(bqg.get_query(claim, truth_tuples=claim_truth_tuples))

    export_df = pd.DataFrame(data={"id": ids, "original": original_claims, "queries": processed})
    save_results(export_df, "basic_query_generator", "queries")


def main():
    create_and_export_queries()


if __name__ == '__main__':
    main()