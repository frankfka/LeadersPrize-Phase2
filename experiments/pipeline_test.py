import os

import pandas as pd

from experiments.experiment_util import train_data_generator, save_results, get_query_generator, get_search_client, \
    get_timestamp


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
    bqg = get_query_generator()

    ids = []
    original_claims = []
    processed = []

    for claim in train_data_generator():
        ids.append(claim.id)
        original_claims.append(claim.claim)
        processed.append(bqg.get_query(claim))

    export_df = pd.DataFrame(data={"id": ids, "original": original_claims, "processed": processed})
    save_results(export_df, "basic_query_generator", "queries")


def main():
    execute_queries_export_urls()


if __name__ == '__main__':
    main()