import pandas as pd

from experiments.experiment_util import train_data_generator, save_results
from query_generator.query_generator import QueryGenerator
from search_client.client import ArticleSearchClient


def execute_queries_export_urls(data_path: str, client: ArticleSearchClient):
    bqg = QueryGenerator()

    ids = []
    original_claims = []
    data = []

    for claim in train_data_generator(data_path):
        ids.append(claim.id)
        original_claims.append(claim.claim)
        q = bqg.get_query(claim)
        res = client.search(q)
        export_str = ""
        for r in res.results:
            export_str += f"{r.score}: {r.url} | "
        data.append(export_str)

    export_df = pd.DataFrame(data={"id": ids, "original": original_claims, "results": data})
    save_results(export_df, "basic_query_generator", "result_url")


def create_and_export_queries():
    bqg = QueryGenerator()

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
    create_and_export_queries()


if __name__ == '__main__':
    main()