import os
from datetime import datetime

from experiments.util.experiment_util import get_search_client, get_timestamp


def execute_query_export_html(query: str, save_html=False):
    client = get_search_client()
    results = client.search_async(query)
    time_str = get_timestamp()
    for i, r in enumerate(results):
        print(r.url)
        if save_html:
            # Write Result
            filepath = f"output/manual_client_search/{time_str}_{i}.html"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'a') as f:
                f.write(r.content)


def sync_vs_async(query: str):
    client = get_search_client()
    now = datetime.now()
    async_results = client.search_async(query, num_results=30)
    print(f"{len(async_results)} async results in {datetime.now() - now}")
    now = datetime.now()
    sync_results = client.search(query, num_results=30)
    print(f"{len(sync_results)} sync results in {datetime.now() - now}")


def main():
    q = "Obama no experience background national security affairs John McCain"
    sync_vs_async(q)


if __name__ == '__main__':
    main()