import os

from experiments.util.experiment_util import get_search_client, get_timestamp


def execute_query_export_html(query: str, save_html=False):
    client = get_search_client()
    res = client.search(query)
    time_str = get_timestamp()
    for i, r in enumerate(res.results):
        print(r.url)
        if save_html:
            # Write Result
            filepath = f"output/manual_client_search/{time_str}_{i}.html"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'a') as f:
                f.write(r.content)


def main():
    q = "Obama no experience background national security affairs John McCain"
    execute_query_export_html(q, save_html=False)


if __name__ == '__main__':
    main()