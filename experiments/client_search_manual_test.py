import os

from experiments.util.experiment_util import get_search_client, get_timestamp


def execute_query_export_html(query: str):
    client = get_search_client()
    res = client.search(query)
    time_str = get_timestamp()
    for i, r in enumerate(res.results):
        print(r.url)
        # Write Result
        filepath = f"output/manual_client_search/{time_str}_{i}.html"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'a') as f:
            f.write(r.content)


def main():
    q = "179,000 human beings jail country today 2.3-million percent black african-american mike gravel"
    q += " "
    execute_query_export_html(q)


if __name__ == '__main__':
    main()