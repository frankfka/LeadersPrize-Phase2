import os

from experiments.experiment_util import get_search_client, get_timestamp


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
    q = "Rural schools enroll 40 percent American children, receive 22 percent federal education funding"
    q += " countryside education money"
    execute_query_export_html(q)


if __name__ == '__main__':
    main()