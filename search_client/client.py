from typing import List

import requests


class SearchQueryResult:
    """
    Data structure for each article result
    """

    def __init__(self, hit):
        self.content = hit["content"]
        self.score = hit["score"]
        self.url = hit["url"]


def __remove_http__(url) -> str:
    if url.startswith("https://"):
        return url[8:]
    elif url.startswith("http://"):
        return url[7:]
    return url


class SearchQueryResponse:
    """
    Data structure for each individual query
    """

    def __init__(self, r):
        self.error: str = ""
        self.results: List[SearchQueryResult] = []
        if r.status_code == 200:
            # Elasticsearch sometimes gives duplicate results
            urls = set()
            results = []
            for hit in r.json()["hits"]["hits"]:
                result = SearchQueryResult(hit)
                # Remove https/http for deduplication
                clean_url = __remove_http__(result.url)
                if clean_url not in urls:
                    results.append(result)
                    urls.add(clean_url)
            self.results = results
        else:
            self.error = r.text


class ClientSearchResult:
    """
    Data structure for the article client result
    """

    def __init__(self, results: List[SearchQueryResult], error: str):
        self.error = error
        self.results = results


class ArticleSearchClient:

    def __init__(self, host: str, api_key: str):
        self.endpoint = host + '/claimserver/api/v1.0/evidence'
        self.headers = {'X-Api-Key': api_key}

    def search(self, query: str, num_results=30) -> ClientSearchResult:
        """
        Searches the given endpoint with the query
        - num_results is the initial # of results that we ask for, the final number might be lower if there are dupes
        """
        client_response = ClientSearchResult([], "")
        initial_from = 0  # An index for Elasticsearch to begin its query
        while initial_from < num_results:
            # TODO: Parallelize this if needed: https://aiohttp.readthedocs.io/en/stable/client_quickstart.html
            params = {'query': query, 'from': initial_from}
            resp = SearchQueryResponse(requests.get(self.endpoint, params=params, headers=self.headers))
            initial_from += 10
            if resp.error:
                print(f"Error querying {query}: {resp.error}")
                client_response.error = resp.error  # Take the last error
                continue
            client_response.results += resp.results
        return client_response

