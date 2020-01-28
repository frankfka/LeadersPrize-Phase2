import requests


class SearchQueryResult:

    def __init__(self, hit):
        self.content = hit["content"]
        self.score = hit["score"]
        self.url = hit["url"]


class SearchQueryResponse:

    def __init__(self, r):
        self.error = None
        self.results = []
        if r.status_code == 200:
            # Elasticsearch sometimes gives duplicate results
            urls = set()
            results = []
            for hit in r.json()["hits"]["hits"]:
                result = SearchQueryResult(hit)
                if result.url not in urls:
                    results.append(result)
                    urls.add(result.url)
            self.results = results
        else:
            self.error = r.text


class ArticleSearchClient:

    def __init__(self, host: str, api_key: str):
        self.endpoint = host + '/claimserver/api/v1.0/evidence'
        self.headers = {'X-Api-Key': api_key}

    def search(self, query: str) -> SearchQueryResponse:
        params = {'query': query}
        resp = requests.get(self.endpoint, params=params, headers=self.headers)
        return SearchQueryResponse(resp)

