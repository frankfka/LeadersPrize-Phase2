import asyncio
from typing import List

import aiohttp


class SearchQueryResult:
    """
    Data structure for each article result
    """

    def __init__(self, hit=None, content: str = "", url: str = ""):
        # Constructor allows for custom search query results
        self.content = content
        self.score = 0
        self.url = url
        if hit:
            self.content = hit["content"]
            self.score = hit["score"]
            self.url = hit["url"]


def __deduplicate_results__(results: List[SearchQueryResult]) -> List[SearchQueryResult]:
    """
    Elasticsearch gives duplicates, so we dedup the URL's
    """
    urls = set()
    deduplicated_results: List[SearchQueryResult] = []
    for result in results:
        # Remove https/http for deduplication
        clean_url = __remove_http__(result.url)
        if clean_url not in urls:
            deduplicated_results.append(result)
            urls.add(clean_url)
    return deduplicated_results


def __remove_http__(url) -> str:
    if url.startswith("https://"):
        return url[8:]
    elif url.startswith("http://"):
        return url[7:]
    return url


class ArticleSearchClient:

    def __init__(self, host: str, api_key: str):
        self.endpoint = host + '/claimserver/api/v1.0/evidence'
        self.headers = {'X-Api-Key': api_key}

    def search(self, query: str, num_results=90) -> List[SearchQueryResult]:
        # Get search params
        search_params = []
        for from_index in range(0, num_results, 30):
            search_params.append({"query": query, "from": from_index})
        # Fetch
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.search_in_async_session(search_params))

    async def search_in_async_session(self, search_params):
        results = []
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
            nested_results = await asyncio.gather(*[self.search_one_async(session, params) for params in search_params],
                                                  return_exceptions=True)
            for nested_result_list in nested_results:
                results.extend(nested_result_list)
        return __deduplicate_results__(results)

    async def search_one_async(self, session, params):
        results = []
        try:
            async with session.get(self.endpoint, params=params, headers=self.headers) as resp:
                resp_json = await resp.json()
                hits = resp_json["hits"]["hits"]
                for hit in hits:
                    try:
                        results.append(SearchQueryResult(hit=hit))
                    except Exception as parse_exception:
                        print("Error decoding result")
                        print(parse_exception)
        except Exception as search_exception:
            print("Exception calling search client")
            print(search_exception)
        return results
