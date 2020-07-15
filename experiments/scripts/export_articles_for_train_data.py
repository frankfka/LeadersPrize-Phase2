from experiments.util.experiment_util import get_search_client
import json
search_client = get_search_client()

root_folder_path = "/Users/frankjia/Desktop/LeadersPrize/train_phase2_val"
train_json_filepath = "train_val_phase2.json"
new_train_json_path = "train.json"
with open(f"{root_folder_path}/{train_json_filepath}", "r") as f:
    train_json = json.load(f)

new_train_json = []
article_key = 0
for item in train_json:
    print(item["id"])
    new_rel_articles = {}
    given_rel_articles = item["related_articles"]
    for _, article_url in given_rel_articles.items():
        results = search_client.search(article_url, num_results=30)
        url_matched_result = None
        for result in results:
            if result.url == article_url:
                url_matched_result = result
                break
        if url_matched_result:
            # Write the output
            article_filepath = f"train_articles/{article_key}.html"
            with open(f"{root_folder_path}/{article_filepath}", "w") as f:
                f.write(url_matched_result.content)
            article_key += 1
            # add to new rel articles
            new_rel_articles[article_filepath] = url_matched_result.url
        else:
            print(f"Could not find {article_url} through search")
    item["related_articles"] = new_rel_articles
    if len(new_rel_articles.items()) > 0:
        new_train_json.append(item)


with open(f"{root_folder_path}/{train_json_filepath}", "w") as f:
    json.dump(new_train_json, f, indent=4)
