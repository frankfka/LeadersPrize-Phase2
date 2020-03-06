"""
We use "newspaper" to do HTML parsing, but this fails often. We may want to improve our default parsing
if these HTML articles are valuable sources of information.

This script will:
- Iterate over training data
- Detect when newspaper parsing fails
- Write the HTML article file paths to a text file for later review
"""
from experiments.util.experiment_util import get_html_preprocessor
from experiments.util.train_data_util import train_data_generator, get_train_article

PROCESS_RANGE = range(6000, 7000)
TRAIN_DATA_PATH = "/Users/frankjia/Desktop/LeadersPrize/train/"

# IMPORTANT: To run this successfully, edit html_preprocessor to return only the newspaper-parsed text
def main():
    html_preprocessor = get_html_preprocessor()
    error_files = []
    for idx, claim in train_data_generator(TRAIN_DATA_PATH + "train.json"):
        if idx < PROCESS_RANGE.start:
            continue
        elif idx >= PROCESS_RANGE.stop:
            break
        # Add the articles if we're not retrieving from search client
        for related_article in claim.related_articles:
            article_html = get_train_article(TRAIN_DATA_PATH, related_article.filepath)
            if not html_preprocessor.process(article_html).text:
                print(related_article.filepath)
                error_files.append(related_article.filepath)

    output = '\n'.join(error_files)
    with open('../output/html_processor/html_unprocessable_files.txt', 'a') as f:
        f.write(output)


if __name__ == '__main__':
    main()