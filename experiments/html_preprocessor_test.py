import os
import pandas as pd

from experiments.experiment_util import save_results, get_html_preprocessor


def main():
    article_filenames = []
    processed_articles = []
    processor = get_html_preprocessor()

    html_dir = './html_files'
    # Iterate through all HTML files
    for f in os.listdir(html_dir):
        filepath = f"{html_dir}/{f}"
        with open(filepath, 'r', errors='ignore') as c:
            print(f)
            html_content = c.read()
            processed = processor.process(html_content).text.strip()
            if len(processed) == 0:
                print(f"{f} processing gave empty body")
            article_filenames.append(f)
            processed_articles.append(processed)

    results_df = pd.DataFrame(data={"file": article_filenames, "processed": processed_articles})
    save_results(results_df, "html_processor")


if __name__ == '__main__':
    main()