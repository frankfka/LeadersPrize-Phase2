from typing import List

import pandas as pd

from experiments.util.experiment_util import get_text_preprocessor, save_results
from preprocess.text_preprocessor import TextPreprocessor


def __get_mapping_fn(preprocessor: TextPreprocessor):
    """Returns mapping function for lambda - processed texts have |SEP| to indicate a sentence break"""
    def process_and_map(text: str) -> str:
        text = str(text)
        if not text:
            return ''
        sentences = preprocessor.process(text).sentences
        return ' | SEP | '.join(sentences)

    return process_and_map


def __preprocess(preprocessor: TextPreprocessor, texts: List[str]) -> List[str]:
    """Returns preprocessed texts when given a list of texts"""
    return list(map(__get_mapping_fn(preprocessor), texts))


def main():
    # Get text from HTML preprocess output
    text_df = pd.read_csv("../output/html_processor/html_processor_01-23-2020_20-08-47.csv")
    preprocessor = get_text_preprocessor()
    texts = list(text_df["processed"])
    processed = __preprocess(preprocessor, texts)
    export_df = pd.DataFrame(data={"original": texts, "processed": processed})
    save_results(export_df, "text_preprocessor", "text_preprocessor_alphanum")


if __name__ == '__main__':
    main()
