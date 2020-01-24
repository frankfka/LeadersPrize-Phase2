from typing import List
import pandas as pd

from experiments.experiment_util import get_text_preprocessor, save_results
from preprocessor.text_preprocessor import TextPreprocessor


def main():
    text = "I Demand A Plan The horrific school shooting in Newtown, CT is the latest devastating example of the toll " \
           "of gun violence. It's time for our leaders to take action. Every day, 33 people are murdered with guns in " \
           "this country. The ""I Demand A Plan"" video project records the personal stories of Americans whose lives " \
           "were forever changed by these tragedies. Watch and share their videos, add your own and Demand A Plan! "
    preprocessor = get_text_preprocessor()
    sentences = preprocessor.process(text).bert_sentences
    print("\n".join(sentences))


if __name__ == '__main__':
    main()
