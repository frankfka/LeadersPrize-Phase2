from experiments.util.experiment_util import get_text_preprocessor


def main():
    text = "Mark Harris Says Democratic North Carolina House candidate Dan McCready \"took money from the Pelosi crowd\" and is \"with them\" in their agenda and \"wants to repeal your tax cut.\""
    preprocessor = get_text_preprocessor()
    sentences = preprocessor.process(text).bert_sentences
    print("\n".join(sentences))


if __name__ == '__main__':
    main()
