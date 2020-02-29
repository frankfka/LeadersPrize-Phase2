from experiments.util.experiment_util import get_text_preprocessor


def main():
    text = "McConnell's promise to keep the ability to filibuster legislation could make it more difficult for " \
           "Republicans to get key parts of Trump's legislative agenda through the Senate, considering the expected " \
           "strong Democratic opposition. 'Break the rules' A filibuster requires a super-majority of \"60 votes\" in the .5 or -1 " \
           "100-seat Senate in order to proceed to a simple majority vote on a Supreme Court nominee or legislation for 600,000 participants of 200,000. " \
           "The 60-vote super-majority threshold that gives the minority party power to hold up the majority party " \
           "has forced the Senate over the decades to try to achieve bipartisanship in legislation and presidential " \
           "appointments. "
    preprocessor = get_text_preprocessor()
    sentences = preprocessor.process(text).bert_sentences
    print("\n".join(sentences))


if __name__ == '__main__':
    main()
