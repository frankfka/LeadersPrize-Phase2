import csv

import pandas as pd

INVERSE_MAP = {
    0: "not_paraphrase",
    1: "paraphrase",
}


def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    is_paraphrase = pd.Series(eval_set[:full_batch]['is_paraphrase'], dtype='int64')
    correct_vector = is_paraphrase == hypotheses
    num_correct = sum(correct_vector)
    return num_correct / float(len(eval_set)), cost


def evaluate_classifier_genre(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre, 0) for genre in set(genres))
    count = dict((genre, 0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print('welp!')

    accuracy = {k: correct[k] / count[k] for k in correct}

    return accuracy, cost


def evaluate_classifier_bylength(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model by genre, evaluated on a chosen dataset. It returns a dictionary of accuracies by genre and cost for the full evaluation dataset.
    
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    genres, hypotheses, cost = classifier(eval_set)
    correct = dict((genre, 0) for genre in set(genres))
    count = dict((genre, 0) for genre in set(genres))
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        genre = genres[i]
        if hypothesis == eval_set[i]['label']:
            correct[genre] += 1.
        count[genre] += 1.

        if genre != eval_set[i]['genre']:
            print('welp!')

    accuracy = {k: correct[k] / count[k] for k in correct}

    return accuracy, cost


def evaluate_final(restore, classifier, eval_sets, batch_size):
    """
    Function to get percentage accuracy of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)
    percentages = []
    length_results = []
    for eval_set in eval_sets:
        bylength_prem = {}
        bylength_hyp = {}
        genres, hypotheses, cost = classifier(eval_set)
        correct = 0
        cost = cost / batch_size
        full_batch = int(len(eval_set) / batch_size) * batch_size
        sentence0_len = eval_set['sentence0'].str.split().str.len()
        sentence1_len = eval_set['sentence1'].str.split().str.len()

        for i in range(full_batch):
            actual = hypotheses[i]
            expected = int(eval_set['is_paraphrase'].iloc[i])

            if actual == expected:
                correct += 1

            for sentence_len in sentence0_len, sentence1_len:
                length = sentence_len.iloc[i]

                if length not in bylength_prem.keys():
                    bylength_prem[length] = [0, 0]
                bylength_prem[length][1] += 1

                if actual == expected:
                    bylength_prem[length][0] += 1

        percentages.append(correct / float(len(eval_set)))
        length_results.append((bylength_prem, bylength_hyp))
    return percentages, length_results


def get_predictions(classifier, eval_set):
    hypotheses = classifier(eval_set)
    predictions = []

    for i in range(len(eval_set)):
        hypothesis = hypotheses[i]
        prediction = INVERSE_MAP[hypothesis]
        pairID = eval_set["pairID"][i]
        predictions.append((pairID, prediction))
    return predictions


def save_predictions(predictions, name):
    f = open(name + '_predictions.csv', 'w', errors='ignore')  # wb
    w = csv.writer(f, delimiter=',')
    w.writerow(['Hypothesis', 'Premise', 'Label'])
    for example in predictions:
        w.writerow(example)
    f.close()


def predictions_kaggle(classifier, eval_set, batch_size, name):
    """
    Get comma-separated CSV of predictions.
    Output file has two columns: pairID, prediction
    """
    predictions = get_predictions(classifier, eval_set, batch_size)
    save_predictions(predictions, name)
