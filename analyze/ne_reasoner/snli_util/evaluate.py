import csv
import sys

INVERSE_MAP = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}


def evaluate_classifier(classifier, eval_set, batch_size):
    """
    Function to get accuracy and cost of the model, evaluated on a chosen dataset.

    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    correct = 0
    genres, hypotheses, cost = classifier(eval_set)
    cost = cost / batch_size
    full_batch = int(len(eval_set) / batch_size) * batch_size
    for i in range(full_batch):
        hypothesis = hypotheses[i]
        if hypothesis == eval_set[i]['label']:
            correct += 1
    return correct / float(len(eval_set)), cost


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


def evaluate_final(restore, classifier, eval_set, batch_size, num_labels=3):
    """
    Function to get percentage accuracy of the model, evaluated on a set of chosen datasets.
    
    restore: a function to restore a stored checkpoint
    classifier: the model's classfier, it should return genres, logit values, and cost for a given minibatch of the evaluation dataset
    eval_set: the chosen evaluation set, for eg. the dev-set
    batch_size: the size of minibatches.
    """
    restore(best=True)

    genres, hypotheses, cost = classifier(eval_set)
    correct = 0
    full_batch = int(len(eval_set) / batch_size) * batch_size

    true_positives = [0] * num_labels
    false_positives = [0] * num_labels
    false_negatives = [0] * num_labels

    for i in range(full_batch):
        hypothesis = hypotheses[i]
        actual = eval_set[i]['label']
        if hypothesis == actual:
            correct += 1
            true_positives[hypothesis] += 1
        else:
            false_negatives[hypothesis] += 1
            false_positives[actual] += 1
    accuracy = correct / float(len(eval_set))
    precisions = [true_positives[i] / (true_positives[i] + false_positives[i]) for i in range(num_labels)]
    recalls = [true_positives[i] / (true_positives[i] + false_negatives[i]) for i in range(num_labels)]
    fscores = [2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]) for i in range(num_labels)]
    return accuracy, precisions, recalls, fscores


def get_predictions(classifier, eval_set, batch_size):
    hypotheses = classifier(eval_set)
    predictions = []

    for i in range(len(eval_set)):
        hypothesis = hypotheses[i]
        prediction = INVERSE_MAP[hypothesis]
        pairID = eval_set["pairID"].iloc[i]
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
