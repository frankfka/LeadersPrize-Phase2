from collections import Counter
import numpy as np


def compute_score_phase2(predictions, labels):
    """
    predictions and labels are a dict
    check for duplicate relevant articles
    """
    n_class = Counter([labels[o]['label'] for o in labels])
    scores = {'0': [], '1': [], '2': []}
    preds = []
    explanations = []
    if len(predictions) != len(labels):
        print('prediction missing for some claims')
    # loop over predictions as the normalizing factor is n_class (# labels predicted)
    for claim_id in predictions:
        if len(predictions[claim_id]['explanation']) > 1000:
            return {'score': 0.0, 'explanation': "'N/S'",
                    'error': "'MaxCharacterLimitError'",
                    'predictions': "'N/A'"}
        pred = predictions[claim_id]['label']
        preds.append(str(pred))
        label = labels[claim_id]['label']
        if pred != label:
            scores[str(label)].append(0)
            continue
        rel_articles = list(predictions[claim_id]['related_articles'].values())
        if len(rel_articles) > 2:
            return {'score': 0.0, 'explanation': "'N/S'",
                    'error': "'MaxRelatedArticlesLimitError'",
                    'predictions': "'N/A'"}
        # remove any duplicate url links
        rel_articles = set(rel_articles)
        gt_rel_articles = list(labels[claim_id]['related_articles'].values())
        scores[str(label)].append(sum([int(a in gt_rel_articles) for a in rel_articles]))
        explanations.append(predictions[claim_id]['explanation'].replace("'", ""))
    for l in scores:
        if not scores[l]:  # if scores[l] is [], np.mean returns a NaN
            scores[l] = 0.0
        else:
            scores[l] = sum(scores[l])/n_class[int(l)]
    return {'score': np.mean(list(scores.values())),
            'error': "'None'",
            'explanation': "'{}'".format('|'.join(explanations)),
            'predictions': "'[{}]'".format(','.join(preds))}