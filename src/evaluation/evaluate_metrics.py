from sklearn import metrics
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import evaluate
import pandas as pd
import re


def get_tagger_score(y_true, y_pred, file_name):
    labels_flat = [item for sublist in y_true for item in sublist]
    predictions_flat = [item for sublist in y_pred for item in sublist]

    global_metrics = ['precision',
                      'recall',
                      'f1-score']

    per_label_metrics = ['precision_per_label',
                         'recall_per_label',
                         'f1-score_per_label']

    global_scores = {'accuracy': round(metrics.accuracy_score(labels_flat, predictions_flat), 2)}

    for metric, score in zip(global_metrics, precision_recall_fscore_support(labels_flat, predictions_flat,
                                                                             average='micro')):
        global_scores[metric] = round(score, 2)

    cm = metrics.confusion_matrix(labels_flat, predictions_flat)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_label_scores = {'accuracy_per_label': list(np.around(cm.diagonal(), decimals=2))}
    for metric, score in zip(per_label_metrics, precision_recall_fscore_support(labels_flat, predictions_flat)[:-1]):
        per_label_scores[metric] = list(np.around(score, decimals=2))

    return build_dataframes(global_scores, per_label_scores, file_name.replace('.csv', ''))


def get_translation_scores(pairs, file_name):
    bleu = evaluate.load('bleu')

    predictions, references = map(list, zip(*pairs))

    score = bleu.compute(predictions=predictions, references=references)

    scores = {'Model': file_name.replace('.csv', ''), 'bleu': round(score['bleu'], 2)}
    for i, precision in enumerate(score['precisions']):
        scores[str(i + 1) + '-gram_precision'] = round(precision, 2)

    return pd.DataFrame([scores])


def build_dataframes(global_scores, per_label_scores, model_name):

    model_name = model_name
    labels = ["correct", "misspelling", "segmentation", "union"]

    df_global = pd.DataFrame({
        'Model': model_name,
        'Metric': list(global_scores.keys()),
        'Score': list(global_scores.values())
    })

    df_data = []
    for i, label in enumerate(labels):
        row = {'Model': model_name, 'Label': label}
        for metric, scores in per_label_scores.items():
            base_metric = metric.replace('_per_label', '')
            row[base_metric] = scores[i]
        df_data.append(row)
    df_per_label = pd.DataFrame(df_data)

    return df_global, df_per_label


def main():
    file_path = '../files/outputs/'
    file_names = ['corrected_test_set-04-04-vatican-no-finetuning.csv',
                  'corrected_test_set-01-01.csv',
                  'corrected_test_set-01-04.csv',
                  'corrected_test_set-04-01.csv',
                  'corrected_test_set-04-04.csv',
                  'corrected_test_set-04-04-vatican-finetuned.csv']

    tagger_global_scores = []
    tagger_per_label_scores = []

    translation_scores = []

    for file_name in file_names:
        results = pd.read_csv(file_path + file_name)
        correct_texts = results['correct_texts'].tolist()
        corrections = results['corrections'].tolist()
        y_true = results['labels'].map(lambda l: [2 if int(match.group()) == -1 else int(match.group()) for match in
                                                  re.finditer(r'-?\d+', l)]).tolist()
        y_pred = results['predictions'].map(
            lambda l: [2 if int(match.group()) == -1 else int(match.group()) for match in
                       re.finditer(r'-?\d+', l)]).tolist()

        if 'vatican' not in file_name:
            scores = get_tagger_score(y_true, y_pred, file_name)
            tagger_global_scores.append(scores[0])
            tagger_per_label_scores.append(scores[1])
        translation_scores.append(get_translation_scores(zip(corrections, correct_texts), file_name))

    del tagger_global_scores[0]
    del tagger_global_scores[1]
    del tagger_per_label_scores[0]
    del tagger_per_label_scores[1]

    translation_scores = pd.concat(translation_scores, ignore_index=True)
    tagger_global_scores = pd.concat(tagger_global_scores, ignore_index=True)
    tagger_per_label_scores = pd.concat(tagger_per_label_scores, ignore_index=True)

    tagger_global_scores.to_csv('tagger_global_scores.csv', index=False)
    tagger_per_label_scores.to_csv('tagger_per_label_scores.csv', index=False)

    translation_scores.to_csv('translation_scores.csv', index=False)


if __name__ == '__main__':
    main()