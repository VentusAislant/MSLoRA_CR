import argparse
import csv
import json
import collections
import os

from tabulate import tabulate

from nltk.translate.bleu_score import sentence_bleu
from llava.eval.eval_metrics.evaluate_metrics import calculate_f1score
from llava.eval.eval_metrics.glossary import *


def evaluate(items, report_hit_samples=True, report_not_hit_samples=True):
    # items is a list of dict contain keys: "id", "question", "pred", "gt", "model_id"

    hit_sample_ids, not_hit_sample_ids = [], []

    # gt is totally equal to pred
    exact_scores = collections.defaultdict(list)

    # for closed question, return accuracy
    closed_scores = collections.defaultdict(list)

    # for open question, return the recall score and bleu score
    open_scores = collections.defaultdict(list)
    bleu_scores = collections.defaultdict(list)

    for item in items:
        id = item['question_id']
        if 'gt' in item:
            gt_value = str(item['gt']).lower()
        elif 'ground_truth' in item:
            gt_value = str(item['ground_truth']).lower()

        if 'text' in item:
            pred_value = str(item['text']).lower()
        elif 'prediction' in item:
            pred_value = str(item['prediction']).lower()

        pred_value = pred_value.replace('assistant:', '').strip()

        gt_value = normalize_word(gt_value).strip()
        pred_value = normalize_word(pred_value).strip()
        # print(gt_value, pred_value)

        exact_scores['q_id'].append(id)
        if gt_value == pred_value:
            exact_scores['hit'].append(1)
        else:
            exact_scores['hit'].append(0)

        type = item['answer_type'] if 'answer_type' in item.keys() else 'OPEN'
        if type == 'OPEN':

            f1_score, precision, recall = calculate_f1score(pred_value, gt_value)
            open_scores['f1'].append(f1_score)
            open_scores['precision'].append(precision)
            open_scores['recall'].append(recall)
            open_scores['q_id'].append(id)

            b_score = sentence_bleu(references=[str(gt_value).lower().split()],
                                    hypothesis=str(pred_value).lower().split())
            b_score_1 = sentence_bleu(references=[str(gt_value).lower().split()],
                                      hypothesis=str(pred_value).lower().split(), weights=(1, 0, 0, 0))
            b_score_2 = sentence_bleu(references=[str(gt_value).lower().split()],
                                      hypothesis=str(pred_value).lower().split(), weights=(0, 1, 0, 0))
            b_score_3 = sentence_bleu(references=[str(gt_value).lower().split()],
                                      hypothesis=str(pred_value).lower().split(), weights=(0, 0, 1, 0))

            bleu_scores['q_id'].append(id)
            bleu_scores['bleu_score'].append(b_score)
            bleu_scores['bleu_score_1'].append(b_score_1)
            bleu_scores['bleu_score_2'].append(b_score_2)
            bleu_scores['bleu_score_3'].append(b_score_3)
            if recall > 0.5:
                hit_sample_ids.append(id)
            else:
                not_hit_sample_ids.append(id)

        elif type == 'CLOSED':
            closed_scores['q_id'].append(id)
            if gt_value == pred_value or gt_value in pred_value:
                closed_scores['hit'].append(1)
                hit_sample_ids.append(id)
            else:
                closed_scores['hit'].append(0)
                not_hit_sample_ids.append(id)

    exact_score = sum(exact_scores['hit']) / len(exact_scores['hit'])

    table_list = [['exact match score', exact_score * 100], ]

    if len(open_scores['q_id']) > 0:
        f1_score = sum(open_scores['f1']) / len(open_scores['f1'])
        precision = sum(open_scores['precision']) / len(open_scores['precision'])
        recall = sum(open_scores['recall']) / len(open_scores['recall'])

        bleu_score = sum(bleu_scores['bleu_score']) / len(bleu_scores['bleu_score'])
        bleu_score_1 = sum(bleu_scores['bleu_score_1']) / len(bleu_scores['bleu_score_1'])
        bleu_score_2 = sum(bleu_scores['bleu_score_2']) / len(bleu_scores['bleu_score_2'])
        bleu_score_3 = sum(bleu_scores['bleu_score_3']) / len(bleu_scores['bleu_score_3'])

        table_list += [
            ['f1 score', f1_score * 100],
            ['precision', precision * 100],
            ['recall', recall * 100],
            ['bleu_score', bleu_score * 100],
            ['bleu_score_1', bleu_score_1 * 100],
            ['bleu_score_2', bleu_score_2 * 100],
            ['bleu_score_3', bleu_score_3 * 100],
        ]

    if len(closed_scores['q_id']) > 0:
        closed_score = sum(closed_scores['hit']) / len(closed_scores['hit'])
        table_list += [['close accuracy', closed_score * 100]]

    num_open, num_close = len(open_scores['q_id']), len(closed_scores['q_id'])
    print(f'num_open {num_open} || num_close {num_close}')
    print()
    print(tabulate(table_list, headers=['Metric', 'Performance']))

    rst = {}
    if report_hit_samples:
        rst.update({'hit_sample_ids': hit_sample_ids})
    if report_not_hit_samples:
        rst.update({'not_hit_sample_ids': not_hit_sample_ids})
    return tabulate(table_list, headers=['Metric', 'Performance']), rst


def get_metrics(cfg):
    print(f'\n|+_+|{"=" * 15}| eval results on {cfg.result_file} |{"=" * 15}|+_+|')

    samples = []
    with open(cfg.result_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # perform evaluation
    report_hit_samples = cfg.hit_sample_file is not None
    report_not_hit_samples = cfg.not_hit_sample_file is not None
    results, records = evaluate(samples, report_hit_samples, report_not_hit_samples)

    directory = os.path.dirname(cfg.metric_output_file)
    os.makedirs(directory, exist_ok=True)

    with open(cfg.metric_output_file, 'w', encoding='utf-8') as f:
        f.write(results)
    print(f'Save results to {cfg.metric_output_file}')

    if cfg.hit_sample_file is not None:
        with open(cfg.hit_sample_file, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['id', 'question', 'pred', 'answer', 'answer_type', 'image'])
            for item in samples:
                if item['question_id'] in records['hit_sample_ids']:
                    id = item['question_id']

                    if 'question' in item:
                        question = item['question']
                    elif 'prompt' in item:
                        question = item['prompt']

                    answer_type = item['answer_type']
                    image = item['image'] if 'image' in item else None

                    if 'gt' in item:
                        gt = str(item['gt']).lower()
                    elif 'ground_truth' in item:
                        gt = str(item['ground_truth']).lower()

                    if 'text' in item:
                        pred = str(item['text']).lower()
                    elif 'prediction' in item:
                        pred = str(item['prediction']).lower()

                    csv_writer.writerow([id, question, pred, gt, answer_type, image])
        print(f'Save hit sample to {cfg.hit_sample_file}')

    if cfg.not_hit_sample_file is not None:
        with open(cfg.not_hit_sample_file, 'w', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['id', 'question', 'pred', 'answer', 'answer_type', 'image'])
            for item in samples:
                if item['question_id'] in records['not_hit_sample_ids']:
                    id = item['question_id']
                    if 'gt' in item:
                        gt = str(item['gt']).lower()
                    elif 'ground_truth' in item:
                        gt = str(item['ground_truth']).lower()
                    if 'text' in item:
                        pred = str(item['text']).lower()
                    elif 'prediction' in item:
                        pred = str(item['prediction']).lower()
                    if 'question' in item:
                        question = item['question']
                    elif 'prompt' in item:
                        question = item['prompt']
                    answer_type = item['answer_type']
                    image = item['image'] if 'image' in item else None
                    csv_writer.writerow([id, question, pred, gt, answer_type, image])

        print(f'Save not hit sample to {cfg.not_hit_sample_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', type=str, default=None, help='path to result file')
    parser.add_argument('--metric_output_file', type=str, default=None, help='path to output file')
    parser.add_argument('--hit_sample_file', type=str, default=None, help='path to error sample file')
    parser.add_argument('--not_hit_sample_file', type=str, default=None,
                        help='path to not hit(not equal gt for CLOSE, recall score < 0.5 for OPEN) samples file')

    cfg = parser.parse_args()

    get_metrics(cfg)
