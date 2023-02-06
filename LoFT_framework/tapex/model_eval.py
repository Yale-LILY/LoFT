# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb
import re
import numpy as np
from collections import defaultdict
from re import RegexFlag
from tqdm import tqdm
import nltk
from typing import List, Dict
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def extract_structure_data(plain_text_content: str):
    # extracts lines starts with specific flags
    # map id to its related information
    data = []

    predict_outputs = re.findall(
        "^D.+", plain_text_content, RegexFlag.MULTILINE)
    ground_outputs = re.findall(
        "^T.+", plain_text_content, RegexFlag.MULTILINE)
    source_inputs = re.findall("^S.+", plain_text_content, RegexFlag.MULTILINE)

    for predict, ground, source in zip(predict_outputs, ground_outputs, source_inputs):
        try:
            predict_id, _, predict_clean = predict.split('\t')
            ground_id, ground_clean = ground.split('\t')
            source_id, source_clean = source.split('\t')
            assert predict_id[2:] == ground_id[2:]
            assert ground_id[2:] == source_id[2:]
        except Exception:
            print("An error occurred in source: {}".format(source))
            continue
        data.append((predict_clean, ground_clean,
                    source_clean, predict_id[2:]))

    return data


#! =================================================================================================================

def evaluate(data: List, target_delimiter: str):
    def evaluate_example(_predict_str: str, _ground_str: str):
        _predict_spans = _predict_str.split(target_delimiter)
        _ground_spans = _ground_str.split(target_delimiter)
        _predict_values = defaultdict(lambda: 0)
        _ground_values = defaultdict(lambda: 0)
        for span in _predict_spans:
            try:
                _predict_values[float(span)] += 1
            except ValueError:
                _predict_values[span.strip()] += 1
        for span in _ground_spans:
            try:
                _ground_values[float(span)] += 1
            except ValueError:
                _ground_values[span.strip()] += 1
        _is_correct = _predict_values == _ground_values
        return _is_correct

    correct_num = 0
    correct_arr = []
    total = len(data)

    for example in data:
        predict_str, ground_str, source_str, predict_id = example
        is_correct = evaluate_example(predict_str, ground_str)
        if is_correct:
            correct_num += 1
        correct_arr.append(is_correct)

    print("Correct / Total : {} / {}, Denotation Accuracy : {:.3f}".format(correct_num,
          total, correct_num / total))
    return correct_arr


def evaluate_generate_file(generate_file_path, target_delimiter):
    with open(generate_file_path, "r", encoding="utf8") as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content)
        correct_arr = evaluate(data, target_delimiter)
        # write into eval file
        eval_file_path = generate_file_path + ".eval"
        eval_file = open(eval_file_path, "w", encoding="utf8")
        eval_file.write("Score\tPredict\tGolden\tSource\tID\n")
        for example, correct in zip(data, correct_arr):
            eval_file.write(str(correct) + "\t" + "\t".join(example) + "\n")
        eval_file.close()


#! =================================================================================================================


def logicnlg_evaluate(data: List, reference_dict: Dict):
    sent_bleus_1, sent_bleus_2, sent_bleus_3 = [], [], []
    for example in tqdm(data, total=len(data)):
        predict_str, ground_str, source_str, predict_id = example
        references = reference_dict[ground_str]
        references = [reference.lower().split() for reference in references]
        hypothesis = predict_str.lower().split()
        sent_bleu_1 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(1, 0, 0))
        sent_bleu_2 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0))
        sent_bleu_3 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33))
        sent_bleus_1.append(sent_bleu_1)
        sent_bleus_2.append(sent_bleu_2)
        sent_bleus_3.append(sent_bleu_3)

    return sent_bleus_1, sent_bleus_2, sent_bleus_3

def get_bleu_scores(references, hypothesis, sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4):
    sent_bleu_1 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(1, 0, 0))
    sent_bleu_2 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0))
    sent_bleu_3 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33))
    sent_bleu_4 = nltk.translate.bleu_score.sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))
    return sent_bleus_1 + [sent_bleu_1], sent_bleus_2 + [sent_bleu_2], sent_bleus_3 + [sent_bleu_3], sent_bleus_4 + [sent_bleu_4]

def logicnlg_evaluate(data: List, reference_dict: Dict, logicnlg_filename_dict: Dict, logic2text_filename_dict: Dict):
    sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4 = [], [], [], []
    for example in data:
        predict_str, ground_str, source_str, predict_id = example
        references = reference_dict[ground_str]
        references = [reference.lower().split() for reference in references]
        hypothesis = predict_str.lower().split()
        
        sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4 = get_bleu_scores(references, hypothesis, sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4)

    return sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4

def logic2text_for_logicnlg_evaluate(data: List, reference_dict: Dict, logicnlg_filename_dict: Dict, logic2text_filename_dict: Dict):
    sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4 = [], [], [], []
    for example in data:
        predict_str, ground_str, source_str, predict_id = example
        csv_name = logic2text_filename_dict[ground_str]
        references = [reference.lower().split() for reference in logicnlg_filename_dict[csv_name]]
        hypothesis = predict_str.lower().split()
        
        sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4 = get_bleu_scores(references, hypothesis, sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4)

    return sent_bleus_1, sent_bleus_2, sent_bleus_3, sent_bleus_4

def extract_logicnlg_reference_dict(ground_file_path):
    # reference_dict[ref_sent] = [ref_sent_1, ref_sent_2, ...]
    # filename_dict[csv_name] = [ref_sent_1, ref_sent_2, ...]
    reference_dict = {}
    filename_dict = {}
    data = json.load(open(ground_file_path, "r", encoding="utf8"))
    for csv_name in data:
        references = [example[0].lower() for example in data[csv_name]]
        for reference in references:
            reference_dict[reference] = references
        filename_dict[csv_name] = references
    return reference_dict, filename_dict

def extract_logic2text_filename_dict(ground_file_path):
    # filename_dict[ref_sent] = csv_name
    filename_dict = {}
    data = json.load(open(ground_file_path, "r", encoding="utf8"))
    for example in data:
        reference = example["sent"]
        csv_filename = example["url"].split("/")[-1]
        filename_dict[reference] = csv_filename
    return filename_dict


def logicnlg_evaluate_generate_file(generate_file_path, logicnlg_ground_file_path, logic2text_ground_file_path, evaluate_fn, with_logic2text_ref = True):
    with open(generate_file_path, "r", encoding='utf-8') as generate_f:
        file_content = generate_f.read()
        data = extract_structure_data(file_content)
        reference_dict, logicnlg_filename_dict = extract_logicnlg_reference_dict(logicnlg_ground_file_path)
        logic2text_filename_dict = extract_logic2text_filename_dict(logic2text_ground_file_path)

        if with_logic2text_ref:
            for reference, csv_name in logic2text_filename_dict.items():
                if csv_name in logicnlg_filename_dict:
                    logicnlg_filename_dict[csv_name].append(reference)
                
        bleu_1s, bleu_2s, bleu_3s, bleu_4s = evaluate_fn(data, reference_dict, logicnlg_filename_dict, logic2text_filename_dict)

        # write into eval file
        eval_file_path = generate_file_path + ".eval"
        eval_file = open(eval_file_path, "w", encoding="utf8")
        eval_file.write("Bleu 1\tBleu 2\tBleu 3\tPredict\tGolden\tSource\tID\n")
        for example, bleu_1, bleu_2, bleu_3, bleu_4 in zip(data, bleu_1s, bleu_2s, bleu_3s, bleu_4s):
            eval_file.write(str(bleu_1) + "\t" + str(bleu_2) + "\t" + str(bleu_3) + "\t" + str(bleu_4) + "\t" + "\t".join(example) + "\n")
        eval_file.close()

        bleu_1_avg = round(np.array(bleu_1s).mean(), 3)
        bleu_2_avg = round(np.array(bleu_2s).mean(), 3)
        bleu_3_avg = round(np.array(bleu_3s).mean(), 3)
        bleu_4_avg = round(np.array(bleu_4s).mean(), 3)
        
        return bleu_1_avg, bleu_2_avg, bleu_3_avg, bleu_4_avg