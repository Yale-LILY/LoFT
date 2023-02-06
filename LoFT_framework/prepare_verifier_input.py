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
import os, argparse

TABLE_PATH = "../LoFT_data_processing/data/logicnlg/all_csv/"

def extract_orig_inference_dict(inference_file_path):
    inference_dict = {}
    data = json.load(open(inference_file_path, "r", encoding="utf8"))
    for pred_id, example in enumerate(data):
        inference_dict[str(pred_id)] = example
    return inference_dict

def extract_structure_data(generate_file_path: str, output_file_path: str, inference_dict: Dict):
    # extracts lines starts with specific flags
    # map id to its related information
    with open(generate_file_path, "r", encoding='utf-8') as generate_f:
        plain_text_content = generate_f.read()

    data = []
    predictions = {}
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
        predictions[predict_id[2:]] = predict_clean
    
    for i in sorted([int(predict_id) for predict_id in predictions.keys()]):
        predict_id = str(i)
        predict_clean = predictions[predict_id]
        inference_dict[predict_id]["statement"] = predict_clean
    
    for i, example in enumerate(inference_dict):
        cur_example = inference_dict[example]

        table_file = cur_example["csv_id"]+".html.csv"
        cur_example["table_text"] = open(os.path.join(TABLE_PATH, table_file), "r").read()
        cur_example["id"] = i
        cur_example["label"] = 0

        data.append(cur_example)
    json.dump(data, open(output_file_path, "w", encoding="utf8"), indent=4)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_file_path", type=str, default = "processed_LoFT_data/LoFT_inference_input.json")
    parser.add_argument("--generate_file_path", type=str, default = "LoFT_predict/inference_raw.txt")
    parser.add_argument("--generate_json_path", type=str, default = "LoFT_predict/inference_verifier_input.json")
    args = parser.parse_args()


    inference_dict = extract_orig_inference_dict(args.ground_file_path)
    data = extract_structure_data(args.generate_file_path, args.generate_json_path, inference_dict)
