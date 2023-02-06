# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pdb
import json
import os
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd
from tapex.processor import get_default_processor
from tapex.data_utils.preprocess_bpe import fairseq_bpe_translation
from tapex.data_utils.preprocess_binary import fairseq_binary_translation
from tapex.data_utils.format_converter import convert_fairseq_to_hf
from typing import List
import random
random.seed(233)

PROCESSED_DATASET_FOLDER = "processed_dataset"
TABLE_PATH = "../LoFT_data_processing/data/logicnlg/all_csv"

TABLE_PROCESSOR = get_default_processor(
    max_cell_length=15, max_input_length=1024)
MODEL_NAME = "bart.large"   # Options: bart.base, bart.large, tapex.base, tapex.large
logger = logging.getLogger(__name__)

def _select_content_from_file(table_file: str, selected_column_idxs: List[int]):
        assert ".csv" in table_file
        
        table_data = pd.read_csv(os.path.join(TABLE_PATH, table_file), sep='#')
        
        # the first line is header
        header = list(np.array(table_data.columns)[selected_column_idxs])

        rows = []
        for row_data in table_data.values:
            selected_values = np.array(row_data)[selected_column_idxs]
            rows.append([str(_) for _ in selected_values])

        return {
            "header": header,
            "rows": rows
        }

def build_logicnlg_fairseq_dataset(out_prefix, src_file, data_dir):
    """
    out_prefix: train, valid, or test
    src_file: train_lm.json, test_lm.json, val_lm.json
    data_dir: the directory where the processed dataset is to be saved
    """
    
    assert ".json" in src_file

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    input_f = open("{}/{}.src".format(data_dir, out_prefix),
                   "w", encoding="utf8")
    output_f = open("{}/{}.tgt".format(data_dir, out_prefix),
                    "w", encoding="utf8")

    with open(src_file, 'r') as f:
        examples = json.load(f)

    for example in tqdm(examples):
        if "nl_description" in example:
            sentence = example["nl_description"] # string
        else:
            sentence = "test data"
        program = example["logicform_nl"]
        selected_column_idxs = example["col_ids"] # list of int
        title = example["table_title"] # string
        table_file = example["csv_id"]+".html.csv"
        selected_table_content = _select_content_from_file(table_file, selected_column_idxs)
        prefix_text = f"{program} Table Title: {title}"
        answer = sentence

        if out_prefix == "train":
            input_source = TABLE_PROCESSOR.process_input(
                selected_table_content, prefix_text, answer).lower()
        else:
            input_source = TABLE_PROCESSOR.process_input(
                selected_table_content, prefix_text, []).lower()

        output_target = answer.lower()

        input_f.write(input_source + "\n")
        output_f.write(output_target + "\n")

    input_f.close()
    output_f.close()


def build_sqa_huggingface_dataset(fairseq_data_dir):
    convert_fairseq_to_hf(fairseq_data_dir, "train")
    convert_fairseq_to_hf(fairseq_data_dir, "valid")
    convert_fairseq_to_hf(fairseq_data_dir, "test")


def preprocess_sqa_dataset(processed_data_dir):
    fairseq_bpe_translation(processed_data_dir, resource_name=MODEL_NAME)
    fairseq_binary_translation(processed_data_dir, resource_name='bart.large')


if __name__ == '__main__':
    logger.info("You are using the setting of {}".format(MODEL_NAME))

    processed_logicnlg_data_dir = PROCESSED_DATASET_FOLDER

    logger.info("*" * 80)
    logger.info("Process the dataset and save the processed dataset in {}".format(
        processed_logicnlg_data_dir))

    loft_training_examples = json.load(open("processed_LoFT_data/LoFT_train_input.json", "r"))
    # split into train/dev with 8:2 portion
    train_data = []
    dev_data = []
    for example in loft_training_examples:
        if random.random() <= 0.8:
            train_data.append(example)
        else:
            dev_data.append(example)
    json.dump(train_data, open("processed_LoFT_data/LoFT_train_data.json", "w"), indent = 4)
    json.dump(dev_data, open("processed_LoFT_data/LoFT_dev_data.json", "w"), indent = 4)


    build_logicnlg_fairseq_dataset("train", "processed_LoFT_data/LoFT_train_data.json", processed_logicnlg_data_dir)
    build_logicnlg_fairseq_dataset("valid", "processed_LoFT_data/LoFT_dev_data.json", processed_logicnlg_data_dir)
    build_logicnlg_fairseq_dataset("test", "processed_LoFT_data/LoFT_inference_input.json", processed_logicnlg_data_dir)

    logger.info("*" * 80)
    logger.info(
        "Begin to BPE and build the dataset binaries in {}/bin".format(processed_logicnlg_data_dir))

    preprocess_sqa_dataset(processed_logicnlg_data_dir)
    
    logger.info("All Done!")
