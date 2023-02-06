import pdb
import os
import json
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append('../..')
from translate.modules.logictools.TreeNode import *
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))


def translate_single(logic_form, csv_id):
    csv_path = os.path.join(paths.all_csv_root, csv_id + ".html.csv")
    csv_data = pd.read_csv(csv_path, sep='#')
    return Node(csv_data, logic_form).to_nl()


def translate_candidate_logic_forms():
    print("Translating candidate_logic_forms...")
    
    #! input
    candidate_logic_forms_root = os.path.join(paths.inference_output_root, "candidate_logic_forms")

    test_raw_path = os.path.join(paths.logicnlg_root, "test_lm.json")
    test_raw_data = json.load(open(test_raw_path, 'r'))
    
    #! output
    output_root = create_path(os.path.join(paths.inference_output_root, "candidate_logic_forms_translated"))
    
    output_examples = []
    for path, _, file_list in os.walk(candidate_logic_forms_root):
        tot = len(file_list)
        for file in tqdm(file_list):
            fname, _ = os.path.splitext(file)

            csv_id = fname[:-3]
            cols2lfs = json.load(open(os.path.join(path, file), 'r'))
            for col_id, lf_list in cols2lfs.items():
                for i in range(len(lf_list)):
                    nl = translate_single(lf_list[i][0], csv_id)
                    reasoning_type = lf_list[i][1]
                    lf_list[i][0] = nl

                    output_examples.append({
                        "csv_id": csv_id,
                        "logicform_nl": nl,
                        "reasoning_type": reasoning_type,
                        "col_ids": json.loads(col_id),
                        "table_title": test_raw_data[csv_id + ".html.csv"][0][2],
                    })
            json.dump(cols2lfs, open(os.path.join(output_root, csv_id + "-nl.json"), 'w'), indent = 4)
        
    
    json.dump(output_examples, open(os.path.join(paths.inference_output_root, "LoFT_inference_input.json"), 'w'), indent = 4)


def translate_valid_lf():
    print("Translating valid_lf...")
    
    #! input
    valid_lf_root = os.path.join(paths.train_output_root, "valid_lf")
    train_raw_path = os.path.join(paths.logicnlg_root, "train_lm.json")
    train_raw_data = json.load(open(train_raw_path, 'r'))

    output_examples = []
    err_count = 0

    for path, _, file_list in os.walk(valid_lf_root):
        for file in file_list:
            print(f"==> processing {file}...")
            fname, _ = os.path.splitext(file)
            example_list = json.load(open(os.path.join(path, file), 'r'))
            for example in tqdm(example_list):
                csv_id = example["csv_id"]
                cur_example = {
                    "csv_id": csv_id,
                    "nl_description": example["nl_description"],
                }

                for i in train_raw_data[f"{csv_id}.html.csv"]:
                    if i[0] == example["nl_description"]:
                        cur_example["col_ids"] = i[1]
                    cur_example["table_title"] = i[2]
                    

                for prog_envlp in example["prog_descriptions"]:
                    if prog_envlp["is_correct"]:
                        prog_envlp["logicform_nl"] = translate_single(prog_envlp["program_lf"], csv_id)
                        cur_example["logicform_nl"] = prog_envlp["logicform_nl"]
                        # cur_example["program_lf"] = prog_envlp["program_lf"]
                        break
                
                # if all excecution result is false, we use the first one, which has highest prob
                if "logicform_nl" not in cur_example:
                    temp_envlp = example["prog_descriptions"][0]
                    temp_envlp["logicform_nl"] = translate_single(temp_envlp["program_lf"], csv_id)
                    cur_example["logicform_nl"] = temp_envlp["logicform_nl"]
                    # cur_example["program_lf"] = temp_envlp["program_lf"]

                if "col_ids" in cur_example:
                    output_examples.append(cur_example)
                else:
                    err_count += 1
    output_path = os.path.join(paths.train_output_root, "LoFT_train_input.json")
    json.dump(output_examples, open(output_path, 'w'), indent = 4)
    print("==> Collect a total of {} training examples for LoFT, saved at {}".format(len(output_examples), output_path))
    print(err_count)


if __name__ == '__main__':
    translate_valid_lf()
    translate_candidate_logic_forms()
    print("translate.py: ALL DONE!")