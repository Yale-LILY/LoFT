import os
import sys
sys.path.append(os.path.abspath("../.."))

import pdb
import json
import jsonlines
from tqdm import tqdm
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))

""" ideal dict:
{
    "nt_id": xxx,
    "csv_id": xxx,
    "nl_description": xxx,
    "prog_descriptions": [
      {
        "program": xxx,
        "is_correct": xxx,
        "prob": xxx
      },
      ...
    ],
    "source": xxx
}
"""


#! inputs
data_shard_root = os.path.join(paths.train_output_root, "data_shard_with_dep")
results_root = os.path.join(paths.train_output_root, "sasp_results")
total_examples_path = os.path.join(paths.logicnlg_root, "train_lm.json")

create_path(results_root)

#! outputs
processed_results_root = create_path(os.path.join(paths.train_output_root, "processed_sasp_results"))

print("Reading total examples...")
with open(total_examples_path, 'r') as f:
    total_examples = json.load(f)


def _is_positive_statement(statement, csv_id):
    csv_key = csv_id + ".html.csv"
    statements = total_examples[csv_key][0]
    true_or_false_list = total_examples[csv_key][1]
    assert len(statements) == len(true_or_false_list)
    for s, tf in zip(statements, true_or_false_list):
        if s == statement:
            return True if tf == 1 else False

    return False  # ! statement not in dataset


print("Traversing files...")
# ! traverse files under data_shard
for path, dir_list, file_list in os.walk(data_shard_root):
    for file in tqdm(file_list):
        shard_path = os.path.join(path, file)
        fname = os.path.splitext(file)[0]
        result_file = fname + "-result.json"
        processed_result_file = fname + "-result-processed.json"
        result_path = os.path.join(results_root, result_file)
        processed_result_path = os.path.join(processed_results_root, processed_result_file)

        ideal_dict_list = list()
        with open(shard_path, "r+", encoding="utf8") as shard, open(result_path, 'r') as result:
            js_lines = jsonlines.Reader(shard)
            js_dicts = json.load(result)
            for js_line, (nt_id, d) in zip(js_lines, js_dicts.items()):
                # if not _is_positive_statement(js_line["question"], js_line["context"]):
                #     continue
                assert js_line["context"] == nt_id
                ideal_dict = dict()
                ideal_dict["nt_id"] = nt_id
                ideal_dict["csv_id"] = js_line["context"]
                ideal_dict["nl_description"] = js_line["question"]
                ideal_dict["prog_descriptions"] = d["hypotheses"]
                ideal_dict["source"] = fname
                ideal_dict_list.append(ideal_dict)
        json.dump(ideal_dict_list, open(processed_result_path, 'w'))

print("ALL DONE!")
