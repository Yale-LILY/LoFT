import os
import sys
sys.path.append(os.path.abspath("../.."))

from utils.APIs import *

import os
import json
import pandas as pd
import numpy as np
import pdb
from colorama import Style, Fore
from tqdm import tqdm
from collections import defaultdict

from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))

import warnings
warnings.filterwarnings("ignore")


def _exist_func(logic_form, func):
    if not isinstance(logic_form, dict):
        return False

    if logic_form["func"] == func:
        return True

    for arg in logic_form["args"]:
        if isinstance(arg, dict):
            if _exist_func(arg, func):
                return True

    return False


def sasp_validate():
    #! inputs
    processed_results_lf_root = os.path.join(paths.train_output_root, "processed_sasp_results_lf")
    
    #! outputs
    validation_records_root = create_path(os.path.join(paths.train_output_root, "validation_records"))
    valid_lf_root = create_path(os.path.join(paths.train_output_root, "valid_lf"))

    for path, dir_list, file_list in os.walk(processed_results_lf_root):
        for file in tqdm(file_list):
            # print(Style.BRIGHT + Fore.BLUE + f"Validating {file}" + Style.RESET_ALL)
            fname, _ = os.path.splitext(file)
            src_path = os.path.join(path, file)
            
            valid_lf_path = os.path.join(valid_lf_root, fname.replace(
                "-result-processed-lf", "-valid-lf") + ".json")
            record_path = os.path.join(validation_records_root, fname.replace(
                "-result-processed-lf", "-record") + ".json")

            valid_lf_list = []

            #! assorted statistics
            inconsistent_cnt = 0
            error_cnt = 0
            validation_true_cnt = 0
            validation_false_cnt = 0
            double_true_cnt = 0
            total_cnt = 0
            record_dict = dict()
            record_dict["records"] = list()
            record_dict["errors"] = defaultdict(int)

            with open(src_path, 'r') as src, \
                    open(record_path, 'w') as rcd, \
                        open(valid_lf_path, 'w') as vlf:
                ideal_dict_list = json.load(src)
                for ideal_dict in ideal_dict_list:
                    if len(ideal_dict["prog_descriptions"]) == 0:
                        continue

                    #! get csv file info
                    csv_id = ideal_dict["csv_id"]
                    csv_file_path = os.path.join(
                        paths.all_csv_root, csv_id + ".html.csv")
                    csv_data = pd.read_csv(csv_file_path, sep='#')

                    #! get logic forms
                    scores = [d["prob"]
                              for d in ideal_dict["prog_descriptions"]]
                    max_id = np.argmax(scores)
                    best_envelope = ideal_dict["prog_descriptions"][max_id]

                    envelope = best_envelope
                    is_correct = envelope["is_correct"]
                    logic_form = envelope["program_lf"]

                    try:
                        validate_result = Node(csv_data, logic_form).eval()
                    except Exception as e:
                        print(e)
                        pdb.set_trace()
                    total_cnt += 1

                    if validate_result == True:
                        validation_true_cnt += 1
                        if is_correct == True:
                            double_true_cnt += 1
                    elif validate_result == False:
                        validation_false_cnt += 1

                    if not (
                        is_correct == True and validate_result == True
                        or
                        is_correct == False and (
                            validate_result == False or isinstance(validate_result, ExeError))
                    ):
                        record = dict()
                        record["csv id"] = csv_id
                        record["program"] = envelope["program"]
                        record["is correct"] = is_correct
                        record["logic form"] = logic_form
                        record["validation result"] = str(validate_result)
                        record_dict["records"].append(record)

                        inconsistent_cnt += 1
                        if isinstance(validate_result, ExeError):
                            error_cnt += 1
                            record_dict["errors"][str(
                                validate_result).split(":")[0]] += 1
                    else:
                        valid_lf_list.append(ideal_dict)

                if total_cnt == 0:
                    print(f"Pass {file}")
                    continue

                json.dump(valid_lf_list, vlf, indent = 4)

                record_dict["inconsistent count"] = inconsistent_cnt
                record_dict["error count"] = error_cnt
                record_dict["validation true count"] = validation_true_cnt
                record_dict["validation false count"] = validation_false_cnt
                record_dict["double true count"] = double_true_cnt
                record_dict["total count"] = total_cnt
                record_dict["inconsistent ratio"] = float(
                    inconsistent_cnt) / total_cnt
                record_dict["error ratio"] = float(error_cnt) / total_cnt
                record_dict["validation true ratio"] = float(validation_true_cnt) / total_cnt
                record_dict["validation false ratio"] = float(validation_false_cnt) / total_cnt
                record_dict["double true ratio"] = float(double_true_cnt) / total_cnt

                json.dump(record_dict, rcd, indent = 4)


if __name__ == '__main__':
    sasp_validate()
    print("ALL DONE!")
