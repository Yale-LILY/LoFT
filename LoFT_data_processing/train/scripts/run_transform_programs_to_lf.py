import os
import sys
sys.path.append(os.path.abspath("../.."))

import pdb
import json
from train.modules.prog2lf import prog2lf
from tqdm import tqdm
from colorama import Style, Fore
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))

#! inputs
processed_results_root = os.path.join(paths.train_output_root, "processed_sasp_results")

#! outputs
processed_results_lf_root = create_path(os.path.join(paths.train_output_root, "processed_sasp_results_lf"))
error_log = os.path.join(paths.train_output_root, "prog2lf_error_log.out")

# generate logic forms
with open(error_log, 'w') as err:
    for path, dir_list, file_list in os.walk(processed_results_root):
        for file in file_list:
            error_cnt = 0

            tab2prog_path = os.path.join(processed_results_root, file)
            src_fname = os.path.splitext(file)[0]
            tgt_fname = src_fname + "-lf"
            tab2lf_path = os.path.join(processed_results_lf_root, tgt_fname + ".json")

            tab2lf_list = json.load(open(tab2prog_path, 'r'))
            for d in tqdm(tab2lf_list):
                for prog_envlp in d["prog_descriptions"]:
                    try:
                        prog_envlp["program_lf"] = prog2lf(prog_envlp["program"], d["csv_id"])
                    except:
                        err.write(d["nt_id"] + '\n')
                        err.write(d["csv_id"] + '\n')
                        err.write(d["nl_description"] + '\n')
                        err.write(d["source"] + '\n')
                        err.write(prog_envlp["program"] + '\n')
                        err.write('----------------------------------------\n\n')
                        error_cnt += 1
                        continue

            print(Style.BRIGHT + Fore.GREEN + f"file: {file}, error count: {error_cnt}" + Style.RESET_ALL)
            json.dump(tab2lf_list, open(tab2lf_path, 'w'), indent = 4)

print("ALL DONE!")
