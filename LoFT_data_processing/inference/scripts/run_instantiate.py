import os
import sys
sys.path.append('../..')

import pdb
import json
import pandas as pd
import random
from collections import defaultdict
from colorama import Fore, Style
from tqdm import tqdm

from utils.APIs import *
from inference.modules.templatize import *
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))


NUM_TRIALS = 600

#! inputs
all_lm_simplified_path = os.path.join(paths.logicnlg_root, "test_lm.json")

#! outputs
store_root = create_path(os.path.join(paths.inference_output_root, "candidate_logic_forms"))

all_lm_simplified = json.load(open(all_lm_simplified_path, 'r'))


def fill_template(table: pd.DataFrame, template):
    """ func could be:
        'FILTER', 'SUPERLATIVE', 
        'ORDINAL', 'SUPER_ARG', 'ORD_ARG', 'COMPARE', 
        'MAJORITY', 'AGGREGATION', 'Count', 'Only', 
        'Hop', 'Diff', 'And', 'Filter_all'
    """
    if template["func"] in ['FILTER', 'MAJORITY']:
        # input:  ['row', 'header', 'obj'/'str']
        assert len(template["args"]) == 3

        column_sample = random.choice(table.columns.tolist())
        entity_sample = random.choice(table[column_sample].tolist())

        flag = False
        for i in range(len(template["args"])):
            # * func node
            if isinstance(template["args"][i], dict):
                fill_template(table, template["args"][i])
            # * text node
            elif template["args"][i] == 'All_rows':
                template["args"][i] = 'all_rows'
            elif template["args"][i] == 'COL':
                template["args"][i] = str(column_sample)
            elif template["args"][i] == 'OBJ':
                if is_pure_string(str(entity_sample)):
                    flag = True
                template["args"][i] = str(entity_sample)

        if flag:
            template["func"] = random.choice(string_only[template["func"]])
        else:
            template["func"] = random.choice(cat2func[template["func"]])
    elif template["func"] == 'Hop':
        # input:  ['row', 'header']
        assert len(template["args"]) == 2
        assert template["args"][1] == 'COL'

        column_sample = random.choice(table.columns.tolist())

        # * func node
        if isinstance(template["args"][0], dict):
            fill_template(table, template["args"][0])
        # * text node
        elif template["args"][0] == 'All_rows':
            template["args"][0] = 'all_rows'
        else:
            raise ValueError()

        template["args"][1] = str(column_sample)
        if is_pure_string(str(column_sample)):
            template["func"] = 'str_hop'
        else:
            template["func"] = 'num_hop'
    elif template["func"] in ['SUPERLATIVE', 'SUPER_ARG', 'AGGREGATION', 'Filter_all']:
        # input:  ['row', 'header'] or ['row', 'header', 'obj']
        assert len(template["args"]) in [2, 3]

        column_sample = random.choice(table.columns.tolist())
        entity_sample = random.choice(table[column_sample].tolist())

        for i in range(len(template["args"])):
            # * func node
            if isinstance(template["args"][i], dict):
                fill_template(table, template["args"][i])
            # * text node
            elif template["args"][i] == 'All_rows':
                template["args"][i] = 'all_rows'
            elif template["args"][i] == 'COL':
                template["args"][i] = str(column_sample)
            elif template["args"][i] == 'OBJ':
                template["args"][i] = str(entity_sample)

        template["func"] = random.choice(cat2func[template["func"]])
    elif template["func"] in ['ORDINAL', 'ORD_ARG']:
        # input:  ['row', 'header', 'num']
        assert len(template["args"]) == 3

        #! first instantiate children
        column_sample = random.choice(table.columns.tolist())

        for i in range(3):
            # * func node
            if isinstance(template["args"][i], dict):
                fill_template(table, template["args"][i])
            # * text node
            elif template["args"][i] == 'All_rows':
                template["args"][i] = 'all_rows'
            elif template["args"][i] == 'COL':
                template["args"][i] = str(column_sample)
            elif template["args"][i] == 'NUM':
                template["args"][i] = str(random.randint(1, 4))
            else:
                raise NotImplementedError()

        #! then instantiate root
        template["func"] = random.choice(cat2func[template["func"]])
    elif template["func"] == 'COMPARE':
        # first compare the arguments, then choose the function based on the comparison result.
        assert len(template["args"]) == 2
        assert isinstance(template["args"][0], dict)

        fill_template(table, template["args"][0])
        if isinstance(template["args"][1], dict):
            fill_template(table, template["args"][1])
            res0 = Node(table, template["args"][0]).eval()
            res1 = Node(table, template["args"][1]).eval()

            if isinstance(res0, ExeError) or isinstance(res1, ExeError):
                return

            if is_pure_string(str(res0)) or is_pure_string(str(res1)):
                template["func"] = random.choice(string_only['COMPARE'])
            else:
                template["func"] = random.choice(cat2func['COMPARE'])
        elif template["args"][1] == 'OBJ':
            res = Node(table, template["args"][0]).eval()
            if isinstance(res, ExeError):
                return
            template["args"][1] = str(res)
            template["func"] = 'eq'
        else:
            raise NotImplementedError()
    elif template["func"] in ['Count', 'Only']:
        # input:  ['row']
        assert len(template["args"]) == 1

        fill_template(table, template["args"][0])
        template["func"] = random.choice(cat2func[template["func"]])
    elif template["func"] in ['And', 'Diff']:
        # input:  ['obj', 'obj']
        assert len(template["args"]) == 2
        assert (isinstance(template["args"][0], dict) and
                isinstance(template["args"][1], dict))

        fill_template(table, template["args"][0])
        fill_template(table, template["args"][1])

        template["func"] = random.choice(cat2func[template["func"]])
    else:
        raise NotImplementedError()


def random_instantiate(table: pd.DataFrame):
    # from 10000+ templates, weighted by Logic2Text distribution
    template, logic_type = select_a_template()
    fill_template(table, template)

    try:
        valid = Node(table, template).eval()
        if isinstance(valid, ExeError):
            status = Status.ERROR
        else:
            assert isinstance(valid, bool)
            if valid:
                status = Status.SUCCESS
            else:
                status = Status.FAIL
    except:
        status = Status.ERROR

    return template, logic_type, status


""" candidate_logic_forms/
for each table: xxx-lf.json
{
    "[x, x, ...]": {    # selected columns
        "count": [{ }, { }, { }],    # logic type 1
        "aggregation": [{ }, { }, { }],    # logic type 2
        ...
    },
    ...
}
"""


def instantiate_single_csv(table_file_path):
    fname_html, _ = os.path.splitext(os.path.basename(table_file_path))
    fname = fname_html[:-5]
    csv_id = fname
    # print(f"==> Generating logic forms for table {csv_id}...")

    table_data = pd.read_csv(table_file_path, sep='#')
    csv_dict = defaultdict(list)
    # for example in tqdm(all_lm_simplified[csv_id + ".html.csv"]):
    for example in all_lm_simplified[csv_id + ".html.csv"]:
        selected_cols = example[1]

        if str(selected_cols) in csv_dict.keys():
            continue

        sub_table_data = table_data.iloc[:, selected_cols]

        lf_pool = set()
        template2type = dict()
        err_cnt = 0
        add_pool_cnt = 0
        while len(lf_pool) < 50 and err_cnt < 50 and add_pool_cnt < 500:
            status = Status.FAIL
            num_iters = 0
            err_flag = False
            while status != Status.SUCCESS:
                instantiated_template, logic_type, status = random_instantiate(
                    sub_table_data)
                num_iters += 1
                if num_iters > 100:
                    err_flag = True
                    break
            if err_flag:
                err_cnt += 1
                continue
            template_str = json.dumps(instantiated_template)
            lf_pool.add(template_str)
            add_pool_cnt += 1
            template2type[template_str] = logic_type

        # for _ in range(NUM_TRIALS): #! try for a few times
        #     instantiated_template, logic_type, status = random_instantiate(sub_table_data)
        #     if status == Status.SUCCESS:
        #         template_str = json.dumps(instantiated_template)
        #         lf_pool.add(template_str)
        #         template2type[template_str] = logic_type
        #         if len(lf_pool) == 60:
        #             break
        #     elif status == Status.FAIL:
        #         continue
        #     elif status == Status.ERROR:
        #         continue
        #     else:
        #         raise ValueError("success, or fail, or error")

        csv_dict[str(selected_cols)] = [(json.loads(lf), template2type[lf])
                                        for lf in list(lf_pool)]

    store_path = os.path.join(store_root, fname + "-lf.json")
    json.dump(csv_dict, open(store_path, 'w'), indent = 4)


def instantiate_all_csv():
    for csv_id in tqdm(list(all_lm_simplified.keys())):
        table_file_path = os.path.join(paths.all_csv_root, csv_id)
        instantiate_single_csv(table_file_path)


if __name__ == '__main__':
    instantiate_all_csv()
    print("instantiate.py: ALL DONE!")
