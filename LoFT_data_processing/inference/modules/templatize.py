import pdb
import os
import sys
sys.path.append('../..')

import json
from tqdm import tqdm
import random
from colorama import Fore, Style
from collections import Counter
from copy import deepcopy


from utils.APIs import *
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))

def build_abstract_tree(logic_form):
    template = dict()

    #! pre-order: templatize root
    func = logic_form["func"]
    template["func"] = func2cat[func]   # category
    template["args"] = []

    #! recursively build tree
    args = logic_form["args"]
    types_of_args = APIs[func]["argument"]
    assert len(args) == len(types_of_args)

    for i in range(len(types_of_args)):
        arg = args[i]
        type = types_of_args[i]
        if isinstance(arg, str):    # leaf: text node
            # arg_type can be: row; header; obj; num; bool
            if type == 'row':
                assert arg == 'all_rows'
                placeholder = 'All_rows'
            elif type == 'header':
                placeholder = 'COL'
            elif type == 'obj':
                placeholder = 'OBJ'
            elif type == 'num':
                assert template["func"] in ['ORD_ARG', 'ORDINAL']
                placeholder = 'NUM'
            elif type == 'bool':
                raise ValueError('leaf node cannot be bool')
            else:
                raise NotImplementedError()
            template["args"].append(placeholder)
        elif isinstance(arg, dict):  # non-leaf: function node
            template["args"].append(build_abstract_tree(arg))
        else:
            raise NotImplementedError()

    return template


def create_templates_single(type_file_path: str):
    fname, _ = os.path.splitext(os.path.basename(type_file_path))

    print("------------------------------------------------")
    print(f"Building templates for type: {fname} ...")

    with open(type_file_path) as f:
        data_in = json.load(f)  # a list of dict

    template_cnt = Counter()    # key: str of template; val: count
    pool = set()  # a set of str of template envelope
    total_num = 0

    for dictionary in tqdm(data_in):
        logic_form = dictionary["logic"]

        template_envelope = dict()
        template_envelope["logic_type"] = fname
        template = build_abstract_tree(logic_form)
        template_envelope["template"] = template
        pool.add(json.dumps(template_envelope))

        template_cnt[json.dumps(template)] += 1
        total_num += 1

    template_envelopes = list()
    for template_envelope_str in pool:
        template_envelope = json.loads(template_envelope_str)
        template = template_envelope["template"]
        template_str = json.dumps(template)
        template_envelope["count"] = template_cnt[template_str]
        template_envelope["ratio(%)"] = round(
            float(template_cnt[template_str]) / total_num * 100, 3)
        template_envelopes.append(template_envelope)

    print("------------------------------------------------")
    print()
    return template_envelopes


def create_templates_all():
    all_template_envelopes_path = os.path.join(
        paths.logic2text_root, "all_template_envelopes.json")
    all_template_envelopes = json.load(open(all_template_envelopes_path, 'r'))

    num2ratio = dict()
    for k, v in all_template_envelopes.items():
        num = k
        ratio = v["ratio(%)"]
        num2ratio[num] = ratio

    print("Total number of templates:", len(all_template_envelopes))

    type2templates = dict()

    #! collect templates for the split of each type
    for path, dir_list, file_list in os.walk(os.path.join(paths.logic2text_root, "split_by_type")):
        for file in file_list:
            type_file_path = os.path.join(path, file)
            template_envelopes = create_templates_single(type_file_path)
            fname, _ = os.path.splitext(os.path.basename(type_file_path))
            type2templates[fname] = template_envelopes

    #! store templates for each type into json file
    for path, dir_list, file_list in os.walk(os.path.join(paths.logic2text_root, "template_envelopes")):
        for file in file_list:
            tpl_file_path = os.path.join(path, file)
            fname, _ = os.path.splitext(os.path.basename(tpl_file_path))
            with open(tpl_file_path, "w") as js:
                json.dump(type2templates[fname[:-4]], js)

    #! create entire template pool
    big_pool = set()
    for k, v in type2templates.items():
        print(f"{k} has {len(v)} templates.")
        for t in v:
            big_pool.add(json.dumps(t))

    print(Fore.YELLOW + Style.BRIGHT +
          f"{len(big_pool)} templates in total." + Style.RESET_ALL)
    print(Fore.BLUE + Style.BRIGHT + "Creating templates done!" + Style.RESET_ALL)

    return all_template_envelopes, num2ratio, type2templates


all_template_envelopes, num2ratio, type2templates = create_templates_all()


def select_a_template():
    def _weighted_random_choice(weight_data):
        total = sum(weight_data.values())
        ra = random.uniform(0, total)
        curr_sum = 0
        ret = None
        for k in weight_data.keys():
            curr_sum += weight_data[k]
            if ra <= curr_sum:
                ret = k
                break
        return ret

    number = _weighted_random_choice(num2ratio)
    template_envelope = all_template_envelopes[number]
    return deepcopy(template_envelope["template"]), template_envelope["logic_type"]
