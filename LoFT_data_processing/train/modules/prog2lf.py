import pdb
from typing import List
from train.modules.funcs import *
from colorama import Fore, Style
import json
import re, os

from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../..'))

print("Loading replace dictionary...")
replace_dict_path = os.path.join(paths.logicnlg_root, "replace_dictionary.json")
header_pattern = r"(?<=r\.).+(?=-)"
replace_dict = json.load(open(replace_dict_path, 'r'))


def is_v(arg: str):
    return "v" in arg and arg[len("v"):].isdigit()


def is_header(arg: str):
    return 'r.' in arg and '-' in arg


def restore_header(header: str):
    """header e.g.
    "r.year-date"
    """
    res = re.search(header_pattern, header)
    return res.group() if res else ""


def restore_cell(cell: str, csv_id: str):
    if csv_id in replace_dict.keys():
        if cell in replace_dict[csv_id].keys():
            return replace_dict[csv_id][cell]
    return cell


def resolve_op(raw_op: str, csv_id: str):
    """raw_op e.g.
    'filter_eq 1954-xx-xx r.year-date all_rows'
    """
    split = raw_op.split(' ')  
    func = split.pop(0)
    argtype = func2argtype[func]
    try:
        if argtype == 'ROW':
            assert len(split) == 1
            assert split[-1] == 'all_rows' or is_v(split[-1])
            args = split
        elif argtype == 'OBJ':
            assert len(split) == 1
            assert split[-1] == 'all_rows' or is_v(split[-1])
            args = split
        elif argtype == 'BOOL':
            assert len(split) == 1
            assert is_v(split[-1])
            args = split
        elif argtype == 'ROW_HEAD':
            assert len(split) == 2
            assert split[-2] == 'all_rows' or is_v(split[-2])
            assert is_header(split[-1])
            args = [split[-2], restore_header(split[-1])]
        elif argtype == 'ROW_HEAD_OBJ': # ! notice here split is in reverse order
            assert len(split) >= 3
            assert split[-1] == 'all_rows' or is_v(split[-1])
            assert is_header(split[-2])
            args = [split[-1], restore_header(split[-2]), restore_cell(' '.join(split[:-2]), csv_id)]
        elif argtype == 'ROW_ROW_HEAD':
            assert len(split) == 3
            assert split[-3] == 'all_rows' or is_v(split[-3])
            assert split[-2] == 'all_rows' or is_v(split[-2])
            assert is_header(split[-1])
            args = [split[-3], split[-2], restore_header(split[-1])]
        elif argtype == 'OBJ_OBJ':
            assert len(split) >= 2
            if is_v(split[-1]) and not is_v(split[0]):
                args = [split[-1], restore_cell(' '.join(split[:-1]), csv_id)]
            elif is_v(split[0]) and not is_v(split[-1]):
                args = [split[0], restore_cell(' '.join(split[1:]), csv_id)]
            elif is_v(split[0]) and is_v(split[-1]):
                assert len(split) == 2
                args = split
            else:
                raise NotImplementedError()
        elif argtype == 'ROW_ROW':
            assert len(split) == 2
            assert split[-1] == 'all_rows' or is_v(split[-1])
            assert split[-2] == 'all_rows' or is_v(split[-2])
            args = split
        elif argtype == 'BOOL_BOOL':
            assert len(split) == 2
            assert is_v(split[-1])
            assert is_v(split[-2])
            args = split
        else:
            raise ValueError()
    except:
        pdb.set_trace()
        print(Style.BRIGHT + Fore.RED + "ERROR: resolving op fails!" + Style.RESET_ALL)
        print(raw_op)

    for i in range(len(args)):
        #! handle "top_-_5"
        if '_' in args[i] and args[i] != 'all_rows':
            args[i] = args[i].replace('_', ' ')
    
    """
    func e.g.: 'filter_eq'
    args e.g.: ['all_rows', 'year', '1954']
    """
    return func, args


def resolve_prog(prog: str, csv_id: str) -> List:
    """prog e.g.
    "{ filter_eq 1954-xx-xx r.year-date all_rows } 
     { filter_str_contain_any umbrella r.single-string v0 } 
     { filter_str_contain_any david whitfield r.artist-string v1 } <END>"
    """
    ops = list()
    assert " <END>" in prog
    prog = prog[:-(len(" <END>"))]
    raw_ops = prog.split(" } { ")
    raw_ops[0] = raw_ops[0][len("{ "):]
    raw_ops[-1] = raw_ops[-1][:-len(" }")]
    for raw_op in raw_ops:
        op = list()
        func, args = resolve_op(raw_op, csv_id)
        op.append(func)
        op.append(args)
        ops.append(op)

    """ops e.g.
    [
        ['filter_eq', ['all_rows', 'year', '1954']], 
        ['filter_str_contain_any', ['v0', 'single', 'umbrella']], 
        ['filter_str_contain_any', ['v1', 'artist', 'david whitfield']]
    ]
    """
    return ops


def build_lf_tree(ops: List):
    """ops e.g.
    [
        ['filter_eq', ['all_rows', 'year', '1954']], 
        ['filter_str_contain_any', ['v0', 'single', 'umbrella']], 
        ['filter_str_contain_any', ['v1', 'artist', 'david whitfield']]
    ]
    """
    #! build logic form out of program
    lf_records = list()
    is_root_records = list()

    for op in ops:
        lf_node = dict()
        func, args = op[0], op[1]
        assert isinstance(func, str) and isinstance(args, list)

        lf_node["func"] = func
        argtype = func2argtype[func]
        operands = all_funcs[func]['operands']
        lf_node["args"] = list()
        for i in range(len(args)):
            arg = args[i]
            type = operands[i]
            if is_v(arg):
                if argtype == 'ROW_HEAD_OBJ' and type == 'obj':   #! fake v 
                    lf_node["args"].append(arg)
                else:
                    idx = int(arg[len("v"):])
                    assert idx <= len(lf_records) - 1, Style.BRIGHT + Fore.RED + "v out of range!" + Style.RESET_ALL
                    prev_lf_node = lf_records[idx]
                    lf_node["args"].append(prev_lf_node)
                    is_root_records[idx] = False
            else:
                lf_node["args"].append(arg)

        lf_records.append(lf_node)
        is_root_records.append(True)

    assert len(lf_records) == len(is_root_records)
    assert is_root_records[-1] == True

    #! join root nodes
    roots = list()
    for i in range(len(lf_records)):
        if is_root_records[i]:
            roots.append(i)

    def _join_roots(root_id_list: List):
        if len(root_id_list) == 1:
            if all_funcs[lf_records[root_id_list[0]]["func"]]["output"] == 'bool':
                return lf_records[root_id_list[0]]
            elif all_funcs[lf_records[root_id_list[0]]["func"]]["output"] in ['row', 'num']:
                single_lf = dict()
                single_lf["func"] = 'is_not_none'
                single_lf["args"] = [lf_records[root_id_list[0]]]
                return single_lf
            else:
                raise ValueError("base case error")

        new_lf = dict()
        new_lf["func"] = 'and'
        new_lf["args"] = [None, None]
        
        #! fill the first argument
        root_id = root_id_list.pop(0)
        first_arg = lf_records[root_id]
        if all_funcs[first_arg["func"]]["output"] == 'bool':
            new_lf["args"][0] = first_arg
        elif all_funcs[first_arg["func"]]["output"] in ['row', 'num']:   
            new_lf["args"][0] = dict()
            new_lf["args"][0]["func"] = 'is_not_none'
            new_lf["args"][0]["args"] = [first_arg]
        else:
            raise ValueError("output should be either bool or row or num type") 

        #! fill the second argument
        new_lf["args"][1] = _join_roots(root_id_list)

        return new_lf

    lf = _join_roots(root_id_list=roots)
    assert all_funcs[lf["func"]]["output"] == 'bool'

    #! change function names
    def _change_func_names(top_level_lf):
        orig_func = top_level_lf["func"]
        if orig_func in changeFuncName.keys():
            top_level_lf["func"] = changeFuncName[orig_func]
        for arg in top_level_lf["args"]:
            if isinstance(arg, dict):
                _change_func_names(arg)
    
    _change_func_names(top_level_lf=lf)

    """lf e.g.
    {
        'func': 'is_not_none', 
        'args': [
            {
                'func': 'filter_str_eq', 
                'args': [
                    {
                        'func': 'filter_str_eq', 
                        'args': [
                            {
                                'func': 'filter_eq', 
                                'args': [
                                    'all rows', 
                                    'year', 
                                    '1954'
                                ]
                            }, 
                            'single', 
                            'umbrella'
                        ]
                    }, 
                    'artist', 
                    'david whitfield'
                ]
            }
        ]
    }
    """
    return lf


def prog2lf(prog: str, csv_id: str):
    ops = resolve_prog(prog, csv_id)
    lf = build_lf_tree(ops)
    return lf


if __name__ == '__main__':
    #! test "restore_header"
    # header = "r.year-date"
    # print(restore_header(header))
    
    #! test "resolve_op"
    # raw_op = 'argmax all_rows r.score-num2'
    # func, args = resolve_op(raw_op)
    # print(func, args)

    #! test "resolve_prog"
    # prog = "{ filter_eq 2.0 r.away_team_score-number all_rows } { filter_str_contain_any 9.13 ( 67 ) r.home_team_score-string v0 } { minimum all_rows v1 r.crowd-number } <END>"
    # ops = resolve_prog(prog)
    # print(ops)
    
    #! test "prof2lf"
    # prog = "{ filter_eq 1954-xx-xx r.year-date all_rows } { filter_str_contain_any umbrella r.single-string v0 } { filter_str_contain_any david whitfield r.artist-string v1 } <END>"
    csv_id = "2-18843095-2"
    prog = "{ filter_eq xxxx-10-30 r.date-date all_rows } { filter_eq xxxx-10-23 r.date-date all_rows } { row_greater v1 v0 r.attendance-number } <END>"
    lf = prog2lf(prog, csv_id)
    print(lf)
    print("ALL DONE!")
