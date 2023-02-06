import pdb
import re
import math
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum

import warnings
warnings.filterwarnings("ignore")

# * --------------------------------------------------------------------------------------------------------------------


categories = ["FILTER", "SUPERLATIVE", "ORDINAL", "SUPER_ARG",
              "ORD_ARG", "COMPARE", "MAJORITY", "AGGREGATION"]

ALL_TYPES = ["count", "unique", "comparative",
             "superlative", "ordinal", "aggregation", "majority"]

month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
             'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
             'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

# regex list

# number format:
'''
10
1.12
1,000,000
10:00
1st, 2nd, 3rd, 4th
'''
pat_num = r"([-+]?\s?\d*(?:\s?[:,.]\s?\d+)+\b|[-+]?\s?\d+\b|\d+\s?(?=st|nd|rd|th))"

pat_add = r"((?<==\s)\d+)"

# dates
pat_year = r"\b(\d\d\d\d)\b"
pat_day = r"\b(\d\d?)\b"
pat_month = r"\b((?:jan(?:uary)?|feb(?:ruary)?|mar(?:rch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?))\b"


# * --------------------------------------------------------------------------------------------------------------------


def create_APIs():
    APIs = dict()
    # --------------------------------------------------------------------------------
    # With only one argument
    #! count
    # input a table or a sub-table (as pd.DataFrame), output the table size
    APIs['count'] = {"argument": ['row'], 'output': 'num',
                     'function': lambda t:  len(t),
                     'tostr': lambda t: "count {{ {} }}".format(t),
                     'append': True}

    #! only
    APIs['only'] = {"argument": ['row'], 'output': 'bool',
                    "function": lambda t: len(t) == 1,
                    "tostr": lambda t: "only {{ {} }}".format(t),
                    'append': None}

    # --------------------------------------------------------------------------------
    #! hop
    # With only two argument and the first is row
    APIs['str_hop'] = {"argument": ['row', 'header'], 'output': 'obj',  # ! modified from 'str'
                       'function': lambda t, col:  hop_op(t, col),
                       'tostr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
                       'append': True}

    APIs['num_hop'] = {"argument": ['row', 'header'], 'output': 'obj',
                       'function': lambda t, col:  hop_op(t, col),
                       'tostr': lambda t, col: "hop {{ {} ; {} }}".format(t, col),
                       'append': True}

    #! -------------------- AGGREGATION --------------------
    APIs['avg'] = {"argument": ['row', 'header'], 'output': 'num',
                   "function": lambda t, col: agg(t, col, "avg"),
                   "tostr": lambda t, col: "avg {{ {} ; {} }}".format(t, col),
                   'append': True}

    APIs['sum'] = {"argument": ['row', 'header'], 'output': 'num',
                   "function": lambda t, col: agg(t, col, "sum"),
                   "tostr": lambda t, col: "sum {{ {} ; {} }}".format(t, col),
                   'append': True}

    #! -------------------- SUPERLATIVE --------------------
    APIs['max'] = {"argument": ['row', 'header'], 'output': 'obj',
                   "function": lambda t, col: nth_maxmin(t, col, order=1, max_or_min="max", arg=False),
                   "tostr": lambda t, col: "max {{ {} ; {} }}".format(t, col),
                   'append': True}

    APIs['min'] = {"argument": ['row', 'header'], 'output': 'obj',
                   "function": lambda t, col: nth_maxmin(t, col, order=1, max_or_min="min", arg=False),
                   "tostr": lambda t, col: "min {{ {} ; {} }}".format(t, col),
                   'append': True}

    #! -------------------- SUPER_ARG --------------------
    APIs['argmax'] = {"argument": ['row', 'header'], 'output': 'row',
                      'function': lambda t, col: nth_maxmin(t, col, order=1, max_or_min="max", arg=True),
                      'tostr': lambda t, col: "argmax {{ {} ; {} }}".format(t, col),
                      'append': False}

    APIs['argmin'] = {"argument": ['row', 'header'], 'output': 'row',
                      'function': lambda t, col:  nth_maxmin(t, col, order=1, max_or_min="min", arg=True),
                      'tostr': lambda t, col: "argmin {{ {} ; {} }}".format(t, col),
                      'append': False}

    #! -------------------- ORD_ARG --------------------
    # add for ordinal
    APIs['nth_argmax'] = {"argument": ['row', 'header', 'num'], 'output': 'row',
                          'function': lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="max", arg=True),
                          'tostr': lambda t, col, ind: "nth_argmax {{ {} ; {} ; {} }}".format(t, col, ind),
                          'append': False}

    APIs['nth_argmin'] = {"argument": ['row', 'header', 'num'], 'output': 'row',
                          'function': lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="min", arg=True),
                          'tostr': lambda t, col, ind: "nth_argmin {{ {} ; {} ; {} }}".format(t, col, ind),
                          'append': False}

    #! -------------------- ORDINAL --------------------
    APIs['nth_max'] = {"argument": ['row', 'header', 'num'], 'output': 'num',
                       "function": lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="max", arg=False),
                       "tostr": lambda t, col, ind: "nth_max {{ {} ; {} ; {} }}".format(t, col, ind),
                       'append': True}

    APIs['nth_min'] = {"argument": ['row', 'header', 'num'], 'output': 'num',
                       "function": lambda t, col, ind: nth_maxmin(t, col, order=ind, max_or_min="min", arg=False),
                       "tostr": lambda t, col, ind: "nth_min {{ {} ; {} ; {} }}".format(t, col, ind),
                       'append': True}

    #! diff
    # str
    APIs['diff'] = {"argument": ['obj', 'obj'], 'output': 'obj',  # ! modified from 'str'
                    'function': lambda t1, t2: obj_compare(t1, t2, type="diff"),
                    'tostr': lambda t1, t2: "diff {{ {} ; {} }}".format(t1, t2),
                    'append': True}

    #! -------------------- COMPARE --------------------
    # With only two argument and the first is not row
    # obj

    APIs['greater'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                       'function': lambda t1, t2:  obj_compare(t1, t2, type="greater"),
                       'tostr': lambda t1, t2: "greater {{ {} ; {} }}".format(t1, t2),
                       'append': False}

    APIs['less'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                    'function': lambda t1, t2:  obj_compare(t1, t2, type="less"),
                    'tostr': lambda t1, t2: "less {{ {} ; {} }}".format(t1, t2),
                    'append': True}

    APIs['eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                  'function': lambda t1, t2:  obj_compare(t1, t2, type="eq"),
                  'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
                  'append': None}

    APIs['not_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                      'function': lambda t1, t2: obj_compare(t1, t2, type="not_eq"),
                      'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
                      "append": None}

    APIs['round_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                        'function': lambda t1, t2:  obj_compare(t1, t2, round=True, type="eq"),
                        'tostr': lambda t1, t2: "round_eq {{ {} ; {} }}".format(t1, t2),
                        'append': None}

    # --------------------------------------------------------------------------------

    # str
    # APIs['str_eq'] = {"argument": ['str', 'str'], 'output': 'bool',
    #                   'function': lambda t1, t2:  t1 in t2 or t2 in t1,
    #                   'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
    #                   "append": None}
    APIs['str_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                      'function': lambda t1, t2:  obj_compare(t1, t2, type="eq"),
                      'tostr': lambda t1, t2: "eq {{ {} ; {} }}".format(t1, t2),
                      'append': None}

    # APIs['not_str_eq'] = {"argument": ['str', 'str'], 'output': 'bool',
    #                       'function': lambda t1, t2:  t1 not in t2 and t2 not in t1,
    #                       'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
    #                       "append": None}
    APIs['not_str_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                          'function': lambda t1, t2: obj_compare(t1, t2, type="not_eq"),
                          'tostr': lambda t1, t2: "not_eq {{ {} ; {} }}".format(t1, t2),
                          "append": None}

    # --------------------------------------------------------------------------------

    #! And
    # bool
    APIs['and'] = {"argument": ['bool', 'bool'], 'output': 'bool',
                   'function': lambda t1, t2:  t1 and t2,
                   'tostr': lambda t1, t2: "and {{ {} ; {} }}".format(t1, t2),
                   "append": None}

    #! -------------------- FILTER --------------------
    # filter
    # obj: num or str
    APIs["filter_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                         "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="eq"),
                         "tostr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                         'append': False}

    APIs["filter_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                             "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="not_eq"),
                             "tostr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                             'append': False}

    APIs["filter_less"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                           "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less"),
                           "tostr": lambda t, col, value: "filter_less {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": False}

    APIs["filter_greater"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                              "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater"),
                              "tostr": lambda t, col, value: "filter_greater {{ {} ; {} ; {} }}".format(t, col, value),
                              "append": False}

    APIs["filter_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                                 "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="greater_eq"),
                                 "tostr": lambda t, col, value: "filter_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                                 "append": False}

    APIs["filter_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                              "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="less_eq"),
                              "tostr": lambda t, col, value: "filter_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                              "append": False}

    # --------------------------------------------------------------------------------
    # filter string
    # With only three argument and the first is row

    # APIs["filter_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "row",
    #                          "function": lambda t, col, value: fuzzy_match_filter(t, col, value, negate=False),
    #                          "tostr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
    #                          'append': False}
    APIs["filter_str_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                             "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="eq"),
                             "tostr": lambda t, col, value: "filter_eq {{ {} ; {} ; {} }}".format(t, col, value),
                             'append': False}

    # APIs["filter_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "row",
    #                              "function": lambda t, col, value: fuzzy_match_filter(t, col, value, negate=True),
    #                              "tostr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
    #                              'append': False}
    APIs["filter_str_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "row",
                                 "function": lambda t, col, value: fuzzy_compare_filter(t, col, value, type="not_eq"),
                                 "tostr": lambda t, col, value: "filter_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                                 'append': False}
    # --------------------------------------------------------------------------------

    APIs["filter_all"] = {"argument": ['row', 'header'], "output": "row",
                          "function": lambda t, col: t,
                          "tostr": lambda t, col: "filter_all {{ {} ; {} }}".format(t, col),
                          'append': False}

    #! -------------------- MAJORITY --------------------
    # all
    # obj: num or str
    APIs["all_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                      "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="eq")),
                      "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                      "append": None}

    APIs["all_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: 0 == len(fuzzy_compare_filter(t, col, value, type="eq")),
                          "tostr": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}

    APIs["all_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="less")),
                        "tostr": lambda t, col, value: "all_less {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

    APIs["all_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="less_eq")),
                           "tostr": lambda t, col, value: "all_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": None}

    APIs["all_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="greater")),
                           "tostr": lambda t, col, value: "all_greater {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": None}

    APIs["all_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                              "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="greater_eq")),
                              "tostr": lambda t, col, value: "all_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                              "append": None}

    # most
    # obj: num or str
    APIs["most_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                       "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="eq")),
                       "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                       "append": None}

    APIs["most_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: len(t) // 3 > len(fuzzy_compare_filter(t, col, value, type="eq")),
                           "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": None}

    APIs["most_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                         "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="less")),
                         "tostr": lambda t, col, value: "most_less {{ {} ; {} ; {} }}".format(t, col, value),
                         "append": None}

    APIs["most_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                            "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="less_eq")),
                            "tostr": lambda t, col, value: "most_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                            "append": None}

    APIs["most_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                            "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="greater")),
                            "tostr": lambda t, col, value: "most_greater {{ {} ; {} ; {} }}".format(t, col, value),
                            "append": None}

    APIs["most_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                               "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="greater_eq")),
                               "tostr": lambda t, col, value: "most_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                               "append": None}

    # --------------------------------------------------------------------------------
    # all string
    # APIs["all_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
    #                       "function": lambda t, col, value: len(t) == len(fuzzy_match_filter(t, col, value)),
    #                       "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
    #                       "append": None}
    APIs["all_str_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: len(t) == len(fuzzy_compare_filter(t, col, value, type="eq")),
                          "tostr": lambda t, col, value: "all_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}

    # APIs["all_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
    #                           "function": lambda t, col, value: 0 == len(fuzzy_match_filter(t, col, value)),
    #                           "tostr": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
    #                           "append": None}
    APIs["all_str_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                              "function": lambda t, col, value: 0 == len(fuzzy_compare_filter(t, col, value, type="eq")),
                              "tostr": lambda t, col, value: "all_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                              "append": None}

    # most string
    # APIs["most_str_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
    #                        "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_match_filter(t, col, value)),
    #                        "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
    #                        "append": None}
    APIs["most_str_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: len(t) // 3 <= len(fuzzy_compare_filter(t, col, value, type="eq")),
                           "tostr": lambda t, col, value: "most_eq {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": None}

    # APIs["most_str_not_eq"] = {"argument": ['row', 'header', 'str'], "output": "bool",
    #                            "function": lambda t, col, value: len(t) // 3 > len(fuzzy_match_filter(t, col, value)),
    #                            "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
    #                            "append": None}
    APIs["most_str_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                               "function": lambda t, col, value: len(t) // 3 > len(fuzzy_compare_filter(t, col, value, type="eq")),
                               "tostr": lambda t, col, value: "most_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                               "append": None}

    # --------------------------------------------------------------------------------

    # **** new functions
    APIs["is_none"] = {"argument": ['obj'], "output": "bool",
                       "function": lambda t: is_none(t, negate=False),
                       "tostr": lambda t: "is_none {{ {} }}".format(t),
                       "append": None}

    APIs["is_not_none"] = {"argument": ['obj'], "output": "bool",
                           "function": lambda t: is_none(t, negate=True),
                           "tostr": lambda t: "is_not_none {{ {} }}".format(t),
                           "append": None}

    APIs["is_not"] = {"argument": ['bool'], "output": "bool",
                      "function": lambda t: t == False,
                      "tostr": lambda t: "is_not {{ {} }}".format(t),
                      "append": None}

    APIs["maximum"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                       "function": lambda r1, r2, col: row_maxmin(r1, r2, col, type='maximum'),
                       "tostr": lambda r1, r2, col: "maximum {{ {} ; {} ; {} }}".format(r1, r2, col),
                       "append": None}

    APIs["minimum"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                       "function": lambda r1, r2, col: row_maxmin(r1, r2, col, type='minimum'),
                       "tostr": lambda r1, r2, col: "minimum {{ {} ; {} ; {} }}".format(r1, r2, col),
                       "append": None}

    APIs["hop_str_contain_not_any"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                                       "function": lambda t, col, value: hop_obj(t, col, value, type='not_contain'),
                                       "tostr": lambda t, col, value: "hop_str_contain_not_any {{ {} ; {} ; {} }}".format(t, col, value),
                                       "append": None}

    APIs["hop_str_contain_any"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                                   "function": lambda t, col, value: hop_obj(t, col, value, type='contain'),
                                   "tostr": lambda t, col, value: "hop_str_contain_any {{ {} ; {} ; {} }}".format(t, col, value),
                                   "append": None}

    APIs["hop_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                      "function": lambda t, col, value: hop_obj(t, col, value, type='eq'),
                      "tostr": lambda t, col, value: "hop_eq {{ {} ; {} ; {} }}".format(t, col, value),
                      "append": None}

    APIs["hop_not_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                          "function": lambda t, col, value: hop_obj(t, col, value, type='not_eq'),
                          "tostr": lambda t, col, value: "hop_not_eq {{ {} ; {} ; {} }}".format(t, col, value),
                          "append": None}

    APIs["hop_less"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                        "function": lambda t, col, value: hop_obj(t, col, value, type='less'),
                        "tostr": lambda t, col, value: "hop_less {{ {} ; {} ; {} }}".format(t, col, value),
                        "append": None}

    APIs["hop_less_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: hop_obj(t, col, value, type='less_eq'),
                           "tostr": lambda t, col, value: "hop_less_eq {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": None}

    APIs["hop_greater"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                           "function": lambda t, col, value: hop_obj(t, col, value, type='greater'),
                           "tostr": lambda t, col, value: "hop_greater {{ {} ; {} ; {} }}".format(t, col, value),
                           "append": None}

    APIs["hop_greater_eq"] = {"argument": ['row', 'header', 'obj'], "output": "bool",
                              "function": lambda t, col, value: hop_obj(t, col, value, type='greater_eq'),
                              "tostr": lambda t, col, value: "hop_greater_eq {{ {} ; {} ; {} }}".format(t, col, value),
                              "append": None}

    APIs["row_diff"] = {"argument": ['row', 'row', 'header'], "output": "obj",
                        "function": lambda r1, r2, col: row_compare_1(r1, r2, col, type='diff'),
                        "tostr": lambda r1, r2, col: "row_diff {{ {} ; {} ; {} }}".format(r1, r2, col),
                        "append": None}

    APIs["same"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                    "function": lambda r1, r2, col: row_compare_1(r1, r2, col, type='eq'),
                    "tostr": lambda r1, r2, col: "same {{ {} ; {} ; {} }}".format(r1, r2, col),
                    "append": None}

    APIs["row_less"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                        "function": lambda r1, r2, col: row_compare_1(r1, r2, col, type='less'),
                        "tostr": lambda r1, r2, col: "row_less {{ {} ; {} ; {} }}".format(r1, r2, col),
                        "append": None}

    APIs["row_less_eq"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                           "function": lambda r1, r2, col: row_compare_1(r1, r2, col, type='less_eq'),
                           "tostr": lambda r1, r2, col: "row_less_eq {{ {} ; {} ; {} }}".format(r1, r2, col),
                           "append": None}

    APIs["row_greater"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                           "function": lambda r1, r2, col: row_compare_1(r1, r2, col, type='greater'),
                           "tostr": lambda r1, r2, col: "row_greater {{ {} ; {} ; {} }}".format(r1, r2, col),
                           "append": None}

    APIs["row_greater_eq"] = {"argument": ['row', 'row', 'header'], "output": "bool",
                              "function": lambda r1, r2, col: row_compare_1(r1, r2, col, type='greater_eq'),
                              "tostr": lambda r1, r2, col: "row_greater_eq {{ {} ; {} ; {} }}".format(r1, r2, col),
                              "append": None}

    APIs['less_eq'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                       'function': lambda t1, t2: obj_compare(t1, t2, type="less_eq"),
                       'tostr': lambda t1, t2: "less_eq {{ {} ; {} }}".format(t1, t2),
                       'append': None}

    APIs['ge'] = {"argument": ['obj', 'obj'], 'output': 'bool',
                  'function': lambda t1, t2: obj_compare(t1, t2, type="greater_eq"),
                  'tostr': lambda t1, t2: "ge {{ {} ; {} }}".format(t1, t2),
                  'append': None}

    APIs['mode'] = {"argument": ['row', 'row'], 'output': 'bool',
                    'function': lambda r1, r2: row_compare_2(r1, r2, type="mode"),
                    'tostr': lambda r1, r2: "mode {{ {} ; {} }}".format(r1, r2),
                    'append': None}

    APIs['all'] = {"argument": ['row', 'row'], 'output': 'bool',
                   'function': lambda r1, r2: row_compare_2(r1, r2, type="all"),
                   'tostr': lambda r1, r2: "all {{ {} ; {} }}".format(r1, r2),
                   'append': None}

    APIs['row_only'] = {"argument": ['row', 'row'], 'output': 'bool',
                        'function': lambda r1, r2: row_compare_2(r1, r2, type="only"),
                        'tostr': lambda r1, r2: "only {{ {} ; {} }}".format(r1, r2),
                        'append': None}

    APIs['or'] = {"argument": ['bool', 'bool'], 'output': 'bool',
                  'function': lambda t1, t2: t1 or t2,
                  'tostr': lambda t1, t2: "or {{ {} ; {} }}".format(t1, t2),
                  "append": None}

    return APIs


APIs = create_APIs()


def create_maps():
    all_funcs = APIs.keys()

    #! build map: category -> function
    cat2func = dict()
    cat2func['FILTER'] = ['filter_eq', 'filter_not_eq', 'filter_less', 'filter_greater',
                          'filter_greater_eq', 'filter_less_eq', 'filter_str_eq', 'filter_str_not_eq']
    cat2func['SUPERLATIVE'] = ['max', 'min']
    cat2func['ORDINAL'] = ['nth_max', 'nth_min']
    cat2func['SUPER_ARG'] = ['argmax', 'argmin']
    cat2func['ORD_ARG'] = ['nth_argmax', 'nth_argmin']
    cat2func['COMPARE'] = ['greater', 'less', 'eq',
                           'not_eq', 'round_eq', 'str_eq', 'not_str_eq']
    cat2func['MAJORITY'] = ['all_eq', 'all_not_eq', 'all_less', 'all_less_eq', 'all_greater', 'all_greater_eq', 
                            'most_eq', 'most_not_eq', 'most_less', 'most_less_eq', 'most_greater', 'most_greater_eq', 
                            'all_str_eq', 'all_str_not_eq', 
                            'most_str_eq', 'most_str_not_eq', ]
    cat2func['AGGREGATION'] = ['avg', 'sum']
    cat2func['Hop'] = ['num_hop', 'str_hop']

    abstract_funcs = [func for func_list in cat2func.values()
                      for func in func_list]
    unique_funcs = [func for func in all_funcs if func not in abstract_funcs]

    #! build map: function -> category
    func2cat = dict()
    for k, v in cat2func.items():
        for func in v:
            func2cat[func] = k

    for func in unique_funcs:
        func2cat[func] = func.capitalize()
        cat2func[func.capitalize()] = [func]

    string_only = dict()
    string_only['FILTER'] = ['filter_str_eq', 'filter_str_not_eq']
    string_only['COMPARE'] = ['str_eq', 'not_str_eq']
    string_only['MAJORITY'] = ['all_str_eq',
                               'all_str_not_eq', 'most_str_eq', 'most_str_not_eq']

    # print(f"{len(func2cat.keys())} functions in total.")
    # for k, v in cat2func.items():
    #     print(f"{k} has {len(v)} functions.")

    return cat2func, func2cat, string_only


cat2func, func2cat, string_only = create_maps()


# * --------------------------------------------------------------------------------------------------------------------


class Node(object):
    def __init__(self, full_table, dict_in):
        '''
        construct tree
        '''

        self.full_table = full_table
        self.func = dict_in["func"]

        # arg_type_list contains: row; num; str; obj; header; bool
        self.arg_type_list = APIs[self.func]["argument"]
        self.arg_list = []

        # child_list contains: ("text_node", a); ("func_node", b)
        self.child_list = []
        child_list = dict_in["args"]

        assert len(self.arg_type_list) == len(child_list), pdb.set_trace()

        # out_type can be bool; num; str; row; obj
        self.out_type = APIs[self.func]["output"]

        for each_child in child_list:
            if isinstance(each_child, str):  # reach leaf node
                self.child_list.append(("text_node", each_child))
            elif isinstance(each_child, dict):
                sub_func_node = Node(self.full_table, each_child)
                self.child_list.append(("func_node", sub_func_node))
            else:
                pdb.set_trace()
                raise ValueError("child type error")

        self.result = None

    def eval(self):
        for each_child, each_type in zip(self.child_list, self.arg_type_list):
            #! this child is a text node
            if each_child[0] == "text_node":
                if each_child[1] == "all_rows":
                    self.arg_list.append(self.full_table)
                else:
                    self.arg_list.append(each_child[1])

            #! this child is a function node
            else:
                sub_result = each_child[1].eval()

                # invalid
                if isinstance(sub_result, ExeError):
                    # print("sublevel error")
                    return sub_result
                elif each_type == "row":
                    if not isinstance(sub_result, pd.DataFrame):
                        # pdb.set_trace()
                        # print("error function return type")
                        return ExeError('TYPE ERROR: row should be DataFrame')
                elif each_type == "bool":
                    if not isinstance(sub_result, bool):
                        # pdb.set_trace()
                        # print("error function return type")
                        return ExeError('TYPE ERROR: bool should be bool')
                elif each_type == "str":
                    if not isinstance(sub_result, str):
                        # pdb.set_trace()
                        # print("error function return type")
                        return ExeError('TYPE ERROR: str should be str')

                self.arg_list.append(sub_result)

        # pdb.set_trace()
        result = APIs[self.func]["function"](*self.arg_list)
        return result


class ExeError(object):
    def __init__(self, message="DEFAULT ERROR: execution error"):
        self.message = message
        
    def __str__(self) -> str:
        return self.message


class Status(Enum):
    SUCCESS = 1
    FAIL = 2
    ERROR = 3


#! I add this func
def is_pure_string(val):
    return not (len(re.findall(pat_month, val)) > 0 or len(re.findall(pat_num, val)) > 0)


def modify_header(t: pd.DataFrame, col: str):
    if col not in t.columns:
        for post_fix in ['s', 'd', 'es', 'ed']:
            if (col + post_fix) in t.columns:
                return col + post_fix
        if len(col) > 0 and col[-1] == "y" and (col[:-1] + "ies") in t.columns:
            return col[:-1] + "ies"
    return col


# * --------------------------------------------------------------------------------------------------------------------


#! I add this func
def is_none(val, negate):
    if (isinstance(val, int) or isinstance(val, float)) and val == 0:
        return True if not negate else False
    elif isinstance(val, list) and len(val) == 0:
        return True if not negate else False
    elif isinstance(val, pd.DataFrame) and val.empty:
        return True if not negate else False
    elif val is None:
        return True if not negate else False
    elif isinstance(val, ExeError):
        return True if not negate else False

    return False if not negate else True


#! I add this func
def hop_obj(t: pd.DataFrame, col, value, type):
    col = modify_header(t, col)

    try:
        if type == 'contain':
            return value in t[col].values
        elif type == 'not_contain':
            return not value in t[col].values
        elif type == 'eq':
            return obj_compare(t[col].values[0], value, type="eq")
        elif type == 'not_eq':
            return obj_compare(t[col].values[0], value, type="not_eq")
        elif type == 'less':
            return obj_compare(t[col].values[0], value, type="less")
        elif type == 'less_eq':
            return obj_compare(t[col].values[0], value, type="less_eq")
        elif type == 'greater':
            return obj_compare(t[col].values[0], value, type="greater")
        elif type == 'greater_eq':
            return obj_compare(t[col].values[0], value, type="greater_eq")
    except Exception as e:
        return ExeError(f'HEADER ERROR: in hop_obj, {e}')


#! I add this func
def row_compare_1(r1: pd.DataFrame, r2: pd.DataFrame, col, type):
    if r1.empty or r2.empty:
        return ExeError('EMPTY ERROR: in row_compare_1, r1 or r2 is empty')

    try:
        col = modify_header(r1, col)
        return obj_compare(r1[col].values[0], r2[col].values[0], type)
    except Exception as e:
        return ExeError(f'GET_TABLE_VAL ERROR: in row_compare_1, r[{col}], {e}')


#! I add this func
def row_compare_2(r1: pd.DataFrame, r2: pd.DataFrame, type):
    try:
        r1 = r1.values.tolist()
        r2 = r2.values.tolist()
        if type == "all":
            """Return whether the first sub-table takes all rows of the second sub-table"""
            return not any(row for row in r2 if row not in r1)
        elif type == "only":
            """Return whether the given sub-table r2 only has one row of r1"""
            return len(r2) == 1 and r2[0] in r1
        elif type == "mode":
            """Return whether the first sub-table dominates the second sub-table with more than half of rows"""
            cnt = 0
            for row in r2:
                if row in r1:
                    cnt += 1
                if cnt >= len(r2) / 2:
                    return True
            return False
        else:
            raise ValueError()
    except Exception as e:
        return ExeError(f'GET_TABLE_VAL ERROR: in row_compare_2, {e}')


#! I add this func
def row_maxmin(r1: pd.DataFrame, r2: pd.DataFrame, col, type):
    try:
        if type == 'maximum':
            return r2[col].values[0] == r1[col].nlargest(1).values[0]
        elif type == 'minimum':
            return r2[col].values[0] == r1[col].nsmallest(1).values[0]
        else:
            raise ValueError()
    except Exception as e:
        return ExeError(f'GET_TABLE_VAL ERROR: in row_maxmin, {e}')


# for filter functions. we reset index for the result

# filter_str_eq / not_eq
def _full_string_matching(t, col, val, negate=False):
    # print(f"In fuzzy_match_filter, t is \n{t}\n, col is {col}, t[col] is \n{t[col]}\n, val is {val}.")
    # t[col] = t[col].astype('str')   #! I add this line
    col = modify_header(t, col)

    try:
        test = t[col]
    except Exception as e:
        return ExeError(f'HEADER ERROR: in _full_string_matching, {e}')

    try:
        trim_t = t[col].str.replace(" ", "")
        trim_val = val.replace(" ", "")

        if negate:
            res = t[~trim_t.str.contains(trim_val, regex=False)]
        else:
            res = t[trim_t.str.contains(trim_val, regex=False)]
        res = res.reset_index(drop=True)
        return res
    except Exception as e:
        return ExeError(f'TABLE TO STR ERROR: in _full_string_matching, {e}')


# filter ...
def fuzzy_compare_filter(t, col, val, type):
    '''fuzzy compare and filter out rows. 
    return empty pd if invalid

    t: pd.DataFrame
    col: header name
    val: compare to what 
    type: eq, not_eq, greater, greater_eq, less, less_eq
    '''
    # print(f"In fuzzy_compare_filter, col is {col}, val is {val}, type is {type}.")

    col = modify_header(t, col)

    try:
        test = t[col]
    except Exception as e:
        return ExeError(f'HEADER ERROR: in fuzzy_compare_filter, {e}')

    try:
        t[col] = t[col].astype('str')
    except Exception as e:
        return ExeError(f'TABLE TO STR ERROR: in fuzzy_compare_filter, {e}')

    if t[col].empty:
        return ExeError(f'EMPTY ERROR: in fuzzy_compare_filter, t[{col}] is empty')

    # dates
    if len(re.findall(pat_month, val)) > 0:
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        try:
            date_frame = pd.to_datetime(pd.DataFrame(
                {'year': year_list, 'month': month_num_list, 'day': day_list}))
        except Exception as e:
            return ExeError(f"DATE ERROR: {e}")

        # for val
        year_val = re.findall(pat_year, val)
        if len(year_val) == 0:
            year_val = year_list.iloc[0]
        else:
            year_val = int(year_val[0])

        day_val = re.findall(pat_day, val)
        if len(day_val) == 0:
            day_val = day_list.iloc[0]
        else:
            day_val = int(day_val[0])

        month_val = re.findall(pat_month, val)
        if len(month_val) == 0:
            month_val = month_num_list.iloc[0]
        else:
            month_val = month_map[month_val[0]]

        try:
            date_val = datetime(year_val, month_val, day_val)
        except Exception as e:
            return ExeError(f"DATE ERROR: {e}")

        try:
            if type == "greater":
                res = t[date_frame > date_val]
            elif type == "greater_eq":
                res = t[date_frame >= date_val]
            elif type == "less":
                res = t[date_frame < date_val]
            elif type == "less_eq":
                res = t[date_frame <= date_val]
            elif type == "eq":
                res = t[date_frame == date_val]
            elif type == "not_eq":
                res = t[date_frame != date_val]  # ! delete ~
        except Exception as e:
            return ExeError(f"DATE ERROR: {e}")

        res = res.reset_index(drop=True)
        return res

    # numbers, or mixed numbers and strings
    val_pat = re.findall(pat_num, val)
    if len(val_pat) == 0:
        #! fall back to full string matching
        if type == "eq":
            return _full_string_matching(t, col, val, negate=False)
        elif type == "not_eq":
            return _full_string_matching(t, col, val, negate=True)
        else:
            return ExeError(f'COMPARE ERROR: in fuzzy_compare_filter, {val} cannot be compared')

    num = val_pat[0].replace(",", "")
    num = num.replace(":", "")
    num = num.replace(" ", "")
    try:
        num = float(num)
    except:
        num = num.replace(".", "")
        num = float(num)

    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return pd.DataFrame(columns=list(t.columns))
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    if type == "greater":
        res = t[np.greater(nums, num)]
    elif type == "greater_eq":
        res = t[np.greater_equal(nums, num)]
    elif type == "less":
        res = t[np.less(nums, num)]
    elif type == "less_eq":
        res = t[np.less_equal(nums, num)]
    elif type == "eq":
        res = t[np.isclose(nums, num)]
    elif type == "not_eq":
        res = t[~np.isclose(nums, num)]

    res = res.reset_index(drop=True)
    return res


# for comparison
def obj_compare(item1, item2, type, round=False):
    if item1 is None or item2 is None:
        return ExeError('NONE ERROR: in obj_compare, obj is None')

    tolerance = 0.15 if round else 1e-9
    # both numeric
    try:
        num_1 = float(item1)
        num_2 = float(item2)

        if type == "eq":
            return math.isclose(num_1, num_2, rel_tol=tolerance)
        elif type == "not_eq":
            return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        elif type == "greater":
            return num_1 > num_2
        elif type == "greater_eq":
            return num_1 >= num_2
        elif type == "less":
            return num_1 < num_2
        elif type == "less_eq":
            return num_1 <= num_2
        elif type == "diff":
            return num_1 - num_2
    except ValueError:
        # strings
        # mixed numbers and strings
        item1 = str(item1)
        item2 = str(item2)

        # dates
        # item1
        if len(re.findall(pat_month, item1)) > 0:
            year_val1 = re.findall(pat_year, item1)
            if len(year_val1) == 0:
                year_val1 = int("2260")
            else:
                year_val1 = int(year_val1[0])

            day_val1 = re.findall(pat_day, item1)
            if len(day_val1) == 0:
                day_val1 = int("1")
            else:
                day_val1 = int(day_val1[0])

            month_val1 = re.findall(pat_month, item1)
            if len(month_val1) == 0:
                month_val1 = int("1")
            else:
                month_val1 = month_map[month_val1[0]]

            try:
                date_val1 = datetime(year_val1, month_val1, day_val1)
            except Exception as e:
                return ExeError(f'DATE ERROR: in obj_compare, {e}')

            # item2
            year_val2 = re.findall(pat_year, item2)
            if len(year_val2) == 0:
                year_val2 = int("2260")
            else:
                year_val2 = int(year_val2[0])

            day_val2 = re.findall(pat_day, item2)
            if len(day_val2) == 0:
                day_val2 = int("1")
            else:
                day_val2 = int(day_val2[0])

            month_val2 = re.findall(pat_month, item2)
            if len(month_val2) == 0:
                month_val2 = int("1")
            else:
                month_val2 = month_map[month_val2[0]]

            try:
                date_val2 = datetime(year_val2, month_val2, day_val2)
            except Exception as e:
                return ExeError(f'DATE ERROR: in obj_compare, {e}')

            # if negate:
            #   return date_val1 != date_val2
            # else:
            #   return date_val1 == date_val2

            if type == "eq":
                return date_val1 == date_val2
            elif type == "not_eq":
                return date_val1 != date_val2
            elif type == "greater":
                return date_val1 > date_val2
            elif type == "greater_eq":
                return date_val1 >= date_val2
            elif type == "less":
                return date_val1 < date_val2
            elif type == "less_eq":
                return date_val1 <= date_val2
            # for diff return string
            elif type == "diff":
                return str((date_val1 - date_val2).days) + " days"

        # mixed string and numerical
        val_pat1 = re.findall(pat_num, item1)
        val_pat2 = re.findall(pat_num, item2)
        if len(val_pat1) == 0 or len(val_pat2) == 0:
            # fall back to full string matching
            if type == "not_eq":
                return (item1 not in item2) and (item2 not in item1)
            elif type == "eq":
                return item1 in item2 or item2 in item1
            else:
                return ExeError('VALUE ERROR: in obj_compare, type')

        num_1 = val_pat1[0].replace(",", "")
        num_1 = num_1.replace(":", "")
        num_1 = num_1.replace(" ", "")
        try:
            num_1 = float(num_1)
        except:
            num_1 = num_1.replace(".", "")
            num_1 = float(num_1)

        num_2 = val_pat2[0].replace(",", "")
        num_2 = num_2.replace(":", "")
        num_2 = num_2.replace(" ", "")
        try:
            num_2 = float(num_2)
        except:
            num_2 = num_2.replace(".", "")
            num_2 = float(num_2)

        # if negate:
        #   return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        # return math.isclose(num_1, num_2, rel_tol=tolerance)

        if type == "eq":
            return math.isclose(num_1, num_2, rel_tol=tolerance)
        elif type == "not_eq":
            return (not math.isclose(num_1, num_2, rel_tol=tolerance))
        elif type == "greater":
            return num_1 > num_2
        elif type == "greater_eq":
            return num_1 >= num_2
        elif type == "less":
            return num_1 < num_2
        elif type == "less_eq":
            return num_1 <= num_2
        elif type == "diff":
            return num_1 - num_2


# for aggregation: sum avg

def agg(t, col, type):
    '''
    sum or avg for aggregation
    '''
    if col not in t.columns:
        for post_fix in ['s', 'es', 'd', 'ed']:
            if (col + post_fix) in t.columns:
                col = col + post_fix
                break

    try:
        test = t[col]
    except Exception as e:
        return ExeError(f'HEADER ERROR: in agg, {e}')

    # unused
    if t.dtypes[col] == np.int64 or t.dtypes[col] == np.float64:
        if type == "sum":
            res = t[col].sum()
        elif type == "avg":
            res = t[col].mean()

        return res
    else:
        pats = t[col].str.extract(pat_add, expand=False)
        if pats.isnull().all():
            pats = t[col].str.extract(pat_num, expand=False)
        if pats.isnull().all():
            return 0.0
        pats.fillna("0.0")
        nums = pats.str.replace(",", "")
        nums = nums.str.replace(":", "")
        nums = nums.str.replace(" ", "")
        try:
            nums = nums.astype("float")
        except:
            nums = nums.str.replace(".", "")
            nums = nums.astype("float")

        if type == "sum":
            return nums.sum()
        elif type == "avg":
            return nums.mean()
        else:
            raise ValueError()


# for hop

def hop_op(t, col):
    if len(t) == 0:
        return ExeError('EMPTY ERROR: in hop_op, t is empty')
    col = modify_header(t, col)
    return t[col].values[0]


# for superlative, ordinal

def nth_maxmin(t, col, order=1, max_or_min="max", arg=False):
    '''
    for max, min, argmax, argmin, 
    nth_max, nth_min, nth_argmax, nth_argmin

    return string or rows
    '''
    col = modify_header(t, col)

    try:
        test = t[col]
    except Exception as e:
        return ExeError(f'HEADER ERROR: in nth_maxmin, {e}')

    order = int(order)
    # return the original content for max,min
    # dates
    try:
        date_pats = t[col].str.extract(pat_month, expand=False)
    except:
        t[col] = t[col].astype('str')  # ! I add this line
        date_pats = t[col].str.extract(pat_month, expand=False)

    if not date_pats.isnull().all():
        year_list = t[col].str.extract(pat_year, expand=False)
        day_list = t[col].str.extract(pat_day, expand=False)
        month_list = t[col].str.extract(pat_month, expand=False)
        month_num_list = month_list.map(month_map)

        # pandas at most 2262
        year_list = year_list.fillna("2260").astype("int")
        day_list = day_list.fillna("1").astype("int")
        month_num_list = month_num_list.fillna("1").astype("int")

        try:
            date_series = pd.to_datetime(pd.DataFrame(
                {'year': year_list, 'month': month_num_list, 'day': day_list}))

            if max_or_min == "max":
                tar_row = date_series.nlargest(order).iloc[[-1]]
            elif max_or_min == "min":
                tar_row = date_series.nsmallest(order).iloc[[-1]]
            ind = list(tar_row.index.values)
            if arg:
                res = t.iloc[ind]
            else:
                res = t.iloc[ind][col].values[0]

            return res
        except:
            pass

    # mixed string and numerical
    pats = t[col].str.extract(pat_add, expand=False)
    if pats.isnull().all():
        pats = t[col].str.extract(pat_num, expand=False)
    if pats.isnull().all():
        return ExeError('MIX STR&NUM ERROR: in nth_maxmin, pats is null for all')
    nums = pats.str.replace(",", "")
    nums = nums.str.replace(":", "")
    nums = nums.str.replace(" ", "")
    try:
        nums = nums.astype("float")
    except:
        nums = nums.str.replace(".", "")
        nums = nums.astype("float")

    try:
        if max_or_min == "max":
            tar_row = nums.nlargest(order).iloc[[-1]]
        elif max_or_min == "min":
            tar_row = nums.nsmallest(order).iloc[[-1]]
        ind = list(tar_row.index.values)
        if arg:
            res = t.iloc[ind]
            res = res.reset_index(drop=True)
        else:
            res = t.iloc[ind][col].values[0]
            res = res.reset_index(drop=True)
    except Exception as e:
        return ExeError(f'MAXMIN ERROR: in nth_maxmin, {e}')

    return res


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


# * --------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    print("END")
