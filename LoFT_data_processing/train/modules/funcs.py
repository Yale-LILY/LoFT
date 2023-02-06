all_funcs = dict()

all_funcs['count'] = {'operands': ['row'], 'output': 'num'}  #* same as Logic2Text

all_funcs['is_none'] = {'operands': ['obj'], 'output': 'bool'}    #! I modified this; untested
all_funcs['is_not_none'] = {'operands': ['obj'], 'output': 'bool'}    #! I added this; untested
all_funcs['is_not'] = {'operands': ['bool'], 'output': 'bool'}    #? untested

all_funcs['average'] = {'operands': ['row', 'header'], 'output': 'num'}  #! same as Logic2Text, but should change func name
all_funcs['sum'] = {'operands': ['row', 'header'], 'output': 'num'}  #* same as Logic2Text
all_funcs['maximum'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}   #? untested
all_funcs['minimum'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}   #? untested

all_funcs['argmax'] = {'operands': ['row', 'header'], 'output': 'row'}  #* same as Logic2Text
all_funcs['argmin'] = {'operands': ['row', 'header'], 'output': 'row'}  #* same as Logic2Text

all_funcs['hop'] = {'operands': ['row', 'header'], 'output': 'obj'}  #! same as Logic2Text, but should change func name  

all_funcs['hop_str_contain_not_any'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}   #? untested
all_funcs['hop_str_contain_any'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}   #? untested

all_funcs['hop_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}    #? untested
all_funcs['hop_not_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}    #? untested
all_funcs['hop_less'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}  #? untested
all_funcs['hop_less_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}   #? untested
all_funcs['hop_greater'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}   #? untested
all_funcs['hop_greater_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'bool'}    #? untested

all_funcs['filter_str_contain_not_any'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}    #! equals "filter_str_not_eq"
all_funcs['filter_str_contain_any'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}    #! equals "filter_str_eq"

all_funcs['filter_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}   #* same as Logic2Text
all_funcs['filter_not_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}   #* same as Logic2Text
all_funcs['filter_less'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}     #* same as Logic2Text
all_funcs['filter_less_eq'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}    #* same as Logic2Text
all_funcs['filter_greater'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'}      #* same as Logic2Text
all_funcs['filter_ge'] = {'operands': ['row', 'header', 'obj'], 'output': 'row'} #! same as Logic2Text, but should change func name  

all_funcs['diff'] = {'operands': ['row', 'row', 'header'], 'output': 'num'}  #! untested, I rename it as "row_diff"

all_funcs['same'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}  #? untested
all_funcs['row_less'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}  #? untested
all_funcs['row_less_eq'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}   #? untested
all_funcs['row_greater'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}   #? untested
all_funcs['row_greater_eq'] = {'operands': ['row', 'row', 'header'], 'output': 'bool'}    #? untested

all_funcs['eq'] = {'operands': ['obj', 'obj'], 'output': 'bool'}  #* same as Logic2Text
all_funcs['less'] = {'operands': ['obj', 'obj'], 'output': 'bool'}    #* same as Logic2Text
all_funcs['less_eq'] = {'operands': ['obj', 'obj'], 'output': 'bool'} #? untested
all_funcs['greater'] = {'operands': ['obj', 'obj'], 'output': 'bool'} #* same as Logic2Text
all_funcs['ge'] = {'operands': ['obj', 'obj'], 'output': 'bool'}  #? untested

all_funcs['mode'] = {'operands': ['row', 'row'], 'output': 'bool'}    #? untested

all_funcs['all'] = {'operands': ['row', 'row'], 'output': 'bool'} #? untested

all_funcs['only'] = {'operands': ['row', 'row'], 'output': 'bool'}    #! untested; rename as "row_only"

all_funcs['and'] = {'operands': ['bool', 'bool'], 'output': 'bool'}   #* same as Logic2Text
all_funcs['or'] = {'operands': ['bool', 'bool'], 'output': 'bool'}    #? untested

#! ============================================================================================================================== 

argtype2func = dict()

argtype2func['ROW'] = ['count']
argtype2func['OBJ'] = ['is_none', 'is_not_none']
argtype2func['BOOL'] = ['is_not']
argtype2func['ROW_HEAD'] = ['average', 'sum', 'argmax', 'argmin', 'hop']
argtype2func['ROW_HEAD_OBJ'] = ['hop_str_contain_not_any', 'hop_str_contain_any',
                               'hop_eq', 'hop_not_eq', 'hop_less', 'hop_less_eq', 'hop_greater', 'hop_greater_eq',
                               'filter_str_contain_not_any', 'filter_str_contain_any',
                               'filter_eq', 'filter_not_eq', 'filter_less', 'filter_less_eq', 'filter_greater', 'filter_ge']
argtype2func['ROW_ROW_HEAD'] = ['maximum', 'minimum', 'diff', 'same', 'row_less', 'row_less_eq', 'row_greater', 'row_greater_eq']
argtype2func['OBJ_OBJ'] = ['eq', 'less', 'less_eq', 'greater', 'ge']
argtype2func['ROW_ROW'] = ['mode', 'all', 'only']
argtype2func['BOOL_BOOL'] = ['and', 'or']

func2argtype = dict()

for argtype in argtype2func:
    for func in argtype2func[argtype]:
        func2argtype[func] = argtype

assert len(func2argtype.keys()) == len(all_funcs.keys())

#! ============================================================================================================================== 

changeFuncName = dict()

changeFuncName['average'] = 'avg'
changeFuncName['hop'] = 'str_hop'
changeFuncName['filter_str_contain_not_any'] = 'filter_str_not_eq'
changeFuncName['filter_str_contain_any'] = 'filter_str_eq'
changeFuncName['filter_ge'] = 'filter_greater_eq'
changeFuncName['diff'] = 'row_diff'
changeFuncName['only'] = 'row_only'
