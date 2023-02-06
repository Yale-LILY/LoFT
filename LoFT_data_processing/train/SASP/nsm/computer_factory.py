
"""Computers can read in tokens, parse them into a program, and execute it."""

from __future__ import print_function

import json
from collections import OrderedDict, namedtuple
import re
import copy
import sys
import os
import nsm.data_utils as data_utils
import pprint
import numpy as np

END_TK = data_utils.END_TK  # End of program token
ERROR_TK = '<ERROR>'
# SPECIAL_TKS = [END_TK, ERROR_TK, '(', ')']
SPECIAL_TKS = [ERROR_TK, '(', ')']
smoothing = 1.e-8

class LispInterpreter(object):
    """Interpreter reads in tokens, parse them into a program and execute it."""

    def __init__(self, type_hierarchy, max_mem, max_n_exp, arcs, tk_list, assisted=True):
        """
        max_mem: maximum number of memory slots.
        max_n_exp: maximum number of expressions.
        assisted: whether to provide assistance to the programmer (used for neural programmer).
        """
        # Create namespace.
        self.namespace = Namespace()

        self.assisted = assisted
        # Configs.
        # Functions used to call
        # Signature: autocomplete(evaled_exp, valid_tokens, evaled_tokens)
        # return a subset of valid_tokens that passed the
        # filter. Used to implement filtering with denotation.
        self.type_hierarchy = type_hierarchy
        self.type_ancestry = create_type_ancestry(type_hierarchy)
        self.max_mem = max_mem
        self.max_n_exp = max_n_exp

        # Initialize the parser state.
        self.n_exp = 0
        self.history = []
        self.exp_stack = []
        self.done = False
        self.result = None

        # +++push arc_list, mem_list, cur_pos
        self.arcs = arcs # example: [2,0,2]
        self.tk_list = tk_list # example: [Bob@@@r-name like banana@@@r-name]
        self.mem_list = [[] for i in range(len(arcs))] # example: [[], [], []]
        self.cur_pos = 0

    @property
    def primitive_names(self):
        primitive_names = []
        for k, v in self.namespace.iteritems():
            if ('property' in self.type_ancestry[v['type']] or
                    'primitive_function' in self.type_ancestry[v['type']]):
                primitive_names.append(k)
        return primitive_names

    @property
    def primitives(self):
        primitives = []
        for k, v in self.namespace.iteritems():
            if ('property' in self.type_ancestry[v['type']] or
                    'primitive_function' in self.type_ancestry[v['type']]):
                primitives.append(v)
        return primitives

    def add_constant(self, value, type, name=None):
        """Generate the code and variables to hold the constants."""
        if name is None:
            name = self.namespace.generate_new_name()
            if 'list' in type:
                for i, tk in enumerate(self.tk_list):
                    if tk and str(value[0]) in tk.split(';')[1].split('@@@'):
                        self.mem_list[i].append(name)
            elif 'property' in type:
                for i, tk in enumerate(self.tk_list):
                    if tk and value in tk.split(';')[2].split('@@@'):
                        self.mem_list[i].append(name)
        self.namespace[name] = dict(
            value=value, type=type,
            is_constant=True, name=name)
        if name == 'all_rows':
            # self.mem_list[0].append(name)
            self.namespace[name]['prop'] = []
            self.namespace[name]['used'] = 0
        return name

    def add_function(self, name, value, args, return_type,
                     autocomplete, type):
        """Add function into the namespace."""
        if name in self.namespace:
            raise ValueError('Name %s is already used.' % name)
        else:
            for i, tk in enumerate(self.tk_list):
                if tk and name in tk.split(';')[0].split('@@@'):
                    self.mem_list[i].append(name)
            self.namespace[name] = dict(
                value=value, type=type,
                autocomplete=autocomplete,
                return_type=return_type, args=args)

    def autocomplete(self, exp, tokens, token_vals, namespace):
        # func = util[0]
        # util = [x['value'] for x in util]
        # token_vals = [x['value'] for x in token_vals]
        # if func['type'] == 'global_primitive_function':
        #   return func['autocomplete'](
        #     util, tokens, token_vals, namespace=namespace)
        # else:
        #   return func['autocomplete'](util, tokens, token_vals)
        function = exp[0]

        return function['autocomplete'](exp, tokens, token_vals)

    def reset(self, only_reset_variables=False):
        """Reset all the interpreter state."""
        if only_reset_variables:
            self.namespace.reset_variables()
        else:
            self.namespace = Namespace()
        self.history = []
        self.n_exp = 0
        self.exp_stack = []
        self.done = False
        self.result = None

        self.arcs = []
        self.tk_list = []
        self.mem_list = []
        self.cur_pos = 0

    def read_token_id(self, token_id):
        token = self.rev_vocab[token_id]
        return self.read_token(token)

    def get_family(self):
        return list(range(len(self.arcs)-1, -1, -1))

    def get_mem_by_ids(self, idxs):
        mem_can_use = []
        for i in idxs:
            mem_can_use.extend(self.mem_list[i])
        return set(mem_can_use)

    def valid_tk_curnode(self, exp = [], result = []):
        idxs = self.get_family()
        if not exp:
            # get func head
            return self.get_mem_by_ids(idxs)
        # find ents/props
        idxs0 = idxs
        if exp[0] in ['filter_eq', 'filter_str_contain_any']:
            idxs = [i for i in idxs if self.mem_list[i] and not(\
                set(self.mem_list[i]).intersection(['filter_le', 'filter_ge', 'filter_less', 'filter_greater'])
                )]
        else:
            idxs = [i for i in idxs if self.mem_list[i] and exp[0] in self.mem_list[i]]
        if not idxs and len(exp) > 1:
            idxs = idxs0
        # find props
        if len(exp) >= 2:
            idxs0 = idxs
            idxs = [i for i in idxs if self.mem_list[i] and exp[1] in self.mem_list[i]]
            if not idxs:
                idxs = idxs0
        # find rows
        if len(exp) == 3:
            idxs0 = idxs
            idxs = [i for i in idxs if self.mem_list[i] and exp[2] in self.mem_list[i]]
            if not idxs:
                idxs = idxs0
        return idxs

    def dist_to_leaf(self, node):
        # for leaves:
        if (node + 1) not in self.arcs: return 0
        # for other nodes
        dist_set = []
        for _i in range(len(self.arcs)):
            if (_i + 1) in self.arcs: continue # do not count nodes that are not leaves
            idx_tmp, dist = _i, 0
            while idx_tmp != 0 and idx_tmp != node:
                dist += 1.
                idx_tmp = self.arcs[idx_tmp] - 1
            if idx_tmp == node:
                dist_set.append(dist)
        return sum(dist_set)/len(dist_set)

    def tk_score_head(self, result, decay):
        res_score = []
        for res in result:
            if res in ['filter_eq', 'filter_str_contain_any', 'count', 'eq']:
                res_score.append(decay)
                continue
            # elif res in ['filter_str_contain_not_any']:
            #     res_score.append(-1*decay)
            #     continue
            # else:
            #     res_score.append(0)
            #     continue
            s_max, s_tmp = 0., 0.
            for i in range(len(self.mem_list)-1, -1, -1):
                if res in self.mem_list[i]:
                    s_tmp = pow(decay, self.dist_to_leaf(i))
                    if s_tmp > s_max:
                        s_max = s_tmp
                        if s_tmp == 1.: break
            res_score.append(s_max)
        return [score * 2. for score in res_score]

    def node_dist(self, node1, node2):
        idx_tmp = node1
        trace1 = []
        while self.arcs[idx_tmp] != 0:
            trace1.append(idx_tmp)
            idx_tmp = self.arcs[idx_tmp] - 1

        idx_tmp = node2
        trace2 = []
        while self.arcs[idx_tmp] != 0:
            trace2.append(idx_tmp)
            idx_tmp = self.arcs[idx_tmp] - 1

        n_ancestor = len(set(trace1).intersection(trace2))
        return len(trace1) - n_ancestor + len(trace2) - n_ancestor

    def tk_score_curnode(self, explore_index, result, decay):
        res_score = []
        for res in result:
            if res == 'all_rows':
                res_score.append(decay)
                continue
            s_max, s_tmp = 0., 0.
            for i in range(len(self.mem_list)-1, -1, -1):
                if res in self.mem_list[i]:
                    s_tmp = pow(decay, self.node_dist(explore_index, i))
                    if s_tmp > s_max:
                        s_max = s_tmp
                        if s_tmp == 1.: break
            res_score.append(s_max)
        return [score * 2. for score in res_score]
        # return res_score

    def delete_info(self, anchor, info, info_type):
        if anchor == -1: return
        info_dist, info_idx = 10, -1
        for i in self.get_family():
            if info in self.mem_list[i]:
                tmp = self.node_dist(i, anchor)
                if tmp < info_dist:
                    info_dist, info_idx = tmp, i
                    if tmp == 0.: break
        if info_idx == -1: return
        if info_type == 'func':
            self.mem_list[info_idx] = [m for m in self.mem_list[info_idx] if \
                self.namespace.get_object(m)['type'] not in ['global_primitive_function', 'primitive_function', 'function']]
        elif info_type == 'ent':
            self.mem_list[info_idx] = [m for m in self.mem_list[info_idx] if \
                not ('list' in self.namespace.get_object(m)['type'] and self.namespace.get_object(m)['is_constant'])]
        else:
            self.mem_list[info_idx] = [m for m in self.mem_list[info_idx] if m != info]

    def update_mem(self, new_exp, new_name, result):
        avg_min_dist, min_dist_idx = 10, -1
        mem_can_use = self.valid_tk_curnode(exp=new_exp, result=result)
        if mem_can_use:
            for i in self.get_family():
                tmp = np.mean([self.node_dist(i, _i) for _i in mem_can_use])
                if tmp < avg_min_dist:
                    avg_min_dist, min_dist_idx = tmp, i
                    if tmp == 0.: break
        if min_dist_idx == -1:
            self.mem_list[0].append(new_name)
        else:
            self.mem_list[min_dist_idx].append(new_name)

        if 'filter' in new_exp[0]:
            # update used for row
            if new_exp[3] != 'all_rows':
                self.namespace.get_object(new_exp[3])['used'] = 1
            # update props and used for new_row
            ori_props = self.namespace.get_object(new_exp[3])['prop']
            result['prop'] = ori_props + [self.namespace.get_object(new_exp[2])['value']]
            if new_exp[3] == 'all_rows':
                result['used'] = None # only filtered once, so unfinished
            else:
                result['used'] = 0
            # update func/ents
            self.delete_info(min_dist_idx, new_exp[0], 'func')
            self.delete_info(min_dist_idx, new_exp[1], 'ent')
        elif new_exp[0] in ['sum', 'average', 'argmin', 'argmax']:
            # update used for row
            if new_exp[1] != 'all_rows':
                self.namespace.get_object(new_exp[1])['used'] = 1
            # update used for new_row
            if new_exp[0] in ['argmin', 'argmax']:
                result['prop'] = self.namespace.get_object(new_exp[1])['prop']
                result['used'] = None # only filtered once, so unfinished
            else:
                result['used'] = 0
            # update func
            self.delete_info(min_dist_idx, new_exp[0], 'func')
        elif new_exp[0] in ['diff', 'row_le', 'row_less', 'row_ge', 'row_greater', 'same']:
            # update used for row, this row be used, so not be None any more (finished)
            self.namespace.get_object(new_exp[1])['used'] = 0 # new_exp[1] [2]never be all_rows, because autocomp...
            self.namespace.get_object(new_exp[2])['used'] = 0
            # update used for new res
            result['used'] = 0
            # update func
            self.delete_info(min_dist_idx, new_exp[0], 'func')
        elif new_exp[0] in ['le', 'less', 'ge', 'greater', 'eq']:
            # update used for bool
            result['used'] = 0
            # update the func/ents
            flg1 = flg2 = flg3 = 1
            self.delete_info(min_dist_idx, new_exp[0], 'func')
            if self.namespace.get_object(new_exp[1])['is_constant']:
                self.delete_info(min_dist_idx, new_exp[1], 'ent')
            else:
                self.delete_info(min_dist_idx, new_exp[1], 'other')
            if self.namespace.get_object(new_exp[2])['is_constant']:
                self.delete_info(min_dist_idx, new_exp[2], 'ent')
            else:
                self.delete_info(min_dist_idx, new_exp[2], 'other')
        elif new_exp[0] in ['mode', 'all', 'only', 'maximum', 'minimum']:
            # update used for row, this row be used, so not be None any more (finished)
            self.namespace.get_object(new_exp[2])['used'] = 0
            # update used for bool
            result['used'] = 0
            # update func
            self.delete_info(min_dist_idx, new_exp[0], 'func')
        elif new_exp[0] in ['is_none', 'is_not', 'count']:
            # update used for row
            if new_exp[1] != 'all_rows':
                self.namespace.get_object(new_exp[1])['used'] = 1
            # update used for new_row/bool
            result['used'] = 0
            if new_exp[0] == 'count':
                result['count'] = 1
            # update func
            self.delete_info(min_dist_idx, new_exp[0], 'func')

    def exist_program(self, new_exp):
        if ' '.join(self.history).count(' '.join(new_exp)) > 1:
            return True
        return False

    def read_token(self, token):
        """Read in one token, parse and execute the expression if completed."""
        if ((self.n_exp >= self.max_n_exp) or
            (self.namespace.n_var >= self.max_mem)):
            self.done = True
        new_exp = self.parse_step(token)
        # If reads in end of program, then return the last value as result.
        if self.done:
            # self.result = self.namespace.get_last_value()
            self.result = []
            for k, v in self.namespace.items():
                if v['type'] == 'bool':
                    self.result.extend(v['value'])
                elif v['type'] == 'entity_list' and v['used'] == 0 and v['name'] != 'all_rows':
                    self.result.append(len(v['value']) != 0)
            return self.result
        elif new_exp:
            if self.assisted:
                name = self.namespace.generate_new_name()
                result = self.eval(['define', name, new_exp])
                result['name'] = name
                self.n_exp += 1
                # If there are errors in the execution, self.eval
                # will return None. We can also give a separate negative
                # reward for errors.
                if result is None or self.exist_program(new_exp):
                    self.namespace.n_var -= 1
                    self.done = True
                    self.result = [ERROR_TK]
                else:
                    self.update_mem(new_exp, name, result)
                #   result = self.eval(['define', name, ERROR_TK])
            else:
                result = self.eval(new_exp)
            return result
        else:
            return None

    # algorithm (2)
    def valid_tokens(self): # here only return token instead of object, token~=obj['name']
        """Return valid tokens for the next step for programmer to pick."""
        # If already exceeded max memory or max expression
        # limit, then must end the program.
        if ((self.n_exp >= self.max_n_exp) or
                (self.namespace.n_var >= self.max_mem)):
            return ([END_TK], [0])
        # If not in an expression, either start a new expression or end the program.
        elif not self.exp_stack:
            #info_can_use = [m for m in self.valid_tk_curnode() \
            #    if self.namespace.get_object(m)['type'] in \
            #        ['num_list', 'string_list', 'datetime_list', \
            #        'global_primitive_function', 'primitive_function', 'function']]
            #if info_can_use:
            #    return ['('], [0]
            #else:
            #    return [END_TK], [0]
            return (['(', END_TK], [0, 0])
        # If currently in an expression.
        else:
            exp = self.exp_stack[-1]
            # If in the middle of a new expression.
            if exp: # exp_stack.append([]) every time, so if exp==[], then should get a func
                # Use number of arguments to check if all arguments are there.
                head = exp[0]
                args = self.namespace[head]['args']
                pos = len(exp) - 1
                if pos == len(args):
                    return ([')'], [0])
                # namespace.valid_tokens: find in [func_name; v_i; special_token;] with args[pos] type
                result = self.namespace.valid_tokens(args[pos], self.get_type_ancestors)
                if self.autocomplete is not None:
                    valid_tokens = result
                    evaled_exp = [self.eval(item) for item in exp] # obj named 'item' in exp
                    evaled_tokens = [self.eval(tk) for tk in valid_tokens] # obj named 'tk' in valid_tokens
                    result = self.autocomplete( # check wheather the token can be select(in case of leading to error)
                        evaled_exp, valid_tokens, evaled_tokens, self.namespace)
                tmp_res = []
                for res in result:
                    if res not in tmp_res:
                        tmp_res.append(res)
                result = tmp_res
                mem_can_use = self.valid_tk_curnode(exp=exp, result=result)
                if mem_can_use:
                    result_score = np.mean(np.array([self.tk_score_curnode(_i, result, .7) for _i in mem_can_use]), axis=0)
                else:
                    result_score = [0] * len(result)
                # result_score = [0] * len(result)
            # If at the beginning of a new expression.
            else:
                result = self.namespace.valid_tokens(
                    {'types': ['head']}, self.get_type_ancestors) # result: funcs
                result_score = self.tk_score_head(result, .7)
        return result, result_score

    def parse_step(self, token):
        """Run the parser for one step with given token which parses tokens into expressions."""
        self.history.append(token)
        if token == END_TK:
            self.done = True
        elif token == '(':
            self.exp_stack.append([])
        elif token == ')':
            # One list is finished.
            new_exp = self.exp_stack.pop()
            if self.exp_stack:
                self.exp_stack[-1].append(new_exp)
            else:
                self.exp_stack = []
                return new_exp
        elif self.exp_stack:
            self.exp_stack[-1].append(token)
        else:
            # Atom expression.
            return token

    def tokenize(self, chars):
        """Convert a string of characters into a list of tokens."""
        return chars.replace('(', ' ( ').replace(')', ' ) ').split()

    def get_type_ancestors(self, type):
        return self.type_ancestry[type]

    def infer_type(self, return_type, arg_types):
        """Infer the type of the returned value of a function."""
        if hasattr(return_type, '__call__'):
            return return_type(*arg_types)
        else:
            return return_type

    def eval(self, x, namespace=None):
        """Another layer above _eval to handle exceptions."""
        try:
            result = self._eval(x, namespace)
        except Exception as e:
            print('Error: ', e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print('when evaluating ', x)
            print(self.history)
            pprint.pprint(self.namespace)
            raise e
            result = None
        return result

    def _eval(self, x, namespace=None):
        """Evaluate an expression in an namespace."""
        if namespace is None:
            namespace = self.namespace
        if is_symbol(x):  # variable reference
            return namespace.get_object(x).copy()
        elif x[0] == 'define':  # (define name exp)
            (_, name, exp) = x
            obj = self._eval(exp, namespace)
            namespace[name] = obj
            return obj
        else:
            # Execute a function.
            proc = self._eval(x[0], namespace)
            args = [self._eval(exp, namespace) for exp in x[1:]]
            arg_values = [arg['value'] for arg in args]
            if proc['type'] == 'global_primitive_function':
                arg_values += [self.namespace]
            value = proc['value'](*(arg_values))
            arg_types = [arg['type'] for arg in args]
            type = self.infer_type(proc['return_type'], arg_types)
            return {'value': value, 'type': type, 'is_constant': False}

    def step(self, token):
        """Open AI gym inferface."""
        result = self.read_token(token)
        observation = token
        reward = 0.0
        done = self.done
        if (result is None) or self.done:
            write_pos = None
        else:
            write_pos = self.namespace.n_var - 1

        info = {'result': result,
                'write_pos': write_pos}
        return observation, reward, done, info

    def get_last_var_loc(self):
        return self.namespace.n_var - 1

    def interactive(self, prompt='> ', assisted=False):
        """A prompt-read-eval-print loop."""
        print('Namespace:')
        for key, val in self.namespace.items():
            if val['type'] == 'primitive_function':
                print('Function: {}'.format(key))
            else:
                print('Entity: {}, Value: {}'.format(key, val))

        temp = self.assisted
        # try:
        self.assisted = assisted
        while True:
            # try:
            query = input(prompt).strip()
            tokens = self.tokenize(query)
            for tk in tokens:
                result = self.read_token(tk)
                print('Read in [{}], valid tokens: {}'.format(tk, self.valid_tokens()))
                if result:
                    print('Result:')
                    print(json.dumps(result, indent=2))
            # except Exception as e:
            #   print(e)
            #   continue
        # finally:
        #   self.assisted = temp

    def has_extra_work(self):
        """Check if the current solution contains some extra/wasted work."""
        all_var_names = ['v{}'.format(i)
                         for i in range(self.namespace.n_var)]
        for var_name in all_var_names:
            obj = self.namespace.get_object(var_name)
            # If some variable is not given as constant, not used
            # in other expressions and not a bool one, then
            # generating it is some extra work that should not be
            # done.
            if ((not obj['is_constant']) and obj['type'] != 'bool' 
                    and obj['type'] != 'entity_list'
                    and (var_name not in self.history)):
                return True
            # filteredFirstTime or argmed rows 's used will be None, filteredSecondTime be 0
            if obj['type'] == 'entity_list' and obj['used'] is None:
                return True
        return False

    # eq(3) and eq(4) in our paper
    def ent_usage_score(self):
        """use tk_list to score current logic form"""
        logic_form = [
            tk if tk == 'all_rows' else 
            self.namespace[tk]['value'] if 'prop' in self.namespace[tk]['type'] else \
            str(self.namespace[tk]['value'][0]) if self.namespace[tk].get('is_constant', False) else
            tk for tk in self.history if tk in self.namespace]

        trigger_use_cnt, ent_use_cnt, prop_use_cnt = 0, 0, 0
        trigger_all_cnt, ent_all_cnt, prop_all_cnt = 0, 0, 0
        for tks in self.tk_list:
            if not tks: continue
            tks = tks.split(';')
            triggers, ents, props = tks[0], tks[1], tks[2]
            if triggers:
                trigger_all_cnt += 1
                for trigger in triggers.split('@@@'):
                    if trigger in logic_form:
                       trigger_use_cnt += 1
                       break
            if ents:
                ent_all_cnt += 1
                for ent in ents.split('@@@'):
                    if ent in logic_form:
                       ent_use_cnt += 1
                       break
            if props:
                prop_all_cnt += 1
                for prop in props.split('@@@'):
                    if prop in logic_form:
                       prop_use_cnt += 1
                       break
        s1 = (trigger_use_cnt)/(trigger_all_cnt+smoothing)
        s2 = (ent_use_cnt)/(ent_all_cnt+smoothing)
        s3 = (prop_use_cnt)/(prop_all_cnt+smoothing)
        if logic_form.count('count') >= 2: return 0.2*s1 + 0.2*s2 + 0.1*s3 # this may do little good
        return 0.4*s1 + 0.4*s2 + 0.2*s3

    def clone(self):
        """Make a copy of itself, used in search."""
        new = LispInterpreter(
            self.type_hierarchy, self.max_mem, self.max_n_exp, self.arcs[:], self.tk_list[:], self.assisted)

        new.history = self.history[:]
        new.exp_stack = copy.deepcopy(self.exp_stack)
        new.n_exp = self.n_exp
        new.namespace = self.namespace.clone()

        new.mem_list = [self.mem_list[i][:] for i in range(len(self.arcs))]
        new.cur_pos = self.cur_pos

        return new

    def get_vocab(self):
        mem_tokens = []
        for i in range(self.max_mem):
            mem_tokens.append('v{}'.format(i))
        vocab = data_utils.Vocab(
            list(self.namespace.get_all_names()) + SPECIAL_TKS + mem_tokens)
        return vocab


class Namespace(OrderedDict):
    """Namespace is a mapping from names to values.

  Namespace maintains the mapping from names to their
  values. It also generates new variable names for memory
  slots (v0, v1...), and support finding a subset of
  variables that fulfill some type constraints, (for
  example, find all the functions or find all the entity
  lists).
  """

    def __init__(self, *args, **kwargs):
        """Initialize the namespace with a list of functions."""
        # params = dict(zip(names, objs))
        super(Namespace, self).__init__(*args, **kwargs)
        self.n_var = 0
        self.last_var = None

    def clone(self):
        new = Namespace(self) # same as self.copy
        new.n_var = self.n_var
        new.last_var = self.last_var
        for k, v in new.items():
            if 'is_constant' in v and v['is_constant'] == False:
                new[k] = v.copy()
                if 'prop' in v:
                    new[k]['prop'] = v['prop'].copy()
        return new

    def clone_and_reset(self):
        copy = self.clone()
        copy.reset_variables()

        return copy

    def generate_new_name(self):
        """Create and return a new variable."""
        name = 'v{}'.format(self.n_var)
        self.last_var = name
        self.n_var += 1
        return name

    def valid_tokens(self, constraint, get_type_ancestors):
        """Return all the names/tokens that fulfill the constraint."""
        return [k for k, v in self.items()
                if self._is_token_valid(v, constraint, get_type_ancestors)]

    def _is_token_valid(self, token, constraint, get_type_ancestors):
        """Determine if the token fulfills the given constraint."""
        type = token['type']
        return set(get_type_ancestors(type) + [type]).intersection(constraint['types'])

    def get_value(self, name):
        return self[name]['value']

    def get_object(self, name):
        return self[name]

    def get_last_value(self):
        if self.last_var is None:
            return None
        else:
            return self.get_value(self.last_var)

    def get_all_names(self):
        return self.keys()

    def reset_variables(self):
        keys = list(self.keys())
        for k in keys:
            if re.match(r'v\d+', k):
                del self[k]
        self.n_var = 0
        self.last_var = None


def is_symbol(x):
    return isinstance(x, str)


def create_type_ancestry(type_tree):
    type_ancestry = {}
    for type, _ in type_tree.items():
        _get_type_ancestors(type, type_tree, type_ancestry)
    return type_ancestry


def _get_type_ancestors(type, type_hrchy, type_ancestry):
    """Compute the ancestors of a type with memorization."""
    if type in type_ancestry:
        return type_ancestry[type]
    else:
        parents = type_hrchy[type]
        result = parents[:]
        for p in parents:
            ancestors = _get_type_ancestors(p, type_hrchy, type_ancestry)
            for a in ancestors:
                if a not in result:
                    result.append(a)
        type_ancestry[type] = result
        return result
