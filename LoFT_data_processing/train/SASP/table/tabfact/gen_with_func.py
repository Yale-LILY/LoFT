import supar
import json
import argparse
import nltk
from preprocess_example import string_in_table, stop_words, date_match, isnum

from nltk.corpus import wordnet

import sys, os
sys.path.append('../../../..')
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../../../..'))
parser = argparse.ArgumentParser(description='read params')
parser.add_argument('--processed_example_path', default = os.path.join(paths.train_output_root, "data_shard"), help='tagged question example path')
parser.add_argument('--data_dict_file', default = os.path.join(paths.train_output_root, "tables.jsonl"),help='tab file path')
parser.add_argument('--train_shard', default=90, help='whether to expand entities')
parser.add_argument('--trigger_word_dict', default = os.path.join(paths.logicnlg_root, "trigger_word_all.json"),help='trigger word for function')
args = parser.parse_args()
trigger_word_dict = json.load(open(args.trigger_word_dict,'r'))
dep_parser = supar.Parser.load('crf2o-dep-en')


# combine the entity in the same span, for example: 'sept 09, 2012' may be in string entity and date entity, 
# just add them into the same tk by @@@:   'sept 09, 2012@@@2012-09-09'
def combine_entity(examples, tabs):
    combined_ents_l = []

    for e in examples:
        tab = tabs[e['context']]

        tk_idx = 0
        prev_ent = None
        combined_ents = []
        ents = sorted(e['entities'], key=lambda it: (it['token_start'], -it['token_end']))
        for i, ent in enumerate(ents):
            if tk_idx <= ent['token_start']:
                tk_idx = ent['token_end']
                combined_ents.append({"token_end": ent['token_end'], "token_start": ent['token_start'], \
                    "type": ent['type'], "value": [str(ent['value'][0])]})
                prev_ent = combined_ents[-1]
                if prev_ent['type'] == 'string_list' and \
                    not string_in_table(prev_ent['value'][0], tab, use_tokens_contain=True):
                    prev_ent['value'][0] = '' # prev value only in caption, del it
            elif tk_idx > ent['token_start']:
                if not(tk_idx >= ent['token_end']):
                    print('!!! cross entity:', e, prev_ent, ent)
                    if prev_ent['value'][0] == '' or not \
                        string_in_table(ent['value'][0], tab, use_tokens_contain=True):
                        continue
                    if prev_ent['type'] == 'string_list':
                        prev_ent['value'][0] = prev_ent['value'][0] + '@@@' + str(ent['value'][0])
                    else:
                        prev_ent['value'][0] = str(ent['value'][0]) + '@@@' + prev_ent['value'][0]
                        prev_ent['token_start'] = ent['token_start']
                else:
                    if prev_ent['value'][0] == '': # this prev_ent is caption, can not use it
                        continue
                    prev_ent['value'][0] = prev_ent['value'][0] + '@@@' + str(ent['value'][0])
                    if (prev_ent['token_start'], prev_ent['token_end']) == (ent['token_start'], ent['token_end']) and \
                        ent['type'] != 'string_list':
                        prev_ent['value'][0] = str(ent['value'][0])
                        prev_ent['type'] = ent['type']
        combined_ents_l.append(combined_ents)

    return combined_ents_l


def string_in_props(string, tab):
    props = []
    for prop in tab['props']:
        prop_tks = [s for s in '-'.join(prop[2:].split('-')[:-1]).split('_') if s not in stop_words]
        # prop_tks = [s for s in '-'.join(prop[2:].split('-')[:-1]).split('_') if s not in stop_words]
        # if tokens_contain(prop_string, string):
        #     props.append(prop)
        if set(string.split(' ')).issubset(set(prop_tks)) and string in ' '.join(prop_tks):
            props.append(prop)
    return props


def dateprops_in_table(date, kg, strict_constrain=False):
    dateprops_in_t = []
    props = set(kg['datetime_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop in props and date_match(val[0], date, strict_constrain=False):
                dateprops_in_t.append(prop)
    return set(dateprops_in_t)


def numprops_in_table(num, kg):
    numprops_in_t = []
    props = set(kg['num_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop in props and val[0] == num:
                numprops_in_t.append(prop)
    return set(numprops_in_t)


def strprops_in_table(string, kg):
    strprops_in_t = []
    other_props = set(kg['num_props']).union(set(kg['datetime_props']))
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop not in other_props and string in val[0]:
                strprops_in_t.append(prop)
    res = set(strprops_in_t)
    for prop in strprops_in_t:
        prop_string = '-'.join(prop[2:].split('-')[:-1])
        for postfix in ['-number', '-num2', '-date']:
            if 'r.' + prop_string + postfix in other_props:
                res.add('r.' + prop_string + postfix)
    return res


def get_all_ent_props(ent, tab):
    string = ent['value'][0].split('@@@')[0]
    if ent['type'] == 'datetime_list':
        return dateprops_in_table(string, tab)
    elif ent['type'] == 'num_list':
        num = float(string)
        return numprops_in_table(num, tab)
    elif ent['type'] == 'string_list':
        return strprops_in_table(string, tab)


num_except_one = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', \
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']


def extract_props_funcs(examples, tabs, combined_ents_l):
    tmp_conj = ['', ' ', '-', ' - ', '.', '_', ' a ', ' the ', "'s ", " 's ", ' of ', ' an ', ' / ', " ' ", "' "]
    props_l = []
    trigger_words_l = []
    for e, combined_ents in zip(examples, combined_ents_l):
        tab = tabs[e['context']]
        ents = combined_ents
        tks = e['tokens']
        is_ent = [0 for i in range(0, len(tks))]
        for ent in ents:
            # if ent['value'][0] == '': continue
            for i in range(ent['token_start'], min(len(tks), ent['token_end'])):
                is_ent[i] = 1
        props = []
        for i, tk in enumerate(tks):
            if is_ent[i] == 1 or (props and i < props[-1]['token_end']):
                continue
            if tk in stop_words:
                continue
            if not string_in_props(tk, tab):
                continue
            prop = {"token_end": i+1, \
                    "token_start": i, "prop": [tk]}
            props.append(prop)
            while props[-1]['token_end'] < len(tks) \
                and is_ent[props[-1]['token_end']] != 1:
                end_tk = tks[props[-1]['token_end']]
                new_prop_strs = [s.join([props[-1]['prop'][0], end_tk]) for s in tmp_conj]
                if props[-1]['token_end'] < len(tks)-1 and end_tk in stop_words and is_ent[props[-1]['token_end']+1] != 1:
                    after_end = tks[props[-1]['token_end']+1]
                    new_prop_strs += [s.join([props[-1]['prop'][0], after_end]) for s in tmp_conj]
                flg = 0
                for str_i, new_prop_str in enumerate(new_prop_strs):
                    if string_in_props(new_prop_str, tab):
                        props[-1]['prop'] = [new_prop_str]
                        props[-1]['token_end'] += 1 if str_i < len(tmp_conj) else 2
                        flg = 1
                        break
                if flg == 0: break
        for prop in props:
            prop['prop'][0] = '@@@'.join(string_in_props(prop['prop'][0], tab))
            assert prop['prop'][0] != ''
        props_l.append(props)

        # add func trigger word into tree
        for prop in props:
            # if ent['value'][0] == '': continue
            for i in range(prop['token_start'], min(len(tks), prop['token_end'])):
                is_ent[i] = 1
        trigger_word_values = set()
        for k, v in trigger_word_dict.items():
            for w in v:
                trigger_word_values.add(w)
        trigger_word_values = list(trigger_word_values)
        trigger_words = []
        for i, tk in enumerate(tks):
            if is_ent[i] == 1:
                continue
            if ' '.join(tks[i-1:i+1]) in trigger_word_values and is_ent[i-1] == 0:
                trigger_words.append({"token_end": i+1, "token_start": i-1, "trigger": [' '.join(tks[i-1:i+1])]})
            elif tk in trigger_word_values:
                if tk == 'only' and i < len(tks)-1:
                    if tks[i+1] in num_except_one or (isnum(tks[i+1]) and float(tks[i+1])!=1):
                        continue
                    elif {"token_end": i+2, "token_start": i+1, "type": "num_list", "value": [1.0]} in e['entities']:
                        e['entities'].remove({"token_end": i+2, "token_start": i+1, "type": "num_list", "value": [1.0]})
                        try:
                            combined_ents.remove({"token_end": i+2, "token_start": i+1, "type": "num_list", "value": ['1.0']})
                        except:
                            print('***combined_ents***')
                    elif {"token_end": i+2, "token_start": i+1, "type": "string_list", "value": ['one']} in e['entities']:
                        e['entities'].remove({"token_end": i+2, "token_start": i+1, "type": "string_list", "value": ['one']})
                        try:
                            combined_ents.remove({"token_end": i+2, "token_start": i+1, "type": "string_list", "value": ['one']})
                        except:
                            print('***combined_ents***')
                trigger_words.append({"token_end": i+1, "token_start": i, "trigger": [tk]})
        trigger_words_l.append(trigger_words)
        if e['context'] == "nt-30318": print(combined_ents)

    for e, combined_ents in zip(examples, combined_ents_l):
        tab = tabs[e['context']]
        for ent in combined_ents:
            if ent['value'][0] == '':
                e['entities'] = [ori_ent for ori_ent in e['entities'] if \
                    max(ori_ent['token_start'], ent['token_start']) >= min(ori_ent['token_end'], ent['token_end'])]
                continue
            all_props = get_all_ent_props(ent, tab)
            if not all_props: continue
            ent['prop'] = ['@@@'.join(all_props) ]
        # !!! important !!! here we can also delete duplicate ent in same cell as another ent

    combined_ents_l = [[ent for ent in combined_ents if ent['value'][0] != ''] for combined_ents in combined_ents_l]
    return combined_ents_l, props_l, trigger_words_l


def get_val(ent):
    if 'value' in ent or 'prop' in ent:
        return [ent, {}] if ent['token_end'] - ent['token_start'] == 2 else [ent]
    elif 'trigger' in ent:
        if ent['token_end'] - ent['token_start'] == 1:
            return [ent]
        elif ent['token_end'] - ent['token_start'] == 2:
            return [{}, ent]
    print('error, not value, not prop and not trigger')
    exit()


def shrink_sents(examples, combined_ents_l, props_l, trigger_words_l):
    shrinked_sents = []
    ent_prop_in_tks_l = []
    for e, combined_ents, props, trigger_words in zip(examples, combined_ents_l, props_l, trigger_words_l):
        ori_tks = e['tokens']
        new_tks = []
        ent_prop_in_tks = []
        ents = sorted(combined_ents + props + trigger_words, key=lambda it: it['token_start'])
        for i in range(1, len(ents)):
            if ents[i-1]['token_end'] > ents[i]['token_start']:
                print('something wrong', ents[i-1], ents[i], e)
                exit()
        tk_idx = 0
        for ent in ents:
            while tk_idx < ent['token_start']:
                new_tks.append(ori_tks[tk_idx])
                ent_prop_in_tks.append({})
                tk_idx += 1
            new_tks.extend(ent['trigger'][0].split() if 'trigger' in ent else \
                ['_'.join([tk for tk in ori_tks[ent['token_start']:ent['token_end']][:3] if tk not in stop_words])] \
                if ent['token_end'] - ent['token_start'] > 2 else ori_tks[ent['token_start']:ent['token_end']]
            )
            ent_prop_in_tks.extend(get_val(ent))
            while tk_idx < ent['token_end']:
                tk_idx += 1
        shrinked_sents.append(ori_tks if new_tks == [] else new_tks)
        ent_prop_in_tks_l.append([{} for tk_ in ori_tks] if new_tks == [] else ent_prop_in_tks)
        if len(shrinked_sents[-1]) != len(ent_prop_in_tks_l[-1]):
            print('\n', '###', shrinked_sents[-1], ent_prop_in_tks_l[-1], e['tokens'])
            l = min(len(shrinked_sents[-1]), len(ent_prop_in_tks_l[-1]))
            shrinked_sents[-1] = shrinked_sents[-1][:l]
            ent_prop_in_tks_l[-1] = ent_prop_in_tks_l[-1][:l]
        # assert len(shrinked_sents[-1]) == len(ent_prop_in_tks_l[-1])
    return shrinked_sents, ent_prop_in_tks_l


def generate_dep_arcs(shrinked_sents):
    shrinked_sents = [['-LRB-' if tk == '(' else '-RRB-' if tk == ')' else tk for tk in s] \
        for s in shrinked_sents]
    data = dep_parser.predict(shrinked_sents, verbose=False)
    return data.arcs


filter_trigger_word = set().union(trigger_word_dict['filter_le'], trigger_word_dict['filter_ge'], trigger_word_dict['filter_less'], trigger_word_dict['filter_greater'])

def isdate(string):
    try:
        if len(string) != 10: return False
        if string[:4] != 'xxxx':
            int(string[:4])
        if string[5:7] != 'xx':
            int(string[5:7])
        if string[8:10] != 'xx':
            int(string[8:10])
        return True
    except:
        return False

def merge(shrinked_c_, shrinked_tree_, care_child = True):
    shrinked_c = shrinked_c_['value']
    root_node = shrinked_tree_['value']
    if ('trigger' in shrinked_c and 'value' in shrinked_c and 'prop' in shrinked_c) or \
        ('trigger' in root_node and 'value' in root_node and 'prop' in root_node):
        return False
    if care_child and len(shrinked_c_['child']) >= 2:
        return False
    if 'trigger' in shrinked_c and 'trigger' in root_node:
        return False
    if 'value' in shrinked_c and 'value' in root_node:
        return False
        # if both have value and prop, and 1) ent1 in prop2 or ent2 in prop1; 2) prop1==prop2 and ent2+ent1 in table
        # then combine two nodes
    final_prop = ''
    prop_inter = set()
    trigger_str = shrinked_c.get('trigger',[''])[0] or root_node.get('trigger',[''])[0]
    value_str = shrinked_c.get('value',[''])[0] or root_node.get('value',[''])[0]
    if 'prop' in shrinked_c and 'prop' in root_node:
        prop_inter = set(shrinked_c['prop'][0].split('@@@')).intersection(root_node['prop'][0].split('@@@'))
        prop_union = set(shrinked_c['prop'][0].split('@@@')).union(root_node['prop'][0].split('@@@'))
        if prop_inter:
            if len(prop_union) <= 3:
                final_prop = '@@@'.join(prop_union)
            else:
                final_prop = '@@@'.join(prop_inter)
        else:
            if not trigger_str and not value_str:
                final_prop = '@@@'.join(prop_union)
            else:
                return False
    else:
        final_prop = shrinked_c.get('prop',[''])[0] or root_node.get('prop',[''])[0]
    
    if trigger_str in filter_trigger_word and value_str:
        tmp_vals = value_str.split('@@@')
        if not any([isnum(tmp_val) or isdate(tmp_val) for tmp_val in tmp_vals]):
            return False

    if 'token_start' in shrinked_c and 'token_start' in root_node:
        a = max(shrinked_c['token_start'], root_node['token_start'])
        b = min(shrinked_c['token_end'], root_node['token_end'])
        if abs(a-b) > 2 and not prop_inter:
            return False

    tmp_dict = {}
    if trigger_str:
        tmp_dict['trigger'] = [trigger_str]
    if value_str:
        tmp_dict['value'] = [value_str]
    if final_prop:
        tmp_dict['prop'] = [final_prop]

    token_start = min(shrinked_c.get('token_start', 1000), root_node.get('token_start', 1000))
    token_end = max(shrinked_c.get('token_end', -1), root_node.get('token_end', -1))
    if token_start != 1000 and token_end != -1:
        tmp_dict['token_start'] = token_start
        tmp_dict['token_end'] = token_end

    shrinked_tree_['value'] = tmp_dict
    shrinked_tree_['child'].extend(shrinked_c_['child'])
    return True


def shrink_dep_tree(root_node):
    shrinked_tree = {'child': [], 'value': root_node['value']}
    for c in root_node['child']:
        shrinked_c = shrink_dep_tree(c)
        if not shrinked_c['value'] and shrinked_c['child'] == []:
            continue
        if merge(shrinked_c, shrinked_tree):
            continue
        shrinked_tree['child'].append(shrinked_c)

    if len(shrinked_tree['child']) == 1:
        shrinked_c = shrinked_tree['child'][0]
        shrinked_tree['child'] = []
        if not merge(shrinked_c, shrinked_tree, care_child = False):
            shrinked_tree['child'] = [shrinked_c]

    return shrinked_tree


def get_explore_seq(arcs, root_idx):
    # explore_seq = [root_idx]
    explore_seq = []
    flg = 0
    for arc_idx, arc in enumerate(arcs):
        if arc-1 == root_idx:
            if arcs.count(arc_idx + 1) == 0: continue # without leaf nodes
            explore_seq += get_explore_seq(arcs, arc_idx)
            flg = 1
    if flg == 0: 
        return [root_idx]
    return explore_seq + [root_idx]


def get_node_span(shrinked_arcs, shrinked_spans, explore_seq):
    if explore_seq == [0] and shrinked_spans[0] == '':
        return []
    node_spans = []
    for i, node in enumerate(explore_seq):
        if explore_seq.count(node) == 2 and i == explore_seq.index(node):
            new_span = shrinked_spans[node]
        else:
            nodes_in_span = [i_ for i_, arc in enumerate(shrinked_arcs) if arc-1 == node]
            nodes_in_span += [node] if shrinked_spans[node] != '' else []
            new_span = (min([shrinked_spans[node_in_span][0] for node_in_span in nodes_in_span]), \
                max([shrinked_spans[node_in_span][1] for node_in_span in nodes_in_span]))
            shrinked_spans[node] = new_span
        assert new_span != ''
        node_spans.append(new_span)
    return node_spans


def shrink_dep_trees(examples, arcs, ent_prop_in_tks_l):
    for e, arc, ent_prop_in_tks in zip(examples, arcs, ent_prop_in_tks_l):
        if ent_prop_in_tks_l == []:
            print('error: maybe sentence length is 0')
            e['dependency'] = {
                'arcs': [0],
                'tk_list': [''],
                'explore_seq': [0],
                'node_span':[(0, len(e['tokens'])-1)]
            }
            continue
        nodes = [{'value': v, 'child': []} for v in ent_prop_in_tks]
        for i, a in enumerate(arc):
            if a == 0: continue
            nodes[a-1]['child'].append(nodes[i])
        if len(arc) != len(ent_prop_in_tks):
            print('error, arc len not equal to entprop len')
        shrinked_tree = shrink_dep_tree(nodes[arc.index(0)])

        flat_tree = [(shrinked_tree['child'], shrinked_tree['value'], 0)]
        idx = 0
        while idx < len(flat_tree):
            for c in flat_tree[idx][0]:
                flat_tree.append((c['child'], c['value'], idx+1))
            idx += 1

        for node in flat_tree:
            # try:
            trigger = node[1].get('trigger',[''])[0]
            # except:
            #     print(flat_tree)
            #     print(ent_prop_in_tks)
            #     print(node[1])
            #     print([noded[1] for noded in flat_tree])
            func_list = []
            for k, v in trigger_word_dict.items():
                if trigger in v:
                    func_list.append(k)
            if 'token_start' in node[1]:
                node[1]['func'] = ['@@@'.join(func_list)]
                if 'filter_g' in node[1]['func'][0] or 'filter_l' in node[1]['func'][0]:
                    if not node[1].get('value',[''])[0] or not node[1].get('prop',[''])[0]:
                        if func_list[0] == 'filter_ge' or func_list[0] == 'filter_le':
                            node[1]['func'] = ['@@@'.join(func_list[1:])]
                        else:
                            node[1]['func'] = ['@@@'.join(func_list[1:]) + '@@@diff']
                    #if node[1]['token_start'] > 0 and isnum(e['tokens'][node[1]['token_start']-1]):
                    #    node[1]['func'] = ['diff']
                    #    print('@'*100, '\n', e['tokens'], '\n')
                    #if node[1]['token_start'] > 1 and isnum(e['tokens'][node[1]['token_start']-2]):
                    #    node[1]['func'] = ['diff']
                    #    print('@'*100, '\n', e['tokens'], '\n')

        shrinked_arcs, shrinked_values, shrinked_spans = \
            [node[2] for node in flat_tree], \
            [';'.join([node[1].get('func',[''])[0], node[1].get('value',[''])[0], node[1].get('prop',[''])[0]]) if node[1] else '' for node in flat_tree], \
            [(node[1]['token_start'], node[1]['token_end']) if node[1] else '' for node in flat_tree]

        explore_seq = get_explore_seq(shrinked_arcs, shrinked_arcs.index(0))
        # explore_seq = [node for idx, node in enumerate(explore_seq) \
        #     if shrinked_values[node] != '' or explore_seq.count(node) == 1 or idx != explore_seq.index(node)]
        node_spans = get_node_span(shrinked_arcs, shrinked_spans, explore_seq)
        e['dependency'] = {
            'arcs': shrinked_arcs,
            'tk_list': shrinked_values,
            'explore_seq': explore_seq if node_spans != [] else [0],
            'node_span': node_spans if node_spans != [] else [(0, len(e['tokens'])-1)]
        }


if __name__ == '__main__':
    args = parser.parse_args()

    tabfile = open(args.data_dict_file, 'r')
    tabs = dict()
    for item in tabfile.readlines():
        obj = json.loads(item[:-1])
        tabs[obj['name']] = obj

    create_path(args.processed_example_path)
    files = []
    for i in range(int(args.train_shard)):
        files.append(\
            open(args.processed_example_path + '/train_shard_{}-{}.jsonl'.format(args.train_shard, i), 'r'))
        # files.append(\
        #     open(args.processed_example_path + '/test.jsonl'.format(args.train_shard, i), 'r'))
    files_dep = []
    create_path(args.processed_example_path + '_with_dep/')
    for i in range(int(args.train_shard)):
        files_dep.append(\
            open(args.processed_example_path + '_with_dep/train_shard_{}-{}.jsonl'.format(args.train_shard, i), 'w'))
        # files_dep.append(\
        #     open(args.processed_example_path + '_with_dep/test.jsonl'.format(args.train_shard, i), 'w'))

    for i, f in enumerate(files):
        all_line = f.readlines()
        examples = []
        for l in all_line:
            example = json.loads(l)
            examples.append(example)
        print(i, 'th file, total examples', len(examples))
        print('entities per sentence:', sum([len(e['entities']) for e in examples])*1.0/len(examples))
        combined_ents_l = combine_entity(examples, tabs)
        print('combined entities per sentence:', sum([len(combined_ents) for combined_ents in combined_ents_l])*1.0/len(examples))
        #       ↓
        # print(examples)
        combined_ents_l, props_l, trigger_words_l = extract_props_funcs(examples, tabs, combined_ents_l)
        print('props per sentence:', sum([len(props) for props in props_l])*1.0/len(examples))
        #       ↓
        shrinked_sents, ent_prop_in_tks_l = shrink_sents(examples, combined_ents_l, props_l, trigger_words_l)
        #       ↓
        # examples = [e for e, s in zip(examples, shrinked_sents) if s]
        # ent_prop_in_tks_l = [e for e, s in zip(ent_prop_in_tks_l, shrinked_sents) if s]
        # shrinked_sents = [s for s in shrinked_sents if s]
        arcs = generate_dep_arcs(shrinked_sents)
        #       ↓
        shrink_dep_trees(examples, arcs, ent_prop_in_tks_l)
        print('arcs per sentence:', sum([len(e['dependency']['arcs']) for e in examples])*1.0/len(examples))
        print('explore_seq per sentence:', sum([len(e['dependency']['explore_seq']) for e in examples])*1.0/len(examples))
        print('seq len more than 2:', sum([len(e['dependency']['explore_seq']) > 2 for e in examples]))
        print('left examples', len(examples), '\n')

        for e in examples:
            json.dump(e, files_dep[i])
            files_dep[i].write('\n')

    for f, f1 in zip(files, files_dep):
        f.close()
        f1.close()
