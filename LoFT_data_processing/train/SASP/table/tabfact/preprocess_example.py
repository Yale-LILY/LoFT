import os
import random
import time
import argparse
import json
from tqdm import tqdm
import stanfordcorenlp
import re
import unicodedata
import multiprocessing
import csv
import nltk
random.seed(1)

import sys
sys.path.append('../../../..')
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../../../..'))

parser = argparse.ArgumentParser(description='read params')
parser.add_argument('--csv_dir', default = os.path.join(paths.logicnlg_root, "all_csv"), help='all csv folder')
parser.add_argument('--used_csv_ids', default = os.path.join(paths.train_output_root, "train_csv_ids.txt"), help='csv ids to be processed')
parser.add_argument('--data_dict_file', default = os.path.join(paths.train_output_root, "tables.jsonl"), help='csv data saved as dict in the file')
parser.add_argument('--min_frac_for_props', help='min frac for date/num props')
parser.add_argument('--corenlp_path', help='corenlp latest path')

parser.add_argument('--example_path_untagged', default = os.path.join(paths.train_output_root, "train_lm_tokenized.json"), help='untagged question example path')
parser.add_argument('--example_path', default = os.path.join(paths.train_output_root, "train_lm_tokenized.tagged"), help='tagged question example path')
parser.add_argument('--processed_example_path', default = os.path.join(paths.train_output_root, "data_shard"), help='tagged question example path')
parser.add_argument('--num_workers', default=1, help='multi process workers number')
parser.add_argument('--expand_entities', default=True, help='whether to expand entities')
parser.add_argument('--process_conjunction', default=True, help='whether to expand entities with or')
parser.add_argument('--combine_date', default=True, help='whether to combine date')
parser.add_argument('--train_shard', default=90, help='whether to expand entities')
# parser.add_argument('--min_frac_for_match', default=0.3, help='entities match cell prop')

def normalize_str(x):
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.U).lower().strip()
    return x


# output date with format xxxx-xx-xx
months_a = ['january', 'february', 'march', 'april', 'may', 'june',
            'july', 'august', 'september', 'october', 'november', 'december']
months_b = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
months_c = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
months_d = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
days_a = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', \
    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
days_b = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', \
    '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
# !!!!!!! this version only process one date, can not process duration
def normalize_date(cell):
    year, month, day = '', '', ''
    words = re.findall(r"[\w']+",cell.lower())

    for word in words:
        if not month and word in months_a:
            month = months_c[months_a.index(word)]
        elif not month and word in months_b:
            month = months_c[months_b.index(word)]
        elif not word.isdigit(): continue
        elif not year and int(word) > 999 and int(word) < 2020:
            year = word
        elif not month and word in months_c:
            month = word
        elif not month and word in months_d:
            month = months_c[int(word)-1]
        elif not day and word in days_a:
            day = word
        elif not day and word in days_b:
            day = days_a[int(word)-1]
    res = ''
    res += year if year else 'XXXX'
    res += '-'
    res += month if month else 'XX'
    res += '-'
    res += day if day else 'XX'
    return res


# !!!!!!! this version used by process sentence
def normalize_date_word(word):
    word = word.replace('th', '')
    year, month, day = '', '', ''

    if not month and word in months_a:
        month = months_c[months_a.index(word)]
    elif not month and word in months_b:
        month = months_c[months_b.index(word)]
    elif not word.isdigit(): return None
    elif not year and int(word) > 999 and int(word) < 2020:
        year = word
    elif not day and word in days_a:
        day = word
    elif not day and word in days_b:
        day = days_a[int(word)-1]
    res = ''
    res += year if year else 'XXXX'
    res += '-'
    res += month if month else 'XX'
    res += '-'
    res += day if day else 'XX'

    if res != 'XXXX-XX-XX':
        return res
    else:
        return None


number_a = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', \
    'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']
number_b = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', \
    'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth']
 
# output num1 num2(if not none)
def normalize_num(cell):
    nums = re.findall(r"\d+\.?\d*|\d*\.?\d+|twentieth|nineteenth|eighteenth|seventeenth|sixteenth|fifteenth|fourteenth|thirteenth|"
        +r"twelfth|eleventh|tenth|ninth|eighth|seventh|sixth|fifth|fourth|third|second|first|twenty|nineteen|eighteen|seventeen|sixteen|"
        +r"fifteen|fourteen|thirteen|twelve|eleven|ten|nine|eight|seven|six|five|four|three|two|one", cell.lower())
    try:
        if len(nums) > 0:
            if nums[0] in number_a: num1 = [number_a.index(nums[0])+1]
            elif nums[0] in number_b: num1 = [number_b.index(nums[0])+1]
            else: num1 = [float(nums[0])]
        else:
            num1 = None

        if len(nums) > 1:
            if nums[1] in number_a: num2 = [number_a.index(nums[1])+1]
            elif nums[1] in number_b: num2 = [number_b.index(nums[1])+1]
            else: num2 = [float(nums[1])]
        else:
            num2 = None
    except:
        print(cell)
        print(nums)
        print('numbers error: {}! cell_have_num = sum([\'NUMBER\' error!'.format(nums)); exit()

    return num1, num2


wnl = nltk.stem.WordNetLemmatizer()
def lemmatize_cell(cell):
    tks = nltk.word_tokenize(cell)
    tags = nltk.pos_tag(tks)
    new_cell = []
    for tk, tag in zip(tks, tags):
        if tag[1][0] == 'J':
            new_cell.append(wnl.lemmatize(tk, pos = nltk.corpus.wordnet.ADJ))
        elif tag[1][0] == 'V':
            new_cell.append(wnl.lemmatize(tk, pos = nltk.corpus.wordnet.VERB))
        elif tag[1][0] == 'N':
            new_cell.append(wnl.lemmatize(tk, pos = nltk.corpus.wordnet.NOUN))
        elif tag[1][0] == 'R':
            new_cell.append(wnl.lemmatize(tk, pos = nltk.corpus.wordnet.ADV))
        else:
            new_cell.append(wnl.lemmatize(tk, pos = nltk.corpus.wordnet.NOUN))
    return ' '.join(new_cell)


def preprocess_tab(args, nlp):
    # ================== tab & example props used in pytorch nsm==================
    # examples: entities;features;prop_features;tokens;answer;pos_tags;question;context;in_table;processed_tokens;
    # tab: kg; name, num_props, row_ents, datetime_props, props
    # ============================================================================
    
    with open(args.used_csv_ids, 'r') as f1, open(args.data_dict_file, 'w') as f2:
        
        allcsv = [line[:-1] for line in f1.readlines()] # [:-1] ddrop \n after the path
        
        for csv in tqdm(allcsv):
            
            tab_dict = dict()
            kg = dict()
            props = set()
            num_props = set()
            datetime_props = set()

            tab_dict['name'] = csv.split('.')[0]
            
            with open(args.csv_dir+'/'+csv) as f3:
                
                l = [line[:-1] for line in f3.readlines()]
                
                header = ['_'.join((normalize_str(c)).split(' ')) for c in l[0].split('#')]
                body = [r.split('#') for r in l[1:]]
                
                tab_dict['row_ents'] = ['row_{}'.format(i) for i in range(len(body))]

                for r_index, row in enumerate(body): 
                    # actually, cell type can be str/date/num/duration/ordinal/time and so on, !!!here only process str/date/num 
                    # exist method can not deal with duration, it is a big problem
                    row_node = dict()
                    
                    for c_index, cell in enumerate(row):

                        if cell == '': continue

                        cell = lemmatize_cell(cell)

                        prop_name = 'r.{}-string'.format(header[c_index])
                        row_node[prop_name] = [normalize_str(cell)]
                        props.add(prop_name)

                        try:
                            # cell = re.sub(r'(?P<value>[\d]+%)', lambda x: '0.'+x.group('value')[:-1], cell); nlp.ner(cell)
                            ner_result = nlp.ner(cell.replace('%', ' '))
                        except:
                            print(cell); nlp.close(); exit()

                        cell_have_date = (sum(['DATE' in item for item in ner_result])) \
                            * 1.0 / len(ner_result) > 2.0 * min_frac_for_props
                        if cell_have_date:
                            # date_str = ' '.join([item[0] for item in ner_result if 'DATE' == item[1]])
                            prop_name = 'r.{}-date'.format(header[c_index])
                            row_node[prop_name] = [normalize_date(cell)]
                            datetime_props.add(prop_name)

                        num1 = None; num2 = None
                        cell_have_num = (sum([('NUMBER' in item or 'DURATION' in item or 'DATE' in item or 'ORDINAL' in item) \
                            for item in ner_result])) * 1.0 / len(ner_result) > min_frac_for_props
                        if cell_have_num:
                            num1, num2 = normalize_num(cell)

                        if num1 is not None:
                            # num_str = ' '.join([item[0] for item in ner_result if 'NUMBER' == item[1]])
                            prop_name = 'r.{}-number'.format(header[c_index])
                            row_node[prop_name] = num1
                            num_props.add(prop_name)

                        if num2 is not None:
                            prop_name = 'r.{}-num2'.format(header[c_index])
                            row_node[prop_name] = num2
                            num_props.add(prop_name)
                    
                    kg['row_{}'.format(r_index)] = row_node
                
                nodes = kg.values()
                total_n = len(nodes)

                num_props = list(num_props); datetime_props = list(datetime_props)
                prop_to_del = []
                for prop in (num_props + datetime_props):
                    nodes_with_prop = [node for node in nodes if prop in node]
                    if len(nodes_with_prop) * 1.0 / total_n < min_frac_for_props:
                        prop_to_del.append(prop)
                        for node in nodes_with_prop:
                            del node[prop]
                num_props = [prop for prop in num_props if prop not in prop_to_del]
                datetime_props = [prop for prop in datetime_props if prop not in prop_to_del]

                tab_dict['kg'] = kg
                tab_dict['props'] = list(props) + num_props + datetime_props
                tab_dict['num_props'] = num_props
                tab_dict['datetime_props'] = datetime_props
            
            f2.write(json.dumps(tab_dict)+'\n')


def change_raw_to_tagged(args, nlp):

    # here example_path is get from tabfact dataset
    with open(args.example_path_untagged, 'r') as f1, open(args.used_csv_ids, 'r') as f2:
        
        allcsv = [line[:-1] for line in f2.readlines()]

        examples = []
        data = json.load(f1)
        for key in data:
            if key in allcsv: # !!!!! remember that there is title for every tab, here i don't add it !!!!!!
                examples.extend([[item, key, data[key][1][index]] for (index, item) in enumerate(data[key][0])])
        print(len(examples))
        
        f = open(args.example_path, 'w')
        f.write('id	utterance	context	targetValue	tokens	lemmaTokens	posTags	nerTags	nerValues\n')
        for i, item in enumerate(tqdm(examples)):

            try:
                item[0] = item[0].replace('%', '') # !!!!!!!! replace because of corenlp bug, so in my setting rate 100 means 100%  !!!!!!!!!!!
                tks = nlp.word_tokenize(item[0]) 
                ns = nlp.ner(item[0]) 
                
                f.write('\t'.join(['nt-{}'.format(str(i)), item[0], item[1], str(item[2]), '|'.join(tks), \
                    '|'.join(tks), '|'.join([p[1] for p in nlp.pos_tag(item[0])]), '|'.join([p[1] for p in ns]), \
                    '|'.join(['' if p[1] == 'O' else normalize_date_word(p[0]) if p[1] == 'DATE' and normalize_date_word(p[0]) is not None else str(normalize_num(p[0])[0][0]) if normalize_num(p[0])[0] is not None else '' for p in ns])]))
                f.write('\n')
            except:
                nlp.close()
                print(item[0])
                print(tks)
                print(ns)
                for p in ns:
                    print(normalize_date(p[0]))
                    print(normalize_num(p[0])[0])
                exit()
        f.close()


def create_small_dataset(args):
    # ============ use part of data (annotated in final version) =================
    # only use 20% data of tfc, finally just use all_ids in tfc instead of generate here
    # ============================================================================
    extracted_csvs = random.sample(os.listdir(args.csv_dir), 4000)

    with open(args.used_csv_ids, 'w') as f:
        for item in  extracted_csvs:
            f.write(item + '\n')
    exit()


stop_words = ['', 'doing', 'aren', 'how', 'wasn', 'few', 'well', 'mph', "'t", 'make', 'in', 'being', 'your', 
    'below', 'she', 'all', 'am', 'up', 'g', 'so', 'each', 'very', 'its', 'you', 'should', 'to', 'time', 'within', 
    'as', 'my', 'without', 'what', '.', 'ought', 'cannot', 'can', 'does', 'two', 'on', 'yourselves', 'times', 
    'between', 'n', 'we', 'he', 'any', 'him', 'this', 'about', 'i', 'same', 'over', 'if', ':', 'from', 'yourself', 
    'nor', 'most', 'hadn', "'m", '!', 'doesn', 'some', 'down', 'these', 'kg', 'myself', 'date', 'is', 'into', 'cm', 
    'once', 'it', 'them', 'yours', 'wouldn', 'which', 'didn', 'against', 'other', 'a', 'through', 'ours', 'more', 
    'me', 'why', 'while', 'first', 'where', 'then', 'hers', 'been', '+', 'her', 'after', "'d", 'whom', 'under', 
    'until', 'were', 'out', 'ft', 'his', 'when', 'theirs', 'lb', 'mi', 'had', 'every', 'before', 'and', 'at', "'s", 
    'those', 'themselves', 'not', 'both', 'further', 'who', 'weren', 'could', 'by', 'do', 'hasn', 'will', 'let', 
    ';', 'only', 'they', '(', 'of', 'are', 'himself', 'would', 'did', 'don', '?', 'or', "'ll", 'schools', 'an', 
    'itself', 'during', 'having', '-', 'than', 'shan', ',', 'shouldn', 'there', 'm', 'many', 'mustn', 'for', 'isn', 
    'such', 'also', 'less', 'above', ')', 'their', 'ourselves', 'too', 'but', 'again', 'has', 'our', 'herself', 
    'no', 'school', "'ve", 'here', 'because', "'nt", 'off', 'own', 'couldn', 'be', 'appear', 'have', 'number', 
    "'re", 'one', '&', 'with', 'was', 'haven', 'the', 'that', 's']


# following several func copyed from neural_symbolic_machine/preprocess.py
def tokens_contain(string_1, string_2):
    tks_1 = nltk.tokenize.word_tokenize(string_1)
    tks_2 = nltk.tokenize.word_tokenize(string_2)
    return set(tks_2).issubset(set(tks_1))


def string_in_table_tk(string, kg):
    props1 = set(kg['num_props'])
    props2 = set(kg['datetime_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop not in props1 and prop not in props2 and (isinstance(val[0], str) and tokens_contain(val[0], string)):
                return True
    return False
    

def string_in_table_str(string, kg):
    props1 = set(kg['num_props'])
    props2 = set(kg['datetime_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop not in props1 and prop not in props2 and isinstance(val[0], str) and string in val[0]:
                return True
    return False


def string_in_table(string, kg, use_tokens_contain=False):
    if use_tokens_contain:
        return string_in_table_tk(string, kg)
    else:
        return string_in_table_str(string, kg)


def date_match(string1, string2, strict_constrain=False):
    if string1 == 'XXXX-XX-XX' or string2 == 'XXXX-XX-XX': return False

    if strict_constrain:
        return string1 == string2
    
    year1, year2 = string1[:4], string2[:4]
    month1, month2 = string1[5:7], string2[5:7]
    day1, day2 = string1[-2:], string2[-2:]
    if year1 != 'XXXX' and year2 != 'XXXX' and year1 != year2:
        return False
    if month1 != 'XX' and month2 != 'XX' and month1 != month2:
        return False
    if day1 != 'XX' and day2 != 'XX' and day1 != day2:
        return False
    return True


def date_in_table(date, kg, strict_constrain=False):
    props = set(kg['datetime_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop in props and date_match(val[0], date, strict_constrain=False):
                return True
    return False


def num_in_table(num, kg):
    props = set(kg['num_props'])
    for k, node in kg['kg'].items():
        for prop, val in node.items():
            if prop in props and val[0] == num:
                return True
    return False


# match score between tab colomn(header cell) and question token
def prop_in_question_score(prop, tks, stop_words, use_one_hot=False):
    score = 0
    prop_tks = ('-'.join(prop[2:].split('-')[:-1])).split('_')
    for tk in prop_tks:
        if tk not in stop_words and tk in tks:
            score += 1
    if use_one_hot:
        score = min(score, 1)
    return score


# there can be several policy to expand: first match/longest match, it is hard to say which is better
# longest match to be implement: find all expanded match(may cover each other), dynamic program to find highest match score
# !!! neural match to be implement: input table+sentence into neural network, output reference/link between them
def expand_entities(examples, tabs):
    for example in examples:
        ents = [ent for ent in example['entities'] if ent['type'] == 'string_list' and ent['value'][0]]
        other_ents = [ent for ent in example['entities'] if ent['type'] != 'string_list']
        new_ents = []
        tab = tabs[example['context']]
        tks = example['tokens']
        
        for ent in ents:
            if new_ents and ent['token_end'] <= new_ents[-1]['token_end']:
                continue
            ent['value'] = [example['tokens'][ent['token_start']]]
            new_ents.append(ent)
            while new_ents[-1]['token_end'] < len(tks):
                end_tk = example['tokens'][new_ents[-1]['token_end']]
                end_tk = normalize_str(end_tk)
                new_value_strs = [s.join([new_ents[-1]['value'][0], end_tk]) for s \
                    in ['', ' ', '-', '- ', ' -', ' - ', '.', '_']]
                flg = 0
                for new_value_str in new_value_strs:
                    if string_in_table(new_value_str, tab):
                        new_ents[-1]['value'] = [new_value_str]
                        new_ents[-1]['token_end'] += 1
                        flg = 1
                        break
                if flg == 0: break

        example['entities'] = new_ents + other_ents


def expand_entities_by_tk_cover_rate(examples, tabs):
    for example in examples:
        ents = [ent for ent in example['entities'] if ent['type'] == 'string_list' and ent['value'][0]]
        other_ents = [ent for ent in example['entities'] if ent['type'] != 'string_list']
        new_ents = []
        tab = tabs[example['context']]
        for ent in ents:
            if new_ents and ent['token_end'] <= new_ents[-1]['token_end']:
                continue

            new_ents.append(ent)
            while new_ents[-1]['token_end'] < len(ents):
                tks = example['tokens'][new_ents[-1]['token_start'] : new_ents[-1]['token_end'] + 1]
                tks = [normalize_str(tk) for tk in tks]
                new_value_str = ' '.join(tks)
                if string_in_table(new_value_str, tab, use_tokens_contain=True) \
                    and len(tks) * 1.0 / len():
                    new_ents[-1]['value'] = [new_value_str]
                    new_ents[-1]['token_end'] += 1
                    break
        example['entities'] = new_ents + other_ents


def process_conjunction(examples):
    cnt = 0
    for example in examples:
        ents = [ent for ent in example['entities'] if ent['type'] == 'string_list']
        other_ents = [ent for ent in example['entities'] if ent['type'] != 'string_list']
        
        tks = example['tokens']

        if len(ents) == 0: continue
        if 'or' not in tks: continue
        
        idx = tks.index('or')
        before_ent = None
        after_ent = None
        before_id = None
        after_id = None
        for i, ent in enumerate(ents):
            if ent['token_end'] <= idx and idx - ent['token_end'] <= 2:
                before_ent = ent
                before_id = i
            elif ent['token_start'] > idx and after_ent is None and ent['token_start'] - idx <= 2:
                after_ent = ent
                after_id = i
        if before_ent is not None and after_ent is not None and before_id + 1 == after_id:
            ents[before_id] = {"token_end": after_ent['token_end'], "token_start": before_ent['token_start'], 
                "type": "string_list", "value": before_ent['value'] + after_ent['value']}
            del ents[after_id]
            cnt += 1

            example['entities'] = ents + other_ents

    return cnt


def combine_date_string(string1, string2, mode):

    if mode == 1 and string1[5:7] != 'XX' and string2[-2:] != 'XX':
        res = 'XXXX' + '-' + string1[5:7] + '-' + string2[-2:]
        return res
    elif mode == 2 and string1[:4] == 'XXXX' and string2[:4] != 'XXXX':
        res = string2[:4] + string1[4:]
        return res
    elif mode == 1 and string1[:4] != 'XXXX' and string2[5:7] != 'XX':
        res = string1[:4] + '-' + string2[5:7] + '-XX'
        return res
    elif mode == 2 and string1[-2:] == 'XX' and string2[-2:] != 'XX':
        res = string1[:-2] + string2[-2:]
        return res
    elif mode == 1 and string1[:4] != 'XXXX' and string2[-2:] != 'XX':
        res = string1[:4] + '-' + string2[-2:] + '-XX'
        return res
    return None


# totally 5 situation: year+month+day; year+month; month+day; year; month
def combine_date(examples):

    cnt1, cnt2, cnt3 = 0, 0, 0

    for example in examples:
        ents = [ent for ent in example['entities'] if ent['type'] == 'datetime_list']
        other_ents = [ent for ent in example['entities'] if ent['type'] != 'datetime_list']

        if len(ents) == 0: continue

        new_ents = []
        for idx, ent in enumerate(ents):
            if new_ents and ent['token_end'] <= new_ents[-1]['token_end']:
                continue

            new_ents.append(ent)
            cnt1 += 1
            
            if idx + 1 == len(ents): continue
            if ents[idx+1]['token_end'] - new_ents[-1]['token_end'] <= 2:
                res = combine_date_string(new_ents[-1]['value'][0], ents[idx+1]['value'][0], mode = 1)
                if res is not None:
                    new_ents[-1]['value'] = [res]
                    new_ents[-1]['token_end'] = ents[idx+1]['token_end']
                    cnt2 += 1
            
            if idx + 2 == len(ents): continue
            if ents[idx+2]['token_end'] - new_ents[-1]['token_end'] <= 2:
                res = combine_date_string(new_ents[-1]['value'][0], ents[idx+2]['value'][0], mode = 2)
                if res is not None:
                    new_ents[-1]['value'] = [res]
                    new_ents[-1]['token_end'] = ents[idx+2]['token_end']
                    cnt3 += 1

        example['entities'] = new_ents + other_ents

    return cnt1, cnt2, cnt3


def isnum(string):
    try:
        float(string)
        return True
    except:
        return False


def preprocess_sent(args):

    # here example_path is tagged examples after change_raw_to_tagged()
    with open(args.example_path, 'r') as f1, open(args.data_dict_file, 'r') as f2: # read tagged sent from example_path
        tabs = dict()
        for item in f2.readlines():
            obj = json.loads(item[:-1])
            tabs[obj['name']] = obj

        reader = csv.reader(f1, delimiter='\t', quotechar=None)
        header = next(reader)
        examples = []
        col_id = {'id': 0, 'utterance': 1, 'context': 2, 'targetValue': 3, 'tokens': 4, 
            'lemmaTokens': 5, 'posTags': 6, 'nerTags': 7, 'nerValues': 8}
        for index, row in tqdm(enumerate(reader)):
            # if index >= 200: continue
            e = dict()
            e['context'] = row[col_id['context']].split('.')[0]
            e['tokens'] = row[col_id['tokens']].split('|')
            e['pos_tags'] = row[col_id['posTags']].split('|')
            e['answer'] = ['True', 'True'] if row[col_id['targetValue']] == '1' else ['False', 'False']
            e['question'] = row[col_id['utterance']]
            
            e['processed_tokens'] = [normalize_str(tk) for tk in e['tokens']]
            e['entities'] = []
            e['in_table'] = [0] * len(e['tokens'])
            
            tks = e['tokens']
            tags = row[col_id['nerTags']].split('|')
            vals = row[col_id['nerValues']].split('|')
            tab = tabs[e['context']]

            for i, (tk, tag, val) in enumerate(zip(tks, tags, vals)):
                if tk not in stop_words and string_in_table(normalize_str(tk), tab, use_tokens_contain=True):
                    e['in_table'][i] = 1
                    e['entities'].append({"token_end": i+1, "token_start": i, "type": "string_list", "value": [normalize_str(tk)]})

                if val == '': continue
                if tag == 'DATE' and len(val) == 10 and val[4] == '-' and val[7] == '-':
                # if tag == 'DATE':
                    e['entities'].append({"token_end": i+1, "token_start": i, "type": "datetime_list", "value": [val]})
                    # ~~~~~~~~ to be determined whether to add year as num entity ~~~~~~~~~~~
                    # if val[4:] == '-XX-XXXX':
                    #     e['entities'].append({"token_end": i+1, "token_start": i, "type": "num_list", "value": [int(val[:4])]})
                    #     if num_in_table(int(val[:4]), tab):
                    #         e['in_table'][i] = 1
                    #         e['processed_tokens'][i] = '<{}>'.format(tag)
                    # elif val[:-2] == 'XXXX-XX-':
                    #     e['entities'].append({"token_end": i+1, "token_start": i, "type": "num_list", "value": [int(val[-2:])]})
                    #     if num_in_table(int(val[-2:]), tab):
                    #         e['in_table'][i] = 1
                    #         e['processed_tokens'][i] = '<{}>'.format(tag)
                    # also can add month
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    if date_in_table(val, tab, strict_constrain=False):
                        e['in_table'][i] = 1
                        e['processed_tokens'][i] = '<{}>'.format(tag)
                
                elif isnum(val):
                    e['entities'].append({"token_end": i+1, "token_start": i, "type": "num_list", "value": [float(val)]})
                    if num_in_table(float(val), tab):
                        e['in_table'][i] = 1
                        e['processed_tokens'][i] = '<{}>'.format(tag)

                elif tag != 'O':
                    e['processed_tokens'][i] = '<{}>'.format(tag)

            e['features'] = [[it] for it in e['in_table']]
            e['prop_features'] = dict([
                (prop, [prop_in_question_score(prop, e['tokens'], stop_words, use_one_hot=False)]) for prop in tab['props']
            ])
            examples.append(e)

        print('entities per sentence: {}'.format(sum([len(e['entities']) for e in examples])*1.0/len(examples)))
        if args.expand_entities:
            expand_entities(examples, tabs)
            print('entities per sentence after expand: {}'.format(sum([len(e['entities']) for e in examples])\
                * 1.0/len(examples)))
        
        if args.process_conjunction:
            print('{} or s be processd'.format(process_conjunction(examples)))
            print('entities per sentence after expand: {}'.format(sum([len(e['entities']) for e in examples]) \
                * 1.0 /len(examples)))
        
        if args.combine_date:
            cnt1, cnt2, cnt3 = combine_date(examples)
            print('{} date s be found'.format(cnt1))
            print('{} month-day s be processd'.format(cnt2))
            print('{} year-month-day s be processd'.format(cnt3))
            print('entities per sentence after expand: {}'.format(sum([len(e['entities']) for e in examples]) \
                * 1.0 /len(examples)))

        create_path(args.processed_example_path)
        # with open(args.processed_example_path + '/dev.jsonl', 'w') as f3:
        #     for e in examples:
        #         json.dump(e, f3)
        #         f3.write('\n')
        
        files = []
        for i in range(args.train_shard):
            files.append(\
                open(args.processed_example_path + '/train_shard_{}-{}.jsonl'.format(args.train_shard, i), 'w'))
        
        for i, e in enumerate(examples):
            json.dump(e, files[i%args.train_shard])
            files[i%args.train_shard].write('\n')
        
        for f in files:
            f.close()


if __name__ == '__main__':
    
    args = parser.parse_args()
    min_frac_for_props = float(args.min_frac_for_props)
    
    # create_small_dataset(args)

    nlp = stanfordcorenlp.StanfordCoreNLP(args.corenlp_path)

    preprocess_tab(args, nlp)
    #       ↓
    change_raw_to_tagged(args, nlp)
    #       ↓
    preprocess_sent(args)

    nlp.close()
    
