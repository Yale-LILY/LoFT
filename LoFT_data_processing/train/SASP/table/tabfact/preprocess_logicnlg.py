import json
import sys
from nltk.tokenize import word_tokenize
import os
from tqdm import tqdm

sys.path.append('../../../..')
from utils.LoFTPaths import LoFTPaths, create_path

paths = LoFTPaths(loft_root=os.path.abspath('../../../../'))

logicnlg_train_path = os.path.join(paths.logicnlg_root, 'train_lm.json')

train_examples = json.load(open(logicnlg_train_path, 'r'))
table_ids = []
output_examples = {}
for table_id in tqdm(train_examples):
    table_ids.append(table_id)
    output_examples[table_id] = [[],[]]
    for example in train_examples[table_id]:
        sent = example[0]
        tokenized_sent = " ".join(word_tokenize(sent))
        output_examples[table_id][0].append(tokenized_sent)
        output_examples[table_id][1].append(1)

output_path = os.path.join(paths.train_output_root, 'train_lm_tokenized.json')
json.dump(output_examples, open(output_path, 'w'), indent=4)

output_csv_ids_path = os.path.join(paths.train_output_root, 'train_csv_ids.txt')
# use writeline
with open(output_csv_ids_path, 'w') as f:
    f.writelines("\n".join(table_ids) + "\n")

        