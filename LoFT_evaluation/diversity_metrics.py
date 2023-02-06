from collections import Counter
import numpy as np
import json, nltk
import argparse
from tqdm import tqdm
import warnings
from multiprocessing import Pool

warnings.filterwarnings("ignore", category=UserWarning)

def read_seqs(path):
    all_seqs = []
    data = json.load(open(path, "r"))
    for table_id in data:
        seqs = []
        for seq in data[table_id]:
            seqs.append(seq.lower().split(' '))
        all_seqs.append(seqs)
    return all_seqs
        

def distinct(all_seqs):
    """ Calculate intra/inter distinct 1/2. """
    distinct2s = []
    for seqs in all_seqs:
        bigrams_all = Counter()
        for seq in seqs:
            bigrams = Counter(zip(seq, seq[1:]))
            bigrams_all.update(bigrams)
    
        distinct2s.append((len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5))
    
    distinct2 = np.average(distinct2s)
    print('Distinct-2: %.1f' % (distinct2*100))
    return distinct2

def func_compute_bleu(sent, other_sents):
    return nltk.translate.bleu_score.sentence_bleu(other_sents, sent, weights=(0.25, 0.25, 0.25, 0.25))


def self_bleu_4(all_seqs):
    """ Calculate self-BLEU-4. """
    pool = Pool(64)

    seqs, sent_list, other_sents_list = [], [], []
    for seq in all_seqs:
        seqs.extend(seq)

    for seqs in all_seqs:
        for idx, sent in enumerate(seqs):
            sent_list.append(sent)
            other_sents_list.append(seqs[:idx] + seqs[idx+1:])
    
    bleus = pool.starmap(func_compute_bleu, zip(sent_list, other_sents_list))
    
    print('Self-BLEU-4: %.1f' % (np.average(bleus)*100))
    return np.average(bleus)

def diversity_metrics(path):
    seqs = read_seqs(path)
    self_bleu = self_bleu_4(seqs)
    inter_dist1 = distinct(seqs)
    return inter_dist1, self_bleu

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    args = parser.parse_args()
    _, self_bleu = diversity_metrics(args.input_file)