import json
import os
import re
import sys
import time
from typing import List, Union
import numpy as np
from tensorboardX import SummaryWriter

from nsm import nn_util
from nsm.parser_module import get_parser_agent_by_name
from nsm.parser_module.agent import PGAgent
from nsm.consistency_utils import ConsistencyModel, QuestionSimilarityModel
from nsm.env_factory import QAProgrammingEnv, Sample

import torch.multiprocessing as torch_mp
from multiprocessing import Queue, Process

import torch

import gc


class Evaluation(object):
    @staticmethod
    def evaluate(model, dataset: List[QAProgrammingEnv], beam_size: int):
        was_training = model.training
        model.eval()

        decode_results = model.decode_examples(dataset, beam_size=beam_size)
        eval_results = Evaluation.evaluate_decode_results(dataset, decode_results)

        if was_training:
            model.train()

        return eval_results

    @staticmethod
    def evaluate_decode_results(dataset: List[QAProgrammingEnv], decoding_results=Union[List[List[Sample]], List[Sample]], verbose=False):
        if isinstance(decoding_results[0], Sample):
            decoding_results = [[hyp] for hyp in decoding_results]

        acc_list = []
        oracle_acc_list = []
        program_pred_list = []

        for env, hyp_list in zip(dataset, decoding_results):
            is_top_correct = len(hyp_list) > 0 and hyp_list[0].trajectory.reward > 0.
            has_correct_program = any(hyp.trajectory.reward > 0. for hyp in hyp_list)
            has_program = len(hyp_list) > 0

            acc_list.append(is_top_correct)
            oracle_acc_list.append(has_correct_program)
            program_pred_list.append(has_program)
        print('acc:', np.average(acc_list))
        print('at least one program predicted:', np.average(program_pred_list))

        eval_result = dict(accuracy=np.average(acc_list),
                           oracle_accuracy=np.average(oracle_acc_list))

        return eval_result


class Evaluator(torch_mp.Process):
    def __init__(self, config, eval_file, device):
        super(Evaluator, self).__init__(daemon=True)
        self.eval_queue = Queue()
        self.config = config
        self.eval_file = eval_file
        self.device = device

        self.model_path = 'INIT_MODEL'
        self.message_var = None

    def run(self):
        # initialize cuda context
        self.device = torch.device(self.device)
        if 'cuda' in self.device.type:
            torch.cuda.set_device(self.device)

        # seed the random number generators
        nn_util.init_random_seed(self.config['seed'], self.device)

        agent_name = self.config.get('parser', 'vanilla')
        self.agent = get_parser_agent_by_name(agent_name).build(self.config, master='evaluator').to(self.device).eval()

        self.load_environments()
        files_in_tb_log = os.listdir(os.path.join(self.config['work_dir'], 'tb_log/dev'))
        for f in files_in_tb_log:
            os.remove(os.path.join(self.config['work_dir'], 'tb_log/dev', f))
        summary_writer = SummaryWriter(os.path.join(self.config['work_dir'], 'tb_log/dev'))

        dev_scores = []

        while True:
            if self.check_and_load_new_model(): # wait until new model was saved (after one step finished in learner, new model will be saved)
                print(f'[Evaluator] evaluate model [{self.model_path}]', file=sys.stderr)
                t1 = time.time()

                decode_results = self.agent.decode_examples(self.environments, beam_size=self.config['beam_size'], batch_size=32)

                eval_results = Evaluation.evaluate_decode_results(self.environments, decode_results)

                t2 = time.time()
                print(f'[Evaluator] step={self.get_global_step()}, result={repr(eval_results)}, took {t2 - t1}s', file=sys.stderr)

                summary_writer.add_scalar('eval/accuracy', eval_results['accuracy'], self.get_global_step())
                summary_writer.add_scalar('eval/oracle_accuracy', eval_results['oracle_accuracy'], self.get_global_step())

                dev_score = eval_results['accuracy']
                if not dev_scores or max(dev_scores) < dev_score:
                    print(f'[Evaluator] save the current best model', file=sys.stderr)

                    with open(os.path.join(self.config['work_dir'], 'dev.log'), 'w') as f:
                        f.write(json.dumps(eval_results))

                    self.agent.save(os.path.join(self.config['work_dir'], 'model.best.bin'))

                dev_scores.append(dev_score)
                if 'cuda' in str(self.device.type):
                    mem_cached_mb = torch.cuda.memory_reserved() / 1000000
                    if mem_cached_mb > 2500:
                        print(f'Evaluator empty cached memory [{mem_cached_mb} MB]', file=sys.stderr)
                        torch.cuda.empty_cache()
                sys.stderr.flush()

                del(decode_results); del(eval_results)
                gc.collect()

            time.sleep(0.1)  # in seconds

    def load_environments(self):
        from table.experiments import load_environments
        envs = load_environments([self.eval_file],
                                 table_file=self.config['table_file'],
                                 table_representation_method=self.config['table_representation'],
                                 bert_tokenizer=self.agent.encoder.bert_model.tokenizer)
        for env in envs:
            env.use_cache = False
            env.punish_extra_work = False

        self.environments = envs

    def check_and_load_new_model(self):
        new_model_path = self.message_var.value.decode()

        if new_model_path and new_model_path != self.model_path:
            t1 = time.time()

            state_dict = torch.load(new_model_path, map_location=lambda storage, loc: storage)
            self.agent.load_state_dict(state_dict)
            self.model_path = new_model_path

            t2 = time.time()
            print('[Evaluator] loaded new model [%s] (took %.2f s)' % (new_model_path, t2 - t1), file=sys.stderr)

            return True
        else:
            return False

    def get_global_step(self):
        if not self.model_path:
            return 0

        model_name = self.model_path.split('/')[-1]
        train_iter = re.search('iter(\d+)?', model_name).group(1)

        return int(train_iter)
