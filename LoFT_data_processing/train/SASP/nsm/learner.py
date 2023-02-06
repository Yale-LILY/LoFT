import ctypes
import heapq
import json
import math
import os
import random
import time
from itertools import chain
from pathlib import Path
from typing import List, Dict, Union

import numpy as np
import sys

import torch.multiprocessing as torch_mp
import multiprocessing

from transformers import AdamW, get_linear_schedule_with_warmup
from table_bert.vertical.vertical_attention_table_bert import VerticalAttentionTableBert

from nsm import nn_util
from nsm.parser_module import get_parser_agent_by_name
from nsm.parser_module.agent import PGAgent
from nsm.consistency_utils import ConsistencyModel, QuestionSimilarityModel
from nsm.evaluator import Evaluation
from nsm.program_cache import SharedProgramCache

import torch
from tensorboardX import SummaryWriter

from nsm.sketch.sketch_predictor import SketchPredictor
from nsm.sketch.trainer import SketchPredictorTrainer


class Learner(torch_mp.Process):
    def __init__(self, config: Dict, devices: Union[List[torch.device], torch.device], shared_program_cache: SharedProgramCache = None):
        super(Learner, self).__init__(daemon=True)

        self.train_queue = multiprocessing.Queue()
        self.checkpoint_queue = multiprocessing.Queue()
        self.config = config
        self.devices = devices
        self.actor_message_vars = []
        self.current_model_path = None
        self.shared_program_cache = shared_program_cache

        self.actor_num = 0

    def run(self):
        # initialize cuda context
        devices = self.devices if isinstance(self.devices, list) else [self.devices]
        self.devices = [torch.device(device) for device in devices]

        # seed the random number generators
        for device in self.devices:
            nn_util.init_random_seed(self.config['seed'], device)

        agent_name = self.config.get('parser', 'vanilla')
        self.agent = get_parser_agent_by_name(agent_name).build(self.config, master='learner').to(self.devices[0]).train()

        self.train()

    def train(self):
        model = self.agent
        config = self.config
        work_dir = Path(config['work_dir'])
        train_iter = 0
        save_every_niter = config['save_every_niter']
        entropy_reg_weight = config['entropy_reg_weight']
        files_in_tb_log = os.listdir(os.path.join(config['work_dir'], 'tb_log/train'))
        for f in files_in_tb_log:
            os.remove(os.path.join(config['work_dir'], 'tb_log/train', f))
        summary_writer = SummaryWriter(os.path.join(config['work_dir'], 'tb_log/train'))
        max_train_step = config['max_train_step']
        save_program_cache_niter = config.get('save_program_cache_niter', 0)
        freeze_bert_for_niter = config.get('freeze_bert_niter', 0)
        gradient_accumulation_niter = config.get('gradient_accumulation_niter', 1) # !!! this parameter not set in config.json
        use_trainable_sketch_predictor = self.config.get('use_trainable_sketch_predictor', False)

        bert_params = [
            (p_name, p)
            for (p_name, p) in model.named_parameters()
            if 'bert_model' in p_name and p.requires_grad
        ]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_grouped_parameters = [
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        bert_optimizer = AdamW(
            bert_grouped_parameters,
            lr=self.config['bert_learning_rate'],
            correct_bias=False)
        bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=0.1*max_train_step, num_training_steps=max_train_step)

        other_params = [
            p
            for n, p
            in model.named_parameters()
            if 'bert_model' not in n and p.requires_grad
        ]

        other_optimizer = torch.optim.Adam(other_params, lr=0.0003)

        cum_loss = cum_examples = 0.

        while train_iter < max_train_step:
            if 'cuda' in self.devices[0].type:
                torch.cuda.set_device(self.devices[0])

            train_iter += 1
            other_optimizer.zero_grad()
            bert_optimizer.zero_grad()

            while True:
                if self.train_queue.qsize() > 0:
                    break
                time.sleep(0.1)

            train_samples, samples_info = self.train_queue.get()
            try:
                queue_size = self.train_queue.qsize()
                # print(f'[Learner] train_iter={train_iter} train queue size={queue_size}', file=sys.stderr)
                summary_writer.add_scalar('train_queue_size', queue_size, train_iter)
            except NotImplementedError:
                pass

            train_trajectories = [sample.trajectory for sample in train_samples]

            # to save memory, for vertical tableBERT, we partition the training trajectories into small chunks
            # !!!!!! maybe we don't need this with rtx 3090, or we can have bigger chunk_size
            if isinstance(self.agent.encoder.bert_model, VerticalAttentionTableBert) and 'large' in self.agent.encoder.bert_model.config.base_model_name:
                chunk_size = 5
            else:
                # chunk_size = len(train_samples)
                chunk_size = 16

            chunk_num = int(math.ceil(len(train_samples) / chunk_size))
            cum_loss = 0.
            if chunk_num > 1:
                for chunk_id in range(0, chunk_num):
                    train_samples_chunk = train_samples[chunk_size * chunk_id: chunk_size * chunk_id + chunk_size]
                    loss_val = self.train_step(train_samples_chunk, train_iter, summary_writer)
                    cum_loss += loss_val

                grad_multiply_factor = 1 / len(train_samples)
                for p in self.agent.parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(grad_multiply_factor)
            else:
                loss_val = self.train_step(train_samples, train_iter, summary_writer, reduction='mean')
                cum_loss = loss_val * len(train_samples)

            # clip gradient
            torch.nn.utils.clip_grad_norm_(other_params, 5.)
            for group in bert_grouped_parameters:
                for p in group['params']:
                    torch.nn.utils.clip_grad_norm_(p, 1.)

            if train_iter % gradient_accumulation_niter == 0:
                other_optimizer.step()

                if train_iter > freeze_bert_for_niter:
                    bert_optimizer.step()
                    bert_scheduler.step()
                if train_iter % freeze_bert_for_niter == 0:
                    print(f'[Learner] train_iter={train_iter} reset Adam optimizer')
                    other_optimizer = torch.optim.Adam(other_params, lr=0.0003) # !!! why reset?

            # print(f'[Learner] train_iter={train_iter} loss={loss_val}', file=sys.stderr)

            if 'clip_frac' in samples_info:
                summary_writer.add_scalar('sample_clip_frac', samples_info['clip_frac'], train_iter)

            # update sketch predictor
            if use_trainable_sketch_predictor:
                if 'cuda' in self.devices[1].type:
                    torch.cuda.set_device(self.devices[1])

                self.sketch_predictor_trainer.step(train_trajectories, train_iter=train_iter)

            cum_examples += len(train_samples)

            self.try_update_model_to_actors(train_iter)

            # if train_iter % save_every_niter == 0:
            #     print(f'[Learner] train_iter={train_iter} avg. loss={cum_loss / cum_examples}, '
            #           f'{cum_examples} examples ({cum_examples / (time.time() - t1)} examples/s)', file=sys.stderr)
            #     cum_loss = cum_examples = 0.
            #     t1 = time.time()

            #     # log stats of the program cache
            #     program_cache_stat = self.shared_program_cache.stat()
            #     summary_writer.add_scalar(
            #         'avg_num_programs_in_cache',
            #         program_cache_stat['num_entries'] / program_cache_stat['num_envs'],
            #         train_iter
            #     )
            #     summary_writer.add_scalar(
            #         'num_programs_in_cache',
            #         program_cache_stat['num_entries'],
            #         train_iter
            #     )

            # if save_program_cache_niter > 0 and train_iter % save_program_cache_niter == 0:
            #     program_cache_file = work_dir / 'log' / f'program_cache.iter{train_iter}.json'
            #     program_cache = self.shared_program_cache.all_programs()
            #     json.dump(
            #         program_cache,
            #         program_cache_file.open('w'),
            #         indent=2
            #     )

            del(train_samples); del(samples_info)
            if 'cuda' in str(self.devices[0].type):
                mem_cached_mb = torch.cuda.memory_reserved() / 1000000
                if mem_cached_mb > 10000:
                    print(f'Learner empty cached memory [{mem_cached_mb} MB]', file=sys.stderr)
                    torch.cuda.empty_cache()

        summary_writer.close()

    def try_update_model_to_actors(self, train_iter):
        save_every_niter = self.config.get('save_every_niter')
        if train_iter % save_every_niter == 0:
            self.update_model_to_actors(train_iter)
        else:
            # self.push_new_model(self.current_model_path)
            pass

    def train_step(self, train_samples, train_iter, summary_writer, reduction='sum'):
        train_trajectories = [sample.trajectory for sample in train_samples]

        # (batch_size)
        batch_log_prob, meta_info = self.agent(train_trajectories, return_info=True)

        train_sample_weights = batch_log_prob.new_tensor([s.weight for s in train_samples])
        # actor.py line 334 add traj with reward 0. so here s.trajectory.reward if s.trajectory.reward > 0. else 0.5 -> correspond to eq(5)
        train_sample_rewards = batch_log_prob.new_tensor([10 * (s.trajectory.reward if s.trajectory.reward > 0. else 0.2) for s in train_samples])
        batch_log_prob = batch_log_prob * train_sample_weights * train_sample_rewards

        if reduction == 'sum':
            loss = -batch_log_prob.sum()
        elif reduction == 'mean':
            loss = -batch_log_prob.mean()
        else:
            raise ValueError(f'Unknown reduction {reduction}')

        gradient_accumulation_niter = self.config.get('gradient_accumulation_niter', 1)
        if gradient_accumulation_niter > 1:
            loss /= gradient_accumulation_niter

        summary_writer.add_scalar('parser_loss', loss.item(), train_iter)

        loss.backward()
        loss_val = loss.item()

        return loss_val

    def update_model_to_actors(self, train_iter):
        t1 = time.time()
        model_state = self.agent.state_dict()
        model_save_path = os.path.join(self.config['work_dir'], 'agent_state.iter%d.bin' % train_iter)
        torch.save(model_state, model_save_path)

        if hasattr(self, 'sketch_predictor_server_msg_val'):
            sketch_predictor_path = str(Path(model_save_path).with_suffix('.sketch_predictor.bin'))
            torch.save(self.sketch_predictor.state_dict(), sketch_predictor_path)
        else:
            sketch_predictor_path = None

        self.push_new_model(model_save_path, sketch_predictor_path=sketch_predictor_path)
        print(f'[Learner] pushed model [{model_save_path}] (took {time.time() - t1}s)', file=sys.stderr)
        if sketch_predictor_path:
            print(f'[Learner] pushed sketch prediction model [{sketch_predictor_path}] (took {time.time() - t1}s)', file=sys.stderr)

        if self.current_model_path:
            os.remove(self.current_model_path)
            sketch_predictor_server_msg_val = getattr(self, 'sketch_predictor_server_msg_val', None)
            if sketch_predictor_server_msg_val:
                os.remove(str(Path(self.current_model_path).with_suffix('.sketch_predictor.bin')))

        self.current_model_path = model_save_path

    def push_new_model(self, model_path, sketch_predictor_path=None):
        self.checkpoint_queue.put(model_path)
        if model_path:
            self.eval_msg_val.value = model_path.encode()

            table_bert_server_msg_val = getattr(self, 'table_bert_server_msg_val', None)
            if table_bert_server_msg_val:
                table_bert_server_msg_val.value = model_path.encode()

        if sketch_predictor_path:
            sketch_predictor_server_msg_val = getattr(self, 'sketch_predictor_server_msg_val', None)
            if sketch_predictor_server_msg_val:
                sketch_predictor_server_msg_val.value = sketch_predictor_path.encode()

    def register_actor(self, actor):
        actor.checkpoint_queue = self.checkpoint_queue
        actor.train_queue = self.train_queue
        self.actor_num += 1

    def register_evaluator(self, evaluator):
        msg_var = multiprocessing.Array(ctypes.c_char, 4096)
        self.eval_msg_val = msg_var
        evaluator.message_var = msg_var

    def register_table_bert_server(self, table_bert_server):
        msg_val = multiprocessing.Array(ctypes.c_char, 4096)
        self.table_bert_server_msg_val = msg_val
        table_bert_server.learner_msg_val = msg_val

    def register_sketch_predictor_server(self, sketch_predictor_server):
        msg_val = multiprocessing.Array(ctypes.c_char, 4096)
        self.sketch_predictor_server_msg_val = msg_val
        sketch_predictor_server.learner_msg_val = msg_val

