### Learning to Generate Programs for Table Fact Verification via Structure-Aware Semantic Parsing
This repository contains a preliminary release of code for SASP used for experiments in our ACL 2022 paper "Learning to Generate Programs for Table Fact Verification via Structure-Aware Semantic Parsing". It is based on the open source implementation of [pytorch neural symbolic machine](https://github.com/pcyin/pytorch_neural_symbolic_machines).

The difference with Pytorch Neural Symbolic Machines:
- We postprocess the dependency tree to generate operation oriented tree, which will be used to sample programs to bootstrap SASP and bias the program generation for Tabfact. 
- We change the reward function to Maximum Likelyhood Most Violation Reward, which achieves better performance than the binary reward.
- We use an interpreter more suitable for Tabfact. We also revised the [data pre-processing script](https://github.com/crazydonkey200/neural-symbolic-machines/blob/master/table/wtq/preprocess.py), and processed the TABFACT dataset with our implementation.
- The original implementation will first sample, then train and evaluate at last. We make sampling, training and evaluating threads work parallelly by multiprocessing and thread lock. What's more, the original implementation uses one GPU for model training, one GPU for model evaluating and multiple GPUs for program sampling. The program sampling is the most time consuming procedure, so we use one GPU for Sampling, and use another GPU for both training and evaluating. In this way, both two GPUs are almost fully loaded, and the training speed is much faster.

### Requirement
- python 3.8.10
- pytorch 1.9.0
- You should install package related to [Tabert](https://github.com/facebookresearch/TaBERT), [Pytorch Neural Symbolic Machine](https://github.com/pcyin/pytorch_neural_symbolic_machines) and [CRF2o](https://github.com/yzhangcs/parser)
- A workstation with 128 GB of RAM and 2 RTX 3090 GPUs

### Dataset Preparation
- Firstly, you should download the dataset from [the official website of Tabfact](https://tabfact.github.io/), then preprocess it with our script SASP/table/tabfact/preprocess_example.py. Or just use the preprocessed data under SASP/data/tabfact/data_shard
- Secondly, generate the operation oriented tree with the script SASP/table/tabfact/gen_with_func.py, or just use processed data inside data/tabfact/data_shard
- Thirdly, sample some examples in our paper to bootstrap SASP with the script SASP/table/random_explore.py. Or just download [saved_program.jsonl](https://drive.google.com/file/d/1Gh4B66NWZ6rVGo8Oir2YJSfNBRmE2uIb/view?usp=sharing), and put it into data/tabfact/

### Training
Run the following command under SASP/:
```
python -m table.experiments train --seed=0 --cuda --config=data/config/config.vanilla_bert.json --work-dir=runs/demo_run --extra-config='{}'

```
It will take around 20 hours.

### Testing
Run the following command under SASP/:
```
python -m table.experiments test --cuda --eval-batch-size=128 --eval-beam-size=4 --save-decode-to=runs/demo_run/test_result.json --model=./runs/demo_run/model.best.bin --test-file=data/tabfact/data_shard/test.jsonl --extra-config='{}'
```
You can also download our [trained model](https://drive.google.com/file/d/1TyleYW54hLJp8ZaC13vD33BJQnVPvbnF/view?usp=sharing), and put it into runs/demo_run for testing.

### Reference
Please consider citing the following papers if you are using our codebase.
```
@incollection{NIPS2018_8204,
title = {Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing},
author = {Liang, Chen and Norouzi, Mohammad and Berant, Jonathan and Le, Quoc V and Lao, Ni},
booktitle = {Advances in Neural Information Processing Systems 31},
editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
pages = {10015--10027},
year = {2018},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8204-memory-augmented-policy-optimization-for-program-synthesis-and-semantic-parsing.pdf}
}

@inproceedings{liang2017neural,
  title={Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision},
  author={Liang, Chen and Berant, Jonathan and Le, Quoc and Forbus, Kenneth D and Lao, Ni},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  volume={1},
  pages={23--33},
  year={2017}
}

@inproceedings{yin20acl,
    title = {Ta{BERT}: Pretraining for Joint Understanding of Textual and Tabular Data},
    author = {Pengcheng Yin and Graham Neubig and Wen-tau Yih and Sebastian Riedel},
    booktitle = {Annual Conference of the Association for Computational Linguistics (ACL)},
    month = {July},
    year = {2020}
}
```
