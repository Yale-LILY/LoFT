# LoFT Evaluation
For BLEU-1/2/3, SP-Acc, and NLI-Acc, please use the official implementation of [LogicNLG evaluation](https://github.com/wenhuchen/LogicNLG).

Prepare the output file from *LoFT Inference* in the following format:
```
[
    "2-18424778-6.html.csv": [
        "2 of the album 's release were not released in the united kingdom",
        "australia and the united kingdom are the only country to appear on the chart more than 1 time",
        "2 of the album 's release date were later than 20 october 2008",
        "sony music did not release the album in the united kingdom , germany , or japan",
        "digital download is the only format besides cd and double lp"
    ],
    "2-1458666-4.html.csv": [
        ...
]
```
## Diversity-level Evaluation
For diversity-level evaluation (i.e., Distinct-2, and Self-BLEU-4), run the following command:
```shell
python diversity_metrics.py --verify_file [output path]
```

## TAPEX-Acc
For TAPEX-Acc, turn to the same virtual enviroment as *LoFT Framework*, and run the following two commands:
```shell
python prepare_tapex_evaluation_data.py --verify_file [output path]
python TAPEX_Acc.py --do_predict --per_device_eval_batch_size 1 --output_dir .
```

## Note:
The overall file structure can be viewed at `file_structure.txt`. 
