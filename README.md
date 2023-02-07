# LoFT
The data and code for EACL 2023 paper [LoFT: Enhancing Faithfulness and Diversity for Table-to-Text Generation via Logic Form Control](https://arxiv.org/abs/2302.02962), which applies logical forms as fact checkers and content planners to improve faithfulness and diversity of logical table-to-text generation simultaneously.
<p align="center">
<img src="model_view.jpg" width="800">
</p>

## Main Modules
The LoFT code is organized into the following three modules (in order of execution):
- `LoFT_data_processing`: which prepares the training and inference data for LoFT.
- `LoFT_framework`: which contains the implementation of training and inference process of LoFT.
- `LoFT_evaluation`: which contains the implementation of evaluation metrics of LoFT.

We provide details of each module in `README.md` under each folder, along with the `requirements.txt` and [Google Drive links](https://drive.google.com/drive/folders/1A2zkN00KJCLc2fq4QR9u0InGLD72HVyC?usp=sharing) of model checkpoints and processed data.


## Contact
For any issues or questions, kindly email us at: Yilun Zhao (yilun.zhao@yale.edu), or Zhenting Qi (qi11@illinois.edu).

## Citation
```
@inproceedings{zhao-etal-2023-loft,
    title = "LoFT: Enhancing Faithfulness and Diversity for Table-to-Text Generation via Logic Form Control",
    author = "Zhao, Yilun  and
      Qi, Zhenting  and
      Nan, Linyong  and
      Flores, Lorenzo Jaime and
      Radev, Dragomir",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month = may,
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/pdf/2302.02962.pdf"
}
```