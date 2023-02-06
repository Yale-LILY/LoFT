import os


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


class LoFTPaths:
    def __init__(self, loft_root: str) -> None:
        #! inputs
        self.loft_root = loft_root
        self.data_root = os.path.join(self.loft_root, "data")
        self.inference_root = os.path.join(self.loft_root, "inference")
        self.train_root = os.path.join(self.loft_root, "train")
        self.translate_root = os.path.join(self.loft_root, "translate")
        self.logicnlg_root = os.path.join(self.data_root, "logicnlg")
        self.logic2text_root = os.path.join(self.data_root, "logic2text")
        self.all_csv_root = os.path.join(self.logicnlg_root, "all_csv")
        self.sasp_root = os.path.join(self.train_root, "SASP")
        
        #! outputs
        self.train_output_root = create_path(os.path.join(self.train_root, "out"))
        self.inference_output_root = create_path(os.path.join(self.inference_root, "out"))
        