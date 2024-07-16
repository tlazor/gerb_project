from torch.utils.data import Dataset
from natsort import natsorted
from rich.progress import track
from pathlib import Path
import torch
import glob
import json
import os

from evaluation.evaluator import read_ground_truth_files

def read_ground_truth_files_progress(truth_folder):
    """
    reads ground truth files into dict
    :param truth_folder: path to folder holding ground truth files
    :return: dict of ground truth files with problem-id as key and file content as value
    """
    truth = {}
    files = glob.glob(os.path.join(truth_folder, 'truth-problem*.json'))

    for truth_file in track(files, description="Reading ground truth files", total=len(files)):
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[6:-5]] = curr_truth
    return truth

class ParagraphDataset(Dataset):
    def __init__(self, x_path, y_path, device='cuda', **kwargs):
        super().__init__(**kwargs)

        self.x = []
        self.y = []

        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Attempting to use CUDA when it is not available, using CPU instead")
            device = 'cpu'

        self.device = device
        
        problem_files = natsorted(Path(x_path).glob('problem-*.txt'))
        gt = read_ground_truth_files_progress(y_path)

        for problem_num, problem_file in track(enumerate(problem_files, start=1), description="Loading problem files", total=len(problem_files)):
            assert problem_file.stem == f"problem-{problem_num}", f"Problem files out of order or missing {problem_num=} {problem_file.stem=}"
            
            paragraph_authors = gt[f"problem-{problem_num}"]["paragraph-authors"]
            with open(problem_file, 'r', encoding="utf8") as fh:
                paragraphs = fh.readlines()

            self.x.append(paragraphs)
            self.y.append(paragraph_authors)
            
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]