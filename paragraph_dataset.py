from torch.utils.data import Dataset
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
import torch

from evaluation.evaluator import read_ground_truth_files

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
        gt = read_ground_truth_files(y_path)

        for problem_num, problem_file in tqdm(enumerate(problem_files), desc="Loading problem and ground truth files"):
            assert problem_file.stem == f"problem-{problem_num}.txt", f"Problem files out of order or missing {problem_num=} {problem_file=}"
            
            paragraph_authors = gt[f"problem-{problem_num+1}"]["paragraph-authors"]
            with open(problem_file, 'r', encoding="utf8") as fh:
                paragraphs = fh.readlines()

            self.x.append(paragraphs)
            self.y.append(paragraph_authors)
            
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]