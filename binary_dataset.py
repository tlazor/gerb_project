from paragraph_dataset import ParagraphDataset

from natsort import natsorted
from rich.progress import track
from pathlib import Path
import torch
import itertools

from evaluation.evaluator import read_ground_truth_files

class BinaryDataset(ParagraphDataset):
    def __init__(self, x_path, y_path, **kwargs):
        super().__init__(x_path, y_path, **kwargs)

        # track metadata about the pairs: Problem number, paragraph a number, paragraph b number
        # later this can be used to associated
        self.paragraph_pair_info = []

        # Task 3 paragraph pairs and binary comparison results
        self.binary_x = []
        self.binary_y = []
        
        for problem_num, (paragraphs, authors) in track(enumerate(zip(self.x,self.y), start=1), description="Generating Task 3 binary data"):
            n = len(paragraphs)
            for para_a_num, para_b_num in itertools.combinations(range(n), 2):
                # metadata
                self.paragraph_pair_info.append((problem_num, para_a_num, para_b_num))
                
                binary_x = (paragraphs[para_a_num], paragraphs[para_b_num])
                self.binary_x.append(binary_x)
                binary_y = float(authors[para_a_num] == authors[para_b_num])
                self.binary_y.append(torch.tensor([binary_y], device=self.device))

            
    def __len__(self):
        return len(self.binary_y)

    def __getitem__(self, idx):
        return self.binary_x[idx], self.binary_y[idx]