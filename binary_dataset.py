from paragraph_dataset import ParagraphDataset

from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
import torch
import itertools

from evaluation.evaluator import read_ground_truth_files

class BinaryDataset(ParagraphDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # track metadata about the pairs: Problem number, paragraph a number, paragraph b number
        # later this can be used to associated
        self.paragraph_pair_info = []
        
        for problem_num, (paragraphs, authors) in tqdm(enumerate(zip(self.x,self.y))):
            n = len(paragraphs)
            for para_a_num, para_b_num in itertools.combinations(range(n), 2):
                # metadata
                self.paragraph_pair_info.append((problem_num, para_a_num, para_b_num))

                self.binary_x.append((paragraphs[para_a_num], paragraphs[para_b_num]))
                self.binary_y.append(int(authors[para_a_num] == authors[para_b_num]))

            
    def __len__(self):
        return len(self.binary_y)

    def __getitem__(self, idx):
        return self.binary_x[idx], self.binary_y[idx]