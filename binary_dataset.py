from embedding_dataset import EmbeddingDataset
from torch.utils.data import Dataset

from rich.progress import track
import torch
import itertools

class BinaryDataset(Dataset):
    def __init__(self, x_path, y_path, device="cuda", **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.embedding_dataset = EmbeddingDataset(x_path, y_path, device=self.device)

        # track metadata about the pairs: Problem number, paragraph a number, paragraph b number
        # later this can be used to associated
        self.paragraph_pair_info = []
        
        for problem_num, authors in track(enumerate(self.embedding_dataset.authors, start=1), description="Generating Task 3 binary data", total=len(self.embedding_dataset)):
            n = len(authors)
            for para_a_num, para_b_num in itertools.combinations(range(n), 2):
                # metadata
                self.paragraph_pair_info.append((problem_num, para_a_num, para_b_num))
            
    def __len__(self):
        return len(self.paragraph_pair_info)

    def __getitem__(self, idx):
        problem_num, para_a_num, para_b_num = self.paragraph_pair_info[idx]

        paragraph_embeddings, authors = self.embedding_dataset[problem_num]

        binary_x = (paragraph_embeddings[para_a_num], paragraph_embeddings[para_b_num])
        binary_y = float(authors[para_a_num] == authors[para_b_num])
        
        return binary_x, torch.tensor(binary_y, device=self.device)