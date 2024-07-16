from paragraph_dataset import ParagraphDataset
from torch.utils.data import Dataset
import json

from rich.progress import track
from pathlib import Path
import torch

import numpy as np
import os
# from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel

class EmbeddingDataset(Dataset):
    def __init__(self, x_path, y_path, tokens_per_paragraph=256, bert_model_name="bert-base-uncased", device="cuda", **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.tokens_per_paragraph = tokens_per_paragraph
        self.embedding_dir = Path(f"{x_path}_embeddings")
        self.author_file = Path(f"{x_path}_authors")
        self.batch_size = 512

        if self.author_file.exists():
            print(f"Using author file: {self.author_file}")
            with open(self.author_file, 'r', encoding='utf-8') as f:
                self.authors = json.load(f)
        else:
            paragraphs = ParagraphDataset(x_path, y_path, device=device)
            self.authors = paragraphs.y
            
            # save authors file
            with open(self.author_file, 'w', encoding='utf-8') as f:
                json.dump(self.authors, f, ensure_ascii=False, indent=4)
            print(f"Saved authors to {self.author_file}")

        if self.embedding_dir.exists():
            # embeddings are loaded JIT, authors can be loaded right away
            print(f"Using embedding dir: {self.embedding_dir}")
        else:
            self.embedding_dir.mkdir(exist_ok=True)
            print(f"Saving embeddings to {self.embedding_dir}")

            if paragraphs is None:
                paragraphs = ParagraphDataset(x_path, y_path, device=device)
            paragraphs = paragraphs.x

            tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            bert_model = BertModel.from_pretrained(bert_model_name).to(self.device)
            # bert_model = BertModel.from_pretrained(bert_model_name)

            # Tokenize the dataset
            def tokenize_function(examples):
                encoded_input = [
                    tokenizer(p, padding='max_length', truncation=True, return_tensors='pt', max_length=self.tokens_per_paragraph)
                    for p in examples
                ]
                return encoded_input
            # def tokenize_function(examples):
            #     return tokenizer(examples, padding='max_length', truncation=True, return_tensors='pt', max_length=self.tokens_per_paragraph)

            tokenized_datasets = [tokenize_function(x) for x in track(paragraphs, total=len(paragraphs), description="Tokenizing paragraphs")]

            # Chosen to work around 8GB video memory, should take ~3GB
            for i in track(range(0, len(tokenized_datasets), self.batch_size), total=len(tokenized_datasets) // self.batch_size, description="Getting embeddings"):
                batch = tokenized_datasets[i:i+self.batch_size]

                batch_embeddings = []
                for problem in batch:
                    paragraph_embeddings = []
                    for e in problem:                
                        e.to(self.device)
                        with torch.no_grad():
                            embedding = bert_model(e['input_ids'], e['attention_mask']).last_hidden_state # shape: (batch_size, max_length, hidden_dim)
                        paragraph_embeddings.append(embedding)
                    batch_embeddings.append(paragraph_embeddings)

                self.save_embeddings(batch_embeddings, i)

        self.current_batch_index = -1
                

    def save_embeddings(self, embeddings, batch_index):
        """Save embeddings to disk."""
        torch.save(embeddings, self.embedding_dir / f"embeddings_batch_{batch_index}.pt")

    def load_embeddings(self, batch_index):
        """Save embeddings to disk."""
        return torch.load(self.embedding_dir / f"embeddings_batch_{batch_index}.pt")
            
    def __len__(self):
        return len(self.authors)

    def __getitem__(self, idx):
        # idx is the problem number, which is 1-indexed
        idx = idx - 1
        batch_index = (idx // self.batch_size) * self.batch_size
        index_in_batch = idx % self.batch_size

        # print(f"{batch_index=} {self.current_batch_index=}")

        if batch_index != self.current_batch_index:
            self.current_batch_index = batch_index
            self.batch_embeddings = self.load_embeddings(self.current_batch_index)

        return self.batch_embeddings[index_in_batch], self.authors[idx]