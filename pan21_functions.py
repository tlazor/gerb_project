import math
from tensorflow.keras.utils import PyDataset
from pathlib import Path
from joblib import Parallel, delayed, parallel_config
import os

import torch
import numpy as np
from tqdm.notebook import tqdm

from evaluation.evaluator import read_ground_truth_files

from pathlib import Path
import itertools
from natsort import natsorted

from transformers import BertTokenizer, BertModel
from joblib import Memory

memory = Memory(".cache", verbose=0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@memory.cache
def get_tokenizer_model():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained("bert-base-cased").to(DEVICE)

    return bert_tokenizer, bert_model

# @memory.cache
def read_problem_files(problem_folder, start=0, stop=None):
    # print(f"{start=} {stop=}")
    problems = []
    files = itertools.islice(natsorted(Path(problem_folder).glob('problem-*.txt')), start, stop)
    for problem_file in files:
        # print(f"{problem_file=}")
        # number = problem_file.name[len("problem-") : -len(".txt")]
        with open(problem_file, 'r', encoding="utf8") as fh:
            problems.append(fh.readlines())
    return problems

def pad_paragraph(paragraph_embedding, desired_length):
    d1, d2, d3 = paragraph_embedding.shape
    # print(f"{paragraph_embedding.shape=}")

    target = torch.zeros(d1, desired_length, d3)
    # print(f"{target.shape=}")
    target[:, :d2, :] = paragraph_embedding

    return target

def get_simple_ground_truth(ground_truth, problem_numbers):
    simple_ground_truth = []
    for num in problem_numbers:
        task_3_ground_truth = ground_truth[f"problem-{num}"]["paragraph-authors"]
        simple_ground_truth.append(task_3_ground_truth)
    return simple_ground_truth

def get_task_3_ground_truth(simple_ground_truth):
    # print(f"{len(simple_ground_truth)=}")
    task_gt = []
    for problem in simple_ground_truth:
        problem_gt = []
        for author1, author2 in itertools.combinations(problem, 2):
            problem_gt.append(int(author1 != author2))
        task_gt.append(problem_gt)
    # print(f"{len(task_gt)=}")
    return task_gt

# TODO: Invert the function get_task_3_ground_truth. Our model will output a bunch of binary labels which need to be converted to the task 3 ground truth format
# Ground truth format (gtf): [1, 2, 2, 2, 2, 3, 2, 2]
# Binary labels for comparisons (bl): [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
# Each binary label is the result of comparing two paragraphs. 1 means there was an author change, 0 means there was no author change
# For example, bl[0], is the result of comparing gtf[0]=1 and gtf[1]=2. 1 != 2, therefore bl[0] = 1. bl[1]=1 is the result of gtf[0] == gtf[2] (1 == 2)
def get_simple_ground_truth_from_task_3(task_3_ground_truth):
    simple_gt = []
    for problem in task_3_ground_truth:
        # k = n*(n-1)/2
        # n**2 - n - 2k = 0
        coeff = [1, -1, len(problem) * -2]
        roots = np.roots(coeff)
        gt_length = int(roots[roots > 0][0])
        # print(gt_length)

        gt = np.zeros(gt_length, dtype=np.uint8)
        gt[0] = 1
        for i in range(1, gt_length):
            # loop for gt[i]
            num_comparisons = i
            pointer = i - 1
            modified_flag = False
            # print(f"{i=} {num_comparisons=} {pointer=}")
            for gt_i, j in enumerate(range(gt_length-2, 1, -1)[:num_comparisons]):
                # comparison between gt[gt_i] and gt[i]
                # print(f"{gt_i=} {j=} {pointer=} {task_3_ground_truth[pointer]=}")
                bin_label = task_3_ground_truth[pointer]
                if bin_label == 0:
                    # print(f"{gt[i]=} {gt[gt_i]=}")
                    gt[i] = gt[gt_i]
                    modified_flag = True
                    break

                pointer += j
            if not modified_flag:
                # print(f"No modified")
                gt[i] = np.max(gt) + 1
            # print(f"{gt}\n")
        simple_gt.append(gt)
    return simple_gt

def get_cuda_memory(i):
    if not torch.cuda.is_available():
        return 0

    total_memory = torch.cuda.get_device_properties(i).total_memory
    reserved_memory = torch.cuda.memory_reserved(i)
    allocated_memory = torch.cuda.memory_allocated(i)
    free_memory = total_memory - (reserved_memory + allocated_memory)
    
    return free_memory / (1024 ** 2) # MB

def flatten_problems(problems_list, squeeze=False):
        # [print(f"{pair=}") for problem in problems_list for pair in problem]
        return [pair.squeeze(0) if squeeze else pair for problem in problems_list for pair in problem]

class Pan21PyDataset(PyDataset):
    bert_tokenizer, bert_model = get_tokenizer_model()
    
    def get_problem_embeddings(problems, max_input_length, verbose=False):
        def get_paragraph_pairs(problem_text, max_input_length):
            def get_embeddings(paragraph):
                inputs = Pan21PyDataset.bert_tokenizer(paragraph, return_tensors='pt', padding=True, truncation=True)

                # Move inputs to GPU
                inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

                # Step 4: Generate embeddings
                with torch.no_grad():
                    outputs = Pan21PyDataset.bert_model(**inputs)

                # The last hidden state contains the embeddings for each token
                return outputs.last_hidden_state
            # print(problem_text)
            paragraph_embeddings = [pad_paragraph(get_embeddings(para[:max_input_length]), max_input_length) for para in problem_text]
            # print(f"{[paras.shape for paras in paragraph_embeddings]=}")
            # print(f"{[x.shape for x in paragraph_embeddings]}")
            # print(f"{len(paragraph_embeddings)=}")
            pairs = itertools.combinations(paragraph_embeddings, 2)
            return [torch.flatten(torch.stack(pair, dim=2), start_dim=1, end_dim=2) for pair in pairs]
        if verbose:
            return [get_paragraph_pairs(problem_text, max_input_length) for problem_text in tqdm(problems)]
        else:
            return [get_paragraph_pairs(problem_text, max_input_length) for problem_text in problems]

    def __init__(self, x_path, y_set, file_path, max_input_length=256, batch_size=32,**kwargs):
        super().__init__(**kwargs)
        

        self.x_path, self.y = x_path, read_ground_truth_files(y_set)
        self.max_input_length = max_input_length
        self.batch_size = batch_size

        self.file_path = Path(file_path)
        self.file_path.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        # Return number of batches.
        return math.ceil(len(self.y) / self.batch_size)

    def get_data(x_path, y, low_idx, high_idx, max_input_length):
        # print(f"{self.x_path=}")
        embeddings = Pan21PyDataset.get_problem_embeddings(read_problem_files(x_path, low_idx, high_idx), max_input_length)
        gt = get_task_3_ground_truth(get_simple_ground_truth(y, range(low_idx+1, high_idx+1)))
        
        # print(f"{len(embeddings)=}")
        # print(f"{len(gt)=}")
        # print(f"{[len(embed) for embed in embeddings]=}")
        # print(f"{[len(g) for g in gt]=}")

        # assert np.all([len(e) == len(g) for e, g in zip(embeddings, gt)]), f"{[len(embed) for embed in embeddings]=}\n" + f"{[len(g) for g in gt]=}"

        batch_x = np.array(flatten_problems(embeddings, squeeze=True))
        batch_y = np.array(flatten_problems(gt))
        return batch_x, batch_y

    def to_file_helper(idx_path, i, x_path, y, batch_size, compress=False, max_input_length=256):
        # print(f"{idx_path=}")
        low = i * batch_size
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + batch_size, len(y))
        # print(f"{low=} {high=}")
        batch_x, batch_y = Pan21PyDataset.get_data(x_path, y, low, high, max_input_length)
        # print(f"{batch_x.shape=}")
        if compress:
            np.savez_compressed(idx_path, batch_x=batch_x, batch_y=batch_y)
        else:
            np.savez(idx_path, batch_x=batch_x, batch_y=batch_y)

    def to_file(self, overwrite=False, compress=False):
        free_memory = get_cuda_memory(0)
        n_jobs = min(free_memory // 850, os.cpu_count() - 2)
        # print(f"{n_jobs=} {free_memory=} {os.cpu_count()=}")
        args_for_jobs = []
        for i in range(len(self)):
            idx_path = self.file_path / f"{i}.npz"
            if not idx_path.exists() or overwrite:
                args_for_jobs.append((idx_path, i, self.x_path, self.y, self.batch_size, compress, self.max_input_length))
        
        with parallel_config(backend='threading', n_jobs=n_jobs):
            Parallel()(delayed(Pan21PyDataset.to_file_helper)(*args) for args in tqdm(args_for_jobs))

    def __getitem__(self, idx, force_compute=False):
        idx_path = self.file_path / f"{idx}.npz"
        # print(f"Pan21PyDataset {idx_path=}")
        if force_compute or not idx_path.exists():
            # print(f"{idx=}")
            # Return x, y for batch idx.
            low = idx * self.batch_size
            # Cap upper bound at array length; the last batch may be smaller
            # if the total number of items is not a multiple of batch size.
            high = min(low + self.batch_size, len(self.y))
            # print(f"{low=} {high=}")
            batch_x, batch_y = Pan21PyDataset.get_data(self.x_path, self.y, low, high, self.max_input_length)

            # print(f"{batch_x[0].shape=}")
            # print(f"{len(batch_x)=}")
            # print(f"{len(batch_y)=}")
        else:
            print(f"file exists")
            npzfile = np.load(idx_path)
            batch_x = npzfile['batch_x']
            batch_y = npzfile['batch_y']

        assert batch_x.shape[0] != 0, f"{idx=}: Dimension is 0 {batch_x.shape=} {batch_y.shape=}"
        assert batch_x.shape[0] == batch_y.shape[0], f"{idx=}: Dimension mismatch {batch_x.shape=} {batch_y.shape=}"

        return batch_x, batch_y

class Pan21FourierDataset(Pan21PyDataset):
    def __init__(self, x_set, y_set, file_path, batch_size=32, num_fourier_features=512, **kwargs):
        super().__init__(x_set, y_set, file_path, batch_size=batch_size, **kwargs)
        self.num_fourier_features = num_fourier_features

    def __getitem__(self, idx, force_compute=False):
        idx_path = self.file_path / "fourier" / f"{idx}.npz"
        return Pan21FourierDataset.__getitem__helper(idx_path, idx, num_fourier_features=self.num_fourier_features, force_compute=force_compute)

    def __getitem__helper(idx_path, idx, num_fourier_features, force_compute=False):
        # idx_path = file_path / "fourier" / f"{idx}.npz"
        # print(f"Pan21FourierDataset {idx_path=}")

        # batch_x, batch_y = super().__getitem__(idx, force_compute)
        # print(f'{idx_path.parent / ".." / f"{idx}.npz"=}')
        embed_file = np.load(idx_path.parent / ".." / f"{idx}.npz")
        batch_x = embed_file['batch_x']
        batch_y = embed_file['batch_y']
        
        if num_fourier_features > 0:
            new_batch_x = batch_x.copy()
            
            num_features = len(batch_x[0])
            # print(f"{num_features=}")
            # 0:x will be BERT embeddings for paragraph 1
            # x:length/2 will be fourier features for paragraph 1
            num_non_fourier_features = (num_features - num_fourier_features) // 2
            para1_fourier_features_low, para1_fourier_features_high = num_non_fourier_features, num_features // 2
            para2_fourier_features_low, para2_fourier_features_high = num_features // 2 + num_non_fourier_features , num_features

            # print(f"{para1_fourier_features_low=} {para1_fourier_features_high=}")
            # print(f"{para2_fourier_features_low=} {para2_fourier_features_high=}")

            if force_compute or not idx_path.exists():
                para1_end = num_features//2
                for i, x in enumerate(batch_x):
                    para1_fft = np.real(np.fft.fft(x[:para1_end], axis=1))
                    para2_fft = np.real(np.fft.fft(x[para1_end:], axis=1))

                    new_batch_x[i, para1_fourier_features_low:para1_fourier_features_high] = para1_fft[:num_fourier_features//2]
                    new_batch_x[i, para2_fourier_features_low:para2_fourier_features_high] = para2_fft[:num_fourier_features//2]
            else:
                npzfile = np.load(idx_path)
                fourier_batch_x = npzfile['fourier_batch_x']

                _,j,_ = fourier_batch_x.shape

                new_batch_x[:, para1_fourier_features_low:para1_fourier_features_high, :] = fourier_batch_x[:, :num_fourier_features//2, :]
                new_batch_x[:, para2_fourier_features_low:para2_fourier_features_high, :] = fourier_batch_x[:, j//2:(j+num_fourier_features)//2, :]

            return new_batch_x, batch_y
        else:
            return batch_x, batch_y
    
    def save_file(idx_path, idx, num_fourier_features, force_compute=False, compress=False):
        fourier_batch_x, _ = Pan21FourierDataset.__getitem__helper(idx_path, idx, num_fourier_features, force_compute)
        if compress:
            np.savez_compressed(idx_path, fourier_batch_x=fourier_batch_x)
        else:
            np.savez(idx_path, fourier_batch_x=fourier_batch_x)

    def to_file(self, overwrite=False, compress=False):
        fourier_file_path = self.file_path / "fourier"
        fourier_file_path.mkdir(parents=True, exist_ok=True)

        n_jobs = os.cpu_count() // 2
        # print(f"{n_jobs=} {free_memory=} {os.cpu_count()=}")
        args_for_jobs = []
        for i in range(len(self)):
            idx_path = self.file_path / "fourier" / f"{i}.npz"
            if not idx_path.exists() or overwrite:
                args_for_jobs.append((idx_path, i, 512, True, compress))
        
        with parallel_config(backend='threading', n_jobs=n_jobs):
            Parallel()(delayed(Pan21FourierDataset.save_file)(*args) for args in tqdm(args_for_jobs))

import gc

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()

from scipy.signal import butter, lfilter
# from cupyx.scipy.signal import lfilter

class Pan21FourierFilterDataset(Pan21FourierDataset):
    def __init__(self, x_set, y_set, file_path,  cutoff_frequencies, **kwargs):
        super().__init__(x_set, y_set, file_path, **kwargs)
        
        lowcut, highcut = cutoff_frequencies
        order = 5
        nyq = 512 / 2
        low = lowcut / nyq
        high = highcut / nyq
        
        self.i, self.u = butter(order, [low, high], btype='bandstop')
    
    def __getitem__(self, idx, force_compute=False):
        idx_path = self.file_path / "fourier" / f"{idx}.npz"

        npzfile = np.load(idx_path)
        fourier_batch_x = npzfile['fourier_batch_x']

        # y = np.zeros(fourier_batch_x.shape)
        # for i in range(fourier_batch_x.shape[0]):
        #     y[i] = lfilter(self.i, self.u, fourier_batch_x[i])
        return lfilter(self.i, self.u, fourier_batch_x)