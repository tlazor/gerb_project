import argparse
import os
from pathlib import Path
import numpy as np
import json
import tensorflow as tf
from tqdm import tqdm

from evaluation.evaluator import read_ground_truth_files

from pan21_functions import Pan21FourierDataset, Pan21FourierFilterDataset

def map_predictions_to_json(predictions, threshold=0.5):
    # Number of paragraphs inferred from the triangular number formula: n(n-1)/2 = len(predictions)
    # Solving for n gives us n = 3 for len(predictions) = 3

    num_paragraphs = int((1 + np.sqrt(1 + 8 * len(predictions))) / 2)
    
    # Initialize authorship
    authors = [1]*num_paragraphs
    
    # Parse predictions into a matrix (or map) for easier access
    prediction_map = {}
    index = 0
    for i in range(num_paragraphs):
        for j in range(i + 1, num_paragraphs):
            prediction_map[(i, j)] = predictions[index][0]
            index += 1
    # print(f"{predictions=}")
    # print(f"{prediction_map.items()=}")
    # Clustering logic based on predictions
    max_author = 1
    for i in range(1, num_paragraphs):
        author_set = False
        for j in range(0, i):
            if (i, j) in prediction_map:
                if prediction_map[(i, j)] < threshold:
                    # Assign the same author ID to paragraphs i and j if they are likely the same author
                    authors[i] = authors[j]
                    author_set = True
            else:
                if prediction_map[(j, i)] < threshold:
                    # Assign the same author ID to paragraphs i and j if they are likely the same author
                    authors[i] = authors[j]
                    author_set = True
        if not author_set:
            max_author += 1
            authors[i] = max_author
    
    # Set of unique authors to count different authors
    unique_authors = set(authors)
    
    # Construct the JSON object
    data = {
        "authors": len(unique_authors),
        "structure": [999],  # Placeholder or specific requirement
        "site": "googole.com",
        "multi-author": len(unique_authors) > 1,
        "changes": [int(authors[i] != authors[i + 1]) for i in range(num_paragraphs - 1)],
        "paragraph-authors": authors
    }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a model directory.')
    parser.add_argument('--models', type=str, default='/default/path', help='The path to the models folder')
    parser.add_argument('-f', '--fourier', action='store_true', help="Use fourier dataset")
    args = parser.parse_args()

    if not os.path.isdir(args.models):
        print(f"The models folder {args.models} does not exist.")
    else:
        print(f"Processing models in folder: {args.models}")

    model_dir = Path(args.models)
    if args.fourier:
        val_ds = Pan21FourierDataset("pan21/validation", "pan21/validation")
    
    truth_folder = "pan21/validation"
    truth = read_ground_truth_files(truth_folder)

    task1_scores = []
    task2_scores = []
    task3_scores = []
    

    for model_path in tqdm(model_dir.glob("*.keras")):
        dict_of_jsons_result = {}
        
        if args.fourier == False:
            cutoff_frequencies = [float(f) for f in model_path.stem.split('_')]
            val_ds = Pan21FourierFilterDataset("pan21/validation", "pan21/validation", cutoff_frequencies)
        
        nums_of_pars = val_ds.task_3_lens
        ending_indices = np.cumsum(nums_of_pars)
        
        predictions_file = Path(f"{model_path.stem}_predictions.npy")
        if predictions_file.exists():
            predictions = np.load(predictions_file)
        else:
            loaded_model = tf.keras.models.load_model(model_path)
            predictions = loaded_model.predict(val_ds)

            np.save(predictions_file, predictions)

        # print(f"{predictions.shape=}")

        for i, (ending_index, num_of_pair) in enumerate(zip(ending_indices, nums_of_pars)):
            problem_id = i + 1
            prediction = predictions[int(ending_index) - int(num_of_pair):int(ending_index)]
            result_json = map_predictions_to_json(prediction)
            dict_of_jsons_result[f"problem-{problem_id}"] = result_json

        # Save dict_of_jsons_result to a JSON file
        with open(f'{model_path.stem}.json', 'w') as json_file:
            json.dump(dict_of_jsons_result, json_file, indent=4)