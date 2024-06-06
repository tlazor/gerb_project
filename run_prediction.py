import argparse
import os
from pathlib import Path
import numpy as np
import json
import tensorflow as tf

def map_predictions_to_json(predictions, threshold=0.5):
    # Number of paragraphs inferred from the triangular number formula: n(n-1)/2 = len(predictions)
    # Solving for n gives us n = 3 for len(predictions) = 3

    num_paragraphs = int((1 + math.sqrt(1 + 8 * len(predictions))) / 2)
    
    # Initialize authorship
    authors = list(range(1, num_paragraphs + 1))
    
    # Parse predictions into a matrix (or map) for easier access
    prediction_map = {}
    index = 0
    for i in range(1, num_paragraphs):
        for j in range(i + 1, num_paragraphs + 1):
            prediction_map[(i, j)] = predictions[index]
            index += 1
    print(prediction_map)
    # Clustering logic based on predictions
    for i in range(1, num_paragraphs):
        for j in range(i + 1, num_paragraphs + 1):
            if prediction_map[(i, j)] < threshold:
                # Assign the same author ID to paragraphs i and j if they are likely the same author
                authors[j-1] = authors[i-1]
    
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
    args = parser.parse_args()

    if not os.path.isdir(args.models):
        print(f"The models folder {args.models} does not exist.")
    else:
        print(f"Processing models in folder: {args.models}")

    model_dir = Path(args.models)
    val_ds = Pan21PyDataset("pan21/validation", "pan21/validation")
    
    truth_folder = "pan21/validation"
    truth = read_ground_truth_files(truth_folder)

    nums_of_pars = val_ds.task_3_lens
    num_of_pairs = [(pars * (pars - 1)) / 2 for pars in nums_of_pars]
    ending_indices = np.cumsum(nums_of_pars)

    task1_scores = []
    task2_scores = []
    task3_scores = []
    dict_of_jsons_result = {}

    for model_path in model_dir.glob("*.keras"):
        loaded_model = tf.keras.models.load_model(model_path)
        predictions = loaded_model.predict(val_ds)
        for i, (ending_index, num_of_pair) in enumerate(zip(ending_indices, nums_of_pars)):
            problem_id = i + 1
            prediction = predictions[int(ending_index) - int(num_of_pair):int(ending_index)]
            result_json = map_predictions_to_json(prediction)
            dict_of_jsons_result[f"problem-{problem_id}"] = result_json

        # Save dict_of_jsons_result to a JSON file
        with open(f'{model_path.stem}.json', 'w') as json_file:
            json.dump(dict_of_jsons_result, json_file, indent=4)