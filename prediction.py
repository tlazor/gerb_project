import numpy as np
from rich.progress import track

from evaluation.evaluator import compute_score_single_predictions, compute_score_multiple_predictions

def build_prediction_dict(model, val_dataset):
    prediction_dict = {}
    for idx, (x, _) in track(enumerate(val_dataset), total=len(val_dataset)):
        prediction = model(x) 
        problem_num, para_a_num, para_b_num = val_dataset.paragraph_pair_info[idx]

        if problem_num not in prediction_dict:
            prediction_dict[problem_num] = {}
        prediction_dict[problem_num][(para_a_num, para_b_num)] = prediction.item()

    return prediction_dict


# not sure if actually better
def multiauthor_prediction(prediction_map, threshold=0.5):
    k = 20  # Adjust to control steepness of sigmoid
    # for the predictions that think there are different authors (x > threshold), how certain are we
    weights = [1 / (1 + np.exp(-1 * k * (x - threshold))) for x in prediction_map.values() if x > threshold]
    # differences = [x > threshold for x in prediction_map.values()]
    if len(weights) > 0:
        # print(f"{np.average(weights)=}")
        return 1 if np.average(weights) > .60 else 0
    else:
        return 0
    

def map_predictions_to_json(prediction_map, threshold=0.5):
    num_paragraphs = max([x[1] for x in prediction_map.keys()])+1
    
    # Initialize authorship
    authors = [1]*num_paragraphs
    
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
        "multi-author": multiauthor_prediction(prediction_map, threshold),
        "changes": [int(authors[i] != authors[i + 1]) for i in range(num_paragraphs - 1)],
        "paragraph-authors": authors
    }
    return data


def build_json_predictions(prediction_dict, threshold=.5):
    return {f"problem-{problem_num}":map_predictions_to_json(prediction_map, threshold) for problem_num, prediction_map in prediction_dict.items()}


def compute_scores(truth, predictions):
    task1_result = compute_score_single_predictions(truth, predictions, 'multi-author')
    task2_result = compute_score_multiple_predictions(truth, predictions, 'changes', labels=[0, 1])
    task3_result = compute_score_multiple_predictions(truth, predictions, 'paragraph-authors', labels=[1, 2, 3, 4])

    return task1_result, task2_result, task3_result