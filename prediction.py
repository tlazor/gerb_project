import numpy as np
from rich.progress import track
import torch

from evaluation.evaluator import compute_score_single_predictions, compute_score_multiple_predictions

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

def construct_similarity_matrix(paragraphs, comparisons):
    n = len(paragraphs)
    similarity_matrix = np.zeros((n, n))
    
    for i, j, same_author in comparisons:
        similarity_matrix[i, j] = same_author
        similarity_matrix[j, i] = same_author
    
    np.fill_diagonal(similarity_matrix, 0)
    
    return similarity_matrix

def cluster_paragraphs(similarity_matrix, num_clusters):
    distance_matrix = 1 - similarity_matrix
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    
    Z = linkage(condensed_distance_matrix, method='average')
    
    labels = fcluster(Z, num_clusters, criterion='maxclust')

    if len(np.unique(labels)) == len(labels):
        if num_clusters == 1:
            # weird fcluster bug handling
            labels = [1, 2] if distance_matrix[0][1] < .5 else [1, 1]

    # fcluster makes no guarantees that the cluster indexes will be monotonically increasing
    unique_labels = {}
    new_label = 1
    for i in range(len(labels)):
        if labels[i] not in unique_labels:
            unique_labels[labels[i]] = new_label
            new_label += 1
        labels[i] = unique_labels[labels[i]]
    
    return labels

def determine_optimal_clusters(similarity_matrix, max_clusters=10):
    distance_matrix = 1 - similarity_matrix
    condensed_distance_matrix = squareform(distance_matrix, checks=False)
    
    Z = linkage(condensed_distance_matrix, method='average')
    
    best_num_clusters = 1
    best_silhouette_score = -1
    
    # range should go through min(max clusters, number of labels) 
    # if num labels == num clusters, every point is in its own cluster and silhouette_score is 1
    # 1 is max so "optimal" would be always every label in its own cluster, clearly not desireable
    for num_clusters in range(1, min(max_clusters, similarity_matrix.shape[0]-1) + 1):
        labels = fcluster(Z, num_clusters, criterion='maxclust')
        # silhouette score for single cluster is 0
        if len(np.unique(labels)) == len(labels):
            if num_clusters == 1:
            # TODO:  seems to be a bug in fcluster, num_clusters == 1 should return 1 unique label, but does not always
            # how to handle this case? silhouette score is 1 when every point is its own cluster
                # print(f"{num_clusters=} bug")
                pass
            else:
                print(f"{num_clusters=} {labels=}")
        else:
            score = silhouette_score(distance_matrix, labels, metric='precomputed') if len(np.unique(labels)) != 1 else 0
            
            if score > best_silhouette_score:
                best_silhouette_score = score
                best_num_clusters = num_clusters
            
    return best_num_clusters


def build_prediction_dict(models, val_dataset):
    
    dicts = []
    for i, model in enumerate(models):
        dicts.append({})
        prediction_dict = dicts[i]

        for idx, (x, _) in track(enumerate(val_dataset), total=len(val_dataset)):
            prediction = model(x) 
            problem_num, para_a_num, para_b_num = val_dataset.paragraph_pair_info[idx]

            if problem_num not in prediction_dict:
                prediction_dict[problem_num] = {}
            prediction_dict[problem_num][(para_a_num, para_b_num)] = prediction.item()

    return dicts


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


def author_prediction(prediction_map, num_paragraphs, original=False, threshold=.5):
    if original:
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
    else:
        # empirically determined
        k = 500
        def sigmoid(x):
            return 1 / (1 + np.exp(-k * (x - threshold)))
        def inverse_sigmoid(x):
            # return threshold - (1 / k) * np.log((1 - x) / x)
            return np.abs(np.log(x/(1 - x)))
        
        # the result of the binary comparison is the probability of two paragraphs being written by the same author 
        # (0 => extremely likely, .5 => even chance, 1 => very unlikely)
        # we can use the probabilities as distances and then cluster the binary comparisons to see which ones
        # are the same author
        distance_matrix = np.ones((num_paragraphs, num_paragraphs))
        for (key_i, key_j), binary_comparison in prediction_map.items():
            # use inverse sigmoid to remove last layer
            distance_matrix[key_i, key_j] = inverse_sigmoid(binary_comparison)
            distance_matrix[key_j, key_i] = inverse_sigmoid(binary_comparison)
        
        np.fill_diagonal(distance_matrix, 0)
        similarity_matrix = 1 - distance_matrix

        num_authors = determine_optimal_clusters(similarity_matrix, max_clusters=4)
        authors = cluster_paragraphs(similarity_matrix, num_clusters=num_authors)

    return authors

def change_prediction(authors, num_paragraphs):
    return [int(authors[i] != authors[i + 1]) for i in range(num_paragraphs - 1)]

def map_predictions_to_json(prediction_map, threshold=0.5):
    num_paragraphs = max([x[1] for x in prediction_map.keys()])+1
    
    # Initialize authorship
    authors = author_prediction(prediction_map, num_paragraphs, threshold=threshold)
    changes = change_prediction(authors, num_paragraphs)
    
    # Set of unique authors to count different authors
    num_authors = len(set(authors))
    
    # Construct the JSON object
    data = {
        "authors": num_authors,
        "structure": [999],  # Placeholder or specific requirement
        "site": "googole.com",
        # "multi-author": multiauthor_prediction(prediction_map, threshold),
        "multi-author": 1 if num_authors > 1 else 0,
        "changes": changes,
        "paragraph-authors": authors
    }
    return data


def build_json_predictions(prediction_dict, threshold=.5):
    return {f"problem-{problem_num}":map_predictions_to_json(problem_predictions, threshold) for problem_num, problem_predictions in prediction_dict.items()}


def compute_scores(truth, predictions):
    task1_result = compute_score_single_predictions(truth, predictions, 'multi-author')
    task2_result = compute_score_multiple_predictions(truth, predictions, 'changes', labels=[0, 1])
    task3_result = compute_score_multiple_predictions(truth, predictions, 'paragraph-authors', labels=[1, 2, 3, 4])

    return task1_result, task2_result, task3_result