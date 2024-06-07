import json
import math
from evaluation.evaluator import compute_score_single_predictions, compute_score_multiple_predictions, read_ground_truth_files
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import re
from natsort import natsorted
import argparse
import os

def print_logs(model_dir, isFourier = True):
    log_dfs = []

    # Loop through sorted log files
    for log_path in natsorted(model_dir.glob('*.log')):
        if isFourier:
            num_regex = "ff_\d*(?=_)"
        else:
            num_regex = ".*_.*(?=_)"

        # Extracting the number of features from the filename
        features = re.search(num_regex, log_path.stem)[0]

        # Reading the log file
        log_df = pd.read_csv(log_path, sep=',', engine='python')
        log_df.columns = ["epoch", "auc", "binary_accuracy", "loss", "val_auc", "val_binary_accuracy", "val_loss"]
        log_df["num_features"] = [features] * len(log_df["epoch"])
        log_dfs.append(log_df)

    if len(log_dfs) < 1:
        print(f"No log files found: {model_dir}")
    else:
        # Concatenate all DataFrame into one
        df = pd.concat(log_dfs)

        # Validation metrics to plot
        val_metrics = ["val_auc", "val_binary_accuracy", "val_loss"]

        # Line styles for different Fourier feature counts
        line_styles = ['-', '--', '-.', ':', 'solid', 'dotted']

        # Create a plot for each validation metric
        for metric in val_metrics:
            plt.figure()
            
            # Group by number of Fourier features and plot each group with different line styles
            for (num_features, group_df), line_style in zip(df.groupby("num_features"), line_styles):
                group_df.plot(x="epoch", y=metric, label=f'{num_features}', ax=plt.gca(), linestyle=line_style)

            # Adding labels and title
            plt.xlabel("Epoch")
            plt.ylabel(metric.replace('_', ' '))
            
            plt.gca().set_ylim(0.5, 0.8)
            plt.gca().get_xaxis().set_major_locator(plt.MaxNLocator(integer=True))
            plt.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4)

            # Display the plot
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a json results directory.')
    parser.add_argument('--model', type=str, default='/default/path', help='The path to the models folder')
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        print(f"The models folder {args.model} does not exist.")
    else:
        print(f"Processing models in folder: {args.model}")

    truth = read_ground_truth_files("pan21/validation")

    for model_path in Path(args.model).glob("*.json"):

        with open(model_path, 'r') as json_file:
            dict_of_jsons_result = json.load(json_file)

        task1_result = compute_score_single_predictions(truth, dict_of_jsons_result, 'multi-author')
        task2_result = compute_score_multiple_predictions(truth, dict_of_jsons_result, 'changes', labels=[0, 1])
        task3_result = compute_score_multiple_predictions(truth, dict_of_jsons_result, 'paragraph-authors', labels=[1, 2, 3, 4])

        # Append scores to the lists
        # task1_scores.append(task1_result)
        # task2_scores.append(task2_result)
        # task3_scores.append(task3_result)

        # Create a DataFrame to store the evaluation results
        # results_df = pd.DataFrame({
        #     'Model': model_path.stem,
        #     'Task 1 Score': task1_result,
        #     'Task 2 Score': task2_result,
        #     'Task 3 Score': task3_result
        # })

        print(
            f'Model: {model_path.stem}\n' +
            f'\tTask 1 Score: {task1_result}\n'+
            f'\tTask 2 Score: {task2_result}\n'+
            f'\tTask 3 Score: {task3_result}\n'
        )

        # Display the results DataFrame
        # print(results_df)

    # Fourier
    print_logs(Path(args.model))

    # Filter
    print_logs(Path(args.model), False)
