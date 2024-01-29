import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def extract_question_type_accuracies(json_line):
    """Extracts question type accuracies from a log line."""
    accuracies = {}
    for key, value in json_line.items():
        if (
            key.startswith("val_")
            and not key.startswith("val_n_")
            and not key.endswith("_acc")
            and not key.endswith("_lr")
            and not key.endswith("_n")
            and not key.endswith("_second")
        ):
            accuracies[key] = value
    return accuracies


def create_plots_for_experiment(experiments_dir,experiment_number):
    log_file_path = os.path.join(experiments_dir,experiment_number, "log.txt")

    if not os.path.exists(log_file_path):
        print(f"Log file not found for experiment {experiment_number}")
        return

    with open(log_file_path, "r") as file:
        log_contents = file.readlines()

    data = []
    question_types = set()
    for line in log_contents:
        try:
            json_line = json.loads(line)
            entry = {
                "epoch": json_line["epoch"],
                "train_loss": json_line.get("train_loss"),
                "val_loss": json_line.get("val_loss"),
                "val_acc_exact_match": json_line.get("val_acc_exact_match"),
            }
            question_accuracies = extract_question_type_accuracies(json_line)
            entry.update(question_accuracies)
            question_types.update(question_accuracies.keys())
            data.append(entry)
        except json.JSONDecodeError:
            continue

    df = pd.DataFrame(data)

    # Check if necessary columns are present
    if not all(
        col in df.columns
        for col in [
            "train_loss",
            "val_loss",
            "val_visual",
            "val_audio",
            "val_both",
            "val_acc_exact_match",
        ]
    ):
        print("Required data not found in logs.")
        # return

    # Plotting Validation Accuracy by Mode
    # sns.set(style="whitegrid")
    # fig, ax = plt.subplots(figsize=(10, 6))
    # sns.lineplot(
    #     ax=ax, x=df["epoch"], y=df["val_visual"], label="Visual", color="purple"
    # )
    # sns.lineplot(ax=ax, x=df["epoch"], y=df["val_audio"], label="Audio", color="orange")
    # sns.lineplot(ax=ax, x=df["epoch"], y=df["val_both"], label="Both", color="green")

    # ax.set_xlabel("Epoch")
    # ax.set_ylabel("Accuracy")
    # ax.set_title(f"Experiment {experiment_number} - Validation Accuracy by Mode")

    # Find best overall accuracy from 'val_acc_exact_match'
    best_val_accuracy = df["val_acc_exact_match"].max()
    best_val_accuracy = round(best_val_accuracy, 4)  # rounding for filename

    # Save the plot
    # output_filename = os.path.join(
    #     experiments_dir, experiment_number, f"{experiment_number}_accuracy_plot_{best_val_accuracy}.png"
    # )
    # fig.savefig(output_filename)
    # print(f"Plot saved as {output_filename}")

    # Plotting Training and Validation Loss & Overall Validation Accuracy
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    sns.lineplot(
        ax=axes[0], x=df["epoch"], y=df["train_loss"], label="Train Loss", color="blue"
    )
    sns.lineplot(
        ax=axes[0],
        x=df["epoch"],
        y=df["val_loss"],
        label="Validation Loss",
        color="red",
    )
    sns.lineplot(
        ax=axes[1],
        x=df["epoch"],
        y=df["val_acc_exact_match"],
        label="Validation Accuracy",
        color="green",
    )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Overall Validation Accuracy")
    axes[1].legend()

    # Save the plot
    output_filename_2 = os.path.join(
        experiments_dir, experiment_number,
        f"{experiment_number}_loss_accuracy_plot_{best_val_accuracy}.png",
    )
    fig.savefig(output_filename_2)
    print(f"Second plot saved as {output_filename_2}")

    # Plotting Accuracies for Different Question Types
    fig, ax = plt.subplots(figsize=(10, 6))
    for question_type in question_types:
        if question_type in df:
            sns.lineplot(ax=ax, x=df["epoch"], y=df[question_type], label=question_type)
    sns.move_legend(ax, "lower right")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Experiment {experiment_number} - Accuracies for Different Question Types"
    )

    output_filename_3 = os.path.join(
        experiments_dir, experiment_number, f"{experiment_number}_question_types_accuracy.png"
    )
    fig.savefig(output_filename_3)
    print(f"Question types accuracy plot saved as {output_filename_3}")


def create_plots_for_multiple_experiments(experiment_numbers):
    for experiment_number in experiment_numbers:
        create_plots_for_experiment("/scratch/project_462000189/ines/Flipped-VQA/checkpoint/nextqa",experiment_number)
        print(f"Completed processing for experiment {experiment_number}")

# Example usage for multiple experiments
experiment_numbers = ["5627272_2", "5627273_3", "5627274_4", "5627275_5", "5627271_6"]  # Replace with actual experiment numbers
create_plots_for_multiple_experiments(experiment_numbers)
