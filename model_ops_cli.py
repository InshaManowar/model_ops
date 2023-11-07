import argparse
import pandas as pd
import os
import json

def main():
    parser = argparse.ArgumentParser(description="CLI for model ops")

    parser.add_argument("--json-files", nargs='+', help="List of JSON files to process")
    parser.add_argument("--output-csv", help="Output CSV file path")

    args = parser.parse_args()

    if not args.json_files:
        print("Please provide a list of JSON files using the --json-files option.")
        return

    combined_info = []
    evaluation_info = []

    for file_name in args.json_files:
        with open(file_name, 'r') as file:
            json_data = json.load(file)

            combined_info.append({
                "Model Name": json_data["model_name"],
                "Model Type": json_data["model_type"],
                "Fine-Tuned Goal": json_data["finetuned_goal"],
                "Date Fine-Tuned": json_data["date_fine_tuned"],
                "Dataset Description": json_data["dataset_description"],
                "Success Criteria": json_data["success_criteria"],
                "Model Architecture": json_data["model_architecture"],
                "Learning Rate": json_data["hyperparameters"]["learning_rate"],
                "Batch Size": json_data["hyperparameters"]["batch_size"],
                "Num Epochs": json_data["hyperparameters"]["num_epochs"],
                "Training Dataset": json_data["fine_tuning_details"]["training_dataset"],
                "Fine-Tuning Script": json_data["fine_tuning_details"]["fine_tuning_script"],
                "Training Duration (hours)": json_data["fine_tuning_details"]["training_duration_hours"],
                "Hardware Used": json_data["fine_tuning_details"]["hardware_used"],
                "WandB Link": json_data["fine_tuning_details"]["WandB_link"]
            })

            evaluation_info.append({
                "Model Name": json_data["model_name"],
                "Evaluation Dataset": json_data["evaluation_results"]["evaluation_dataset"],
                "Accuracy": json_data["evaluation_results"]["metric_scores"]["accuracy"],
                "Precision": json_data["evaluation_results"]["metric_scores"]["precision"],
                "Recall": json_data["evaluation_results"]["metric_scores"]["recall"],
                "Other Observations": json_data["evaluation_results"].get("other_observations", "N/A"),
            })

    combined_info_df = pd.DataFrame(combined_info)
    evaluation_info_df = pd.DataFrame(evaluation_info)

    if args.output_csv:
        combined_info_df.to_csv(args.output_csv, index=False)
        print(f"Combined information saved to {args.output_csv}")
    else:
        print("Combined Information:")
        print(combined_info_df)

        print("\nEvaluation Results:")
        print(evaluation_info_df)

if __name__ == "__main__":
    main()
