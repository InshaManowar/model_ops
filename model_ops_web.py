import streamlit as st
import pandas as pd
import os
import json

data_directory = "model_data"

json_files = [file for file in os.listdir(data_directory) if file.endswith('.json')]

selected_json_files = st.sidebar.multiselect("Select JSON files to display", json_files)

show_all = st.sidebar.checkbox("Show All JSON Files")

combined_info = []
evaluation_info = []
csv_preview_files = [] 

selected_files = json_files if show_all else selected_json_files

active_row = {"index": -1, "model_description": None}

for idx, file_name in enumerate(selected_files):
    with open(file_name, 'r') as file:
        json_data = json.load(file)

        is_active = st.checkbox(f"Toggle {json_data['model_name']} - {json_data['model_description']}", value=(idx == active_row["index"]))

        if is_active:
            active_row["index"] = idx
            active_row["model_description"] = json_data['model_description']

        combined_info.append({
            "Active": is_active,  
            "Model Description": json_data["model_description"],
            "Model Name": json_data["model_name"],
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
            "Future Plan": json_data["evaluation_results"].get("future_plans", "N/A")
        })
        csv_preview_file = json_data["evaluation_results"]["evaluation_dataset"]
        csv_preview_files.append(csv_preview_file)

st.title("Fine-Tuned Models Information")

st.header("Training Data")
combined_info_df = pd.DataFrame(combined_info)

def highlight_active_row(s):
    return [f"background-color: {'lightblue' if idx == active_row['index'] else ''}" for idx in range(len(s))]

styled_combined_info_df = combined_info_df.style.apply(highlight_active_row, axis=1)
st.dataframe(styled_combined_info_df)

st.header("Evaluation Results")
evaluation_info_df = pd.DataFrame(evaluation_info)
st.dataframe(evaluation_info_df)

st.header("Preview for Evaluation Datasets")
for csv_file in csv_preview_files:
    try:
        csv_preview_df = pd.read_csv(csv_file)
        st.write(f"Preview for {csv_file}")
        st.write(csv_preview_df.head(5))
        if st.checkbox(f"Show More Rows for {csv_file}"):
            st.write(csv_preview_df)
    except FileNotFoundError:
        st.warning(f"CSV file for evaluation dataset '{csv_file}' not found.")

st.sidebar.text(f"Active Model : {active_row['model_description']}")
