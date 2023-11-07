from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from transformers import pipeline
import boto3

aws_access_key_id = ''
aws_secret_access_key = ''
s3_bucket_name = 'model-evaluation-llama'  
s3_output_path = 'output_gold_standard.csv'  


s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)

model_path = "lewtun/dummy-trl-model"  ## replace with path of the model on the server

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def generate_output(instruction, input_text, num_iterations=4):
    gen = pipeline('text-generation', model=model, tokenizer=tokenizer)
    outputs = []

    for _ in range(num_iterations):
        result = gen(f"{instruction} :: {input_text}")
        generated_text = result[0]['generated_text'].strip()
        outputs.append(generated_text)

    return outputs

csv_file_path = "gold_std.csv"

df = pd.read_csv(csv_file_path)

for i in range(1, 3):
    df[f'output_iteration_{i}'] = df.apply(lambda row: generate_output(row['instruction'], row['input']), axis=1)

for i in range(1, 3):
    df[f'output_iteration_{i}'] = df[f'output_iteration_{i}'].apply(lambda texts: texts[i-1])  

output_csv_file = "output_gold_standard.csv"
df.to_csv(output_csv_file, index=False)

s3_client.upload_file(output_csv_file, s3_bucket_name, s3_output_path)
