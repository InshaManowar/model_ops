import pandas as pd
import boto3
from io import BytesIO, StringIO
from nltk.translate.bleu_score import sentence_bleu

aws_access_key_id = ''
aws_secret_access_key = ''
s3_bucket_name = "model-evaluation-llama"
s3_object_key = "output_gold_standard.csv"

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

s3 = session.client('s3')

response = s3.get_object(Bucket=s3_bucket_name, Key=s3_object_key)
csv_content = response['Body'].read().decode('utf-8')

csv_file = StringIO(csv_content)

df = pd.read_csv(csv_file)

def calculate_bleu_score(row):
    reference = row['output'].split()
    hypothesis_scores = []
    for i in range(1, n + 1):
        hypothesis = row[f'output_iteration_{i}'].split()
        score = sentence_bleu([reference], hypothesis)
        hypothesis_scores.append(score)
    return hypothesis_scores

n = len([col for col in df.columns if col.startswith('output_iteration_')])

df['bleu_scores'] = df.apply(calculate_bleu_score, axis=1)

# average BLEU score for each row, added to a new column 'average_bleu'
df['average_bleu'] = df['bleu_scores'].apply(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

output_csv_file = 'output_with_bleu_scores.csv'
df.to_csv(output_csv_file, index=False)

s3_output_path = f"{output_csv_file}"

s3.upload_file(output_csv_file, s3_bucket_name, s3_output_path)
