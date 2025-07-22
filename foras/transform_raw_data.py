import pandas as pd
import re

path = './outputs/'
filename = 'GRoLTS_new_raw.csv'

# Function to replace sentences containing 'yes' or 'no' with 1 or 0
def replace_yes_no(sentence):
    if isinstance(sentence, str):  # Check if the value is a string
        if re.search(r'\b(yes)\b', sentence, re.IGNORECASE):
            return 1
        elif re.search(r'\b(no)\b', sentence, re.IGNORECASE):
            return 0
    return sentence 

# Load the CSV
df = pd.read_csv('./outputs/' + filename)

# Convert YES/NO to binary
df['binary_answer'] = df['answer'].apply(replace_yes_no)

# Pivot: each paper becomes a row, each question_id a column
pivot_df = df.pivot_table(index='paper_id', 
                          columns='question_id', 
                          values='binary_answer', 
                          aggfunc='first').fillna(0).astype(int)

# Add a score column that sums all binary answers
pivot_df['score'] = pivot_df.sum(axis=1)

pivot_df = pivot_df.reset_index()
pivot_df.to_csv("transformed_output_with_scores_new_grolts.csv", index=False)