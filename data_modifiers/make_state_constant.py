import pandas as pd

# Read the TSV file
df = pd.read_csv("../data/goodreads_eval_80pc.tsv", sep="\t")

# Display the first few rows of the dataframe
print(df.head())

# Change all user IDs to 1
df['user_id'] = 1

# Save the modified dataset
df.to_csv("../data/goodreads_eval_80pc_constant_state.tsv", sep="\t", index=False)