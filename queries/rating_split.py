import pandas as pd

# Assuming the dataset is stored in a pandas DataFrame named df
# Load the dataset into a DataFrame (replace df with your DataFrame)
df = pd.read_csv("../goodreads_eval_modified.tsv", sep='\t')

# Percentage of '5' ratings: 65.43898809523809
# Percentage of '1' ratings: 34.561011904761905

# Count the number of '5' and '1' ratings
count_5 = df[df['rating'] == 5]['rating'].count()
count_1 = df[df['rating'] == 1]['rating'].count()

# Calculate the total number of ratings
total_ratings = len(df)

# Calculate the percentage of '5' ratings
percentage_5 = (count_5 / total_ratings) * 100

# Calculate the percentage of '1' ratings
percentage_1 = (count_1 / total_ratings) * 100

# Print the percentages
print("Percentage of '5' ratings:", percentage_5)
print("Percentage of '1' ratings:", percentage_1)
