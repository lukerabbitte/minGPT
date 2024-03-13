import pandas as pd

# Read the data
data = pd.read_csv("../data/goodreads_eval_20pc.tsv", delimiter="\t")

# Count the occurrences of each rating
rating_counts = data['rating'].value_counts()

# Calculate the percentage of each rating category
total_ratings = rating_counts.sum()
percentage_1 = (rating_counts.get(1, 0) / total_ratings) * 100
percentage_2 = (rating_counts.get(2, 0) / total_ratings) * 100
percentage_3 = (rating_counts.get(3, 0) / total_ratings) * 100
percentage_4 = (rating_counts.get(4, 0) / total_ratings) * 100
percentage_5 = (rating_counts.get(5, 0) / total_ratings) * 100

# Print the statistics
print("Percentage of ratings:")
print(f"1: {percentage_1:.2f}%")
print(f"2: {percentage_2:.2f}%")
print(f"3: {percentage_3:.2f}%")
print(f"4: {percentage_4:.2f}%")
print(f"5: {percentage_5:.2f}%")
