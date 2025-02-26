import pandas as pd

# Load the CSV file
csv_file = "merged_theses.csv"
df = pd.read_csv(csv_file)

# Display column names
print(df.columns)

