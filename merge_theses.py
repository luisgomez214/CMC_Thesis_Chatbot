import pandas as pd

# Define file paths
file1_path = "cmc_theses_metadata_export_2025_02_05.csv"
file2_path = "CMC Senior Theses Full Text Report 2010-11-24 to 2025-02-05.csv"

# Load the first CSV file (metadata export)
df1 = pd.read_csv(file1_path)

# Load the second CSV file (full text report) with an alternate encoding
df2 = pd.read_csv(file2_path, encoding="ISO-8859-1")

# Rename columns in df2 using the first row, then remove the first row
df2.columns = df2.iloc[0]  
df2 = df2[1:].reset_index(drop=True)

# Standardize column names for merging
df1 = df1.rename(columns={"title": "Title"})
df2 = df2.rename(columns={"Title": "Title"})

# Merge the data on "Title"
merged_df = pd.merge(df1, df2, on="Title", how="inner")

# Save the merged file
merged_df.to_csv("merged_theses.csv", index=False)

print("Merged file saved as merged_theses.csv")

