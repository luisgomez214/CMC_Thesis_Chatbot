import pandas as pd
import sqlite3

# Define file paths
csv_file = "merged_theses.csv"
db_file = "theses.db"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(db_file)

# Save the DataFrame to the SQLite database
df.to_sql("theses", conn, if_exists="replace", index=False)

# Close the connection
conn.close()

print("Database created successfully with the 'theses' table.")

