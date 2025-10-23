import sqlite3
import pandas as pd

csv_file = "merged_theses2.csv"
db_file = "theses2.db"

# Columns we want to keep
columns_to_keep = [
    "Title", "publication_date", "season", "award", "department",
    "second_department", "advisor1", "advisor2", "advisor3", "embargo_date",
    "keywords", "disciplines", "abstract", "author", "author1_institution",
    "author 2", "URL", "First published", "State", "Total"
]

df = pd.read_csv(csv_file)

# Drop all other hidden or extra columns
df = df[columns_to_keep]
df = df.fillna('')

# Create a full_text column for FTS
df["full_text"] = df.astype(str).agg(" ".join, axis=1)

conn = sqlite3.connect(db_file)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS theses")
cur.execute("DROP TABLE IF EXISTS theses_fts")

# Create main table
df.to_sql("theses", conn, if_exists="replace", index=False)

# Create FTS table with only desired columns + full_text
cur.execute("""
CREATE VIRTUAL TABLE theses_fts USING fts5(
    Title,
    publication_date,
    season,
    award,
    department,
    second_department,
    advisor1,
    advisor2,
    advisor3,
    embargo_date,
    keywords,
    disciplines,
    abstract,
    author,
    author1_institution,
    "author 2",
    URL,
    "First published",
    State,
    Total,
    full_text
);
""")

# Prepare quoted column names for SQLite
columns_sql = ",".join([f'"{c}"' for c in columns_to_keep + ["full_text"]])

# Insert data into FTS table
cur.execute(f"""
INSERT INTO theses_fts ({columns_sql})
SELECT {columns_sql} FROM theses;
""")

conn.commit()
conn.close()

print("✅ Database updated with full_text column and searchable FTS index.")

