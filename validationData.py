import pandas as pd

# Load CSV
df = pd.read_csv("D:\\Fincon\\umeedsey-ml-model\\dataset\\CMW data 5.11.csv")
df.loc[
    df["Per speculum examination"] == "No",
    "Per speculum examination (If Yes, specify findings)",
] = "No"
df.loc[
    df["Per vaginal examination"] == "No",
    "Per vaginal examination (If Yes, specify findings)",
] = "No"

columns_to_fill = [
    "Neonatal Apgar Score:(1 Minute)",
    "Neonatal Apgar Score:(5 Minute)",
]  # replace with your columns

# Fill missing values with -1 for specific columns
df[columns_to_fill] = df[columns_to_fill].fillna(-1)

# # Columns you want to exclude from null checking
exclude_cols = [
    "Mode_of_delivery2",
    "Antenatal_Peripartum_Maternal_Complications",
    "Neonatal__Fetal_Complications",
]  # <-- change these to your columns

# Create a list of columns to check for null
cols_to_check = [c for c in df.columns if c not in exclude_cols]

# Drop rows where ANY of the selected columns have null values
df_cleaned = df.dropna(subset=cols_to_check)
postnatal_syamptos_col = df_cleaned["Postnatal_Symptoms"]
postnatal_examination_col = df_cleaned["Postnatal_Examination"]
cols_to_move = ["Postnatal_Symptoms", "Postnatal_Examination"]
target_pos = 53  # 0-based index; where the first moved column should go

# Remove the columns from original position
for col in cols_to_move:
    df_cleaned.pop(col)

# Insert columns at target position
for i, col in enumerate(cols_to_move):
    df_cleaned.insert(
        target_pos + i, col, df_cleaned[col] if col in df_cleaned else None
    )

# Alternatively, reorder all columns by swapping with existing ones
cols = list(df_cleaned.columns)

# Swap positions with the first two target columns
cols[target_pos], cols[target_pos + 1] = cols_to_move[0], cols_to_move[1]

df_cleaned = df_cleaned[cols]

df_cleaned["Postnatal_Symptoms"] = postnatal_syamptos_col
df_cleaned["Postnatal_Examination"] = postnatal_examination_col

columns_to_drop = ["id", "name", "mr_number"]  # replace with your column names

# Drop the columns
df_cleaned = df_cleaned.drop(columns=columns_to_drop)
df_cleaned.to_csv("CMW data 5.11_cleaned.csv", index=False)

# # Save cleaned CSV
# df_cleaned.to_csv("cleaned_file.csv", index=False)

# print("Rows with null values removed (except excluded columns)!")
