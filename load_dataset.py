import os
import pandas as pd

def load_combined_dataset(base_dir='datasets'):
    all_dfs = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)

                try:
                    if file.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    elif file.endswith('.xlsx'):
                        df = pd.read_excel(file_path)
                    else:
                        continue

                    # Normalize column names
                    df.columns = df.columns.str.lower()

                    if 'label' in df.columns and 'text' in df.columns:
                        all_dfs.append(df)
                    else:
                        print(f"⚠ Skipped: {file} — missing 'label' or 'text' columns.")
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")

    if not all_dfs:
        raise ValueError("No valid CSV/XLSX files with 'label' and 'text' found.")

    return pd.concat(all_dfs, ignore_index=True)

