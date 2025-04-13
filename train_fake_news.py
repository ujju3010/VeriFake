import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
from preprocessing.text_preprocessing import clean_text

# Function to load and merge all CSV/XLSX files
def load_combined_dataset(base_dir='datasets'):
    all_dfs = []

    # Loop through folders
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
                        continue  # Skip non-data files

                    df.columns = df.columns.str.lower()  # Normalize column names

                    if 'label' in df.columns and 'text' in df.columns:
                        all_dfs.append(df)
                    else:
                        print(f"⚠ Skipped (missing 'label' or 'text'): {file_path}")
                except Exception as e:
                    print(f"❌ Error reading {file_path}: {e}")

    if not all_dfs:
        raise ValueError("No valid CSV/XLSX files with 'label' and 'text' found.")

    return pd.concat(all_dfs, ignore_index=True)

# Load and preprocess dataset
data = load_combined_dataset('datasets')
data['text'] = data['text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text'])
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')

# Save model and vectorizer
os.makedirs("models", exist_ok=True)
pickle.dump(model, open('models/fake_news_model.pkl', 'wb'))
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
