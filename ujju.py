import pandas as pd

# Load the two datasets
fake = pd.read_csv('Fake.csv')
real = pd.read_csv('True.csv')

# Add labels
fake['label'] = 1   # 1 for fake news
real['label'] = 0    # 0 for real news

# Combine them into a single dataset
data = pd.concat([fake, real], ignore_index=True)

# Save the combined dataset
data.to_csv('fake_news.csv', index=False)

print("Dataset combined successfully!")
