!pip install pandas numpy scikit-learn tqdm

import pandas as pd
import numpy as np
import pickle
import re
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("=" * 60)
print("DATA PREPROCESSING PIPELINE - WELFAKE (RESEARCH-GRADE)")
print("=" * 60)

try:
    import google.colab
    OUTPUT_DIR = '/content'
    print("Detected: Google Colab environment")
except:
    OUTPUT_DIR = 'processed_data'
    print("Detected: Local environment")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory: {OUTPUT_DIR}")

print("\n[1/5] Loading WELFake dataset...")

try:
    df_welfake = pd.read_csv('WELFake_Dataset.csv')
    print(f"  ✓ WELFake loaded: {len(df_welfake)} samples")
    print(f"  Columns found: {df_welfake.columns.tolist()}")
except Exception as e:
    print(f"  ✗ Error loading WELFake: {e}")
    print("  Make sure 'WELFake_Dataset.csv' is in the current directory")
    exit(1)

print("\n[2/5] Defining cleaning functions...")

EMOJI_PATTERN = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]'

class LabelStandardizer:

    def __init__(self):
        self.unrecognized_labels = set()

    def standardize(self, label):

        if pd.isna(label):
            return None

        if isinstance(label, (int, np.integer)):
            if label == 0:
                return 0
            elif label == 1:
                return 1
            else:
                self.unrecognized_labels.add(str(label))
                return None

        label_orig = str(label)
        label = label_orig.lower().strip()

        fake_labels = ['fake', '0', 'false', 'f', 'unreliable', 'fake news',
                       'fakenews', 'not reliable', 'untrustworthy']
        real_labels = ['real', '1', 'true', 't', 'reliable', 'real news',
                       'realnews', 'trustworthy', 'verified']

        if label in fake_labels:
            return 0
        elif label in real_labels:
            return 1
        else:
            self.unrecognized_labels.add(label_orig)
            return None

def clean_text(text):

    if pd.isna(text) or not isinstance(text, str):
        return ""

    text = str(text).strip()
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    text = re.sub(r'[\x00-\x1f\x7f]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

label_standardizer = LabelStandardizer()
print("✓ Cleaning functions defined")

print("\n[3/5] Processing WELFake dataset...")

text_col = None
label_col = None
title_col = None

if 'text' in df_welfake.columns:
    text_col = 'text'
if 'title' in df_welfake.columns:
    title_col = 'title'
if 'label' in df_welfake.columns:
    label_col = 'label'

if text_col is None:
    text_candidates = [col for col in df_welfake.columns
                      if col.lower() in ['text', 'content', 'article', 'body', 'news']]
    if text_candidates:
        text_col = text_candidates[0]
        if len(text_candidates) > 1:
            print(f"  ⚠ Multiple text columns found: {text_candidates}")
            print(f"  Using: {text_col}")

if label_col is None:
    label_candidates = [col for col in df_welfake.columns
                       if col.lower() in ['label', 'class', 'category', 'type']]
    if label_candidates:
        label_col = label_candidates[0]
    else:
        for col in df_welfake.columns:
            if df_welfake[col].nunique() == 2:
                label_col = col
                break

print(f"  Detected text column: {text_col}")
print(f"  Detected title column: {title_col}")
print(f"  Detected label column: {label_col}")

if text_col is None or label_col is None:
    print("✗ Could not identify text or label columns!")
    print("Available columns:", df_welfake.columns.tolist())
    exit(1)

if title_col and title_col in df_welfake.columns:
    non_empty_titles = df_welfake[title_col].notna().sum()
    title_coverage = non_empty_titles / len(df_welfake) * 100

    if non_empty_titles > len(df_welfake) * 0.05:
        print(f"  Combining title and text ({non_empty_titles} titles, {title_coverage:.1f}% coverage)...")
        df_processed = pd.DataFrame({
            'text': df_welfake[title_col].fillna('') + ' ' + df_welfake[text_col].fillna(''),
            'label': df_welfake[label_col],
            'source': 'welfake'
        })
    else:
        print(f"  Title coverage too low ({title_coverage:.1f}%), using text only...")
        df_processed = pd.DataFrame({
            'text': df_welfake[text_col],
            'label': df_welfake[label_col],
            'source': 'welfake'
        })
else:
    print("  Using text column only...")
    df_processed = pd.DataFrame({
        'text': df_welfake[text_col],
        'label': df_welfake[label_col],
        'source': 'welfake'
    })

print("  Cleaning text...")
df_processed['cleaned_text'] = df_processed['text'].apply(clean_text)

print("  Standardizing labels...")
original_labels = df_processed['label'].value_counts()
print(f"  Original label distribution: {original_labels.to_dict()}")
df_processed['label'] = df_processed['label'].apply(label_standardizer.standardize)

if label_standardizer.unrecognized_labels:
    print(f"  ⚠ Unrecognized label(s): {label_standardizer.unrecognized_labels}")
    print(f"    These {len(label_standardizer.unrecognized_labels)} label(s) will be dropped")

initial_count = len(df_processed)

before_null = len(df_processed)
df_processed = df_processed.dropna(subset=['label'])
null_removed = before_null - len(df_processed)

before_empty = len(df_processed)
df_processed = df_processed[df_processed['cleaned_text'].str.strip() != '']
empty_removed = before_empty - len(df_processed)

before_short = len(df_processed)
df_processed = df_processed[df_processed['cleaned_text'].str.len() > 20]
short_removed = before_short - len(df_processed)

print(f"  Removed invalid data:")
print(f"    - {null_removed} rows with null/unrecognized labels")
print(f"    - {empty_removed} rows with empty text")
print(f"    - {short_removed} rows with text ≤20 characters")

unique_labels = df_processed['label'].unique()
if len(unique_labels) < 2:
    print(f"✗ Error: Only one label class found: {unique_labels}")
    exit(1)

print("  Handling duplicates and conflicting labels...")
before_conflicts = len(df_processed)

text_label_counts = df_processed.groupby('cleaned_text')['label'].nunique()
conflicting_texts = text_label_counts[text_label_counts > 1].index

if len(conflicting_texts) > 0:
    print(f"    ⚠ Found {len(conflicting_texts)} texts with conflicting labels - removing all instances...")
    df_processed = df_processed[~df_processed['cleaned_text'].isin(conflicting_texts)]
    conflicts_removed = before_conflicts - len(df_processed)
else:
    conflicts_removed = 0

before_dedup = len(df_processed)
df_processed = df_processed.drop_duplicates(subset=['cleaned_text'], keep='first')
duplicates_removed = before_dedup - len(df_processed)

print(f"\n  ✓ Processed: {len(df_processed)} valid samples")

if len(df_processed) == 0:
    print("\n✗ ERROR: No data remaining after filtering!")
    exit(1)

fake_count = int((df_processed['label'] == 0).sum())
real_count = int((df_processed['label'] == 1).sum())

print(f"  Label distribution:")
print(f"    - Fake: {fake_count}")
print(f"    - Real: {real_count}")

text_lengths = df_processed['cleaned_text'].str.len()
word_counts = df_processed['cleaned_text'].str.split().str.len()

emoji_count = df_processed['cleaned_text'].str.contains(EMOJI_PATTERN, regex=True).sum()
hashtag_count = df_processed['cleaned_text'].str.contains('#').sum()
dollar_count = df_processed['cleaned_text'].str.contains(r'\$').sum()

df_processed = df_processed[['cleaned_text', 'label', 'source']]
print(f"\n  ✓ Kept only necessary columns: {df_processed.columns.tolist()}")

print("\n[4/5] Creating train/val/test splits...")

df_processed = df_processed.sample(frac=1, random_state=42).reset_index(drop=True)

train_df, temp_df = train_test_split(
        df_processed,
        test_size=0.3,
        random_state=42,
        stratify=df_processed['label']
)

val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']
)

print(f"\n✓ Data split complete:")
print(f"  Train: {len(train_df)}")
print(f"  Val:   {len(val_df)}")
print(f"  Test:  {len(test_df)}")

print("\n[5/5] Saving processed data...")

with open(f'{OUTPUT_DIR}/train.pkl', 'wb') as f:
    pickle.dump(train_df, f)

with open(f'{OUTPUT_DIR}/val.pkl', 'wb') as f:
    pickle.dump(val_df, f)

with open(f'{OUTPUT_DIR}/test.pkl', 'wb') as f:
    pickle.dump(test_df, f)

with open(f'{OUTPUT_DIR}/combined_data.pkl', 'wb') as f:
    pickle.dump(df_processed, f)

metadata = {
    'total_samples': len(df_processed),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df),
    'fake_count': fake_count,
    'real_count': real_count
}

with open(f'{OUTPUT_DIR}/metadata.pkl', 'wb') as f:
    pickle.dump(metadata, f)

print("\n" + "=" * 60)
print("SAMPLE DATA PREVIEW")
print("=" * 60)

for idx in range(min(3, len(train_df))):
    row = train_df.iloc[idx]
    label_text = "REAL" if row['label'] == 1 else "FAKE"
    text_preview = row['cleaned_text'][:200]
    print(f"\n[{idx+1}] Label: {label_text}")
    print(f"    {text_preview}")

print("\n" + "=" * 60)
print("RESEARCH-GRADE PREPROCESSING COMPLETE")
print("=" * 60)
