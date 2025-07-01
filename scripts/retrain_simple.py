#!/usr/bin/env python3
"""
Simple improved sentiment analysis model training script
Focuses on fixing positive bias with minimal dependencies
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re

print("TensorFlow version:", tf.__version__)

# Load the datasets
print("Loading datasets...")
imdb_df = pd.read_csv('data/imdb_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])
yelp_df = pd.read_csv('data/yelp_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])
amazon_df = pd.read_csv('data/amazon_cells_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])

# Combine the datasets
combined_df = pd.concat([imdb_df, yelp_df, amazon_df], ignore_index=True)

print(f"Total samples: {len(combined_df)}")
print(f"Sentiment distribution:\n{combined_df['sentiment'].value_counts()}")

# Improved text preprocessing - preserve negation context
def preprocess_text_improved(text):
    """Better preprocessing that handles negation"""
    text = text.lower()
    
    # Handle negations - mark words after "not", "no", "never"
    negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor', "n't", "dont", "wont", "cant", "shouldnt", "wouldnt"]
    
    # Simple negation handling
    words = text.split()
    processed_words = []
    negate_next = False
    
    for word in words:
        # Clean word
        clean_word = re.sub(r'[^a-zA-Z]', '', word)
        if not clean_word:
            continue
            
        # Check for negation
        if any(neg in word for neg in negation_words):
            processed_words.append(clean_word)
            negate_next = True
        elif negate_next and len(clean_word) > 2:
            processed_words.append(f"NOT_{clean_word}")
            negate_next = False
        else:
            processed_words.append(clean_word)
    
    return ' '.join(processed_words)

# Apply improved preprocessing
print("Applying improved preprocessing...")
combined_df['processed_review'] = combined_df['review'].apply(preprocess_text_improved)

# Show some examples
print("\nPreprocessing examples:")
for i in range(3):
    print(f"Original: {combined_df['review'].iloc[i]}")
    print(f"Processed: {combined_df['processed_review'].iloc[i]}")
    print(f"Sentiment: {combined_df['sentiment'].iloc[i]}")
    print("-" * 50)

# Tokenization with larger vocabulary
print("Creating tokenizer...")
tokenizer = Tokenizer(num_words=6000, oov_token='<OOV>')
tokenizer.fit_on_texts(combined_df['processed_review'])

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(combined_df['processed_review'])

# Find optimal max length
seq_lengths = [len(seq) for seq in sequences]
max_length = int(np.percentile(seq_lengths, 90))  # Use 90th percentile
print(f"Using max_length: {max_length}")
print(f"Average sequence length: {np.mean(seq_lengths):.1f}")

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare data
X = padded_sequences
y = combined_df['sentiment'].values

# Simple train/test split (80/20)
split_idx = int(0.8 * len(X))
indices = np.random.permutation(len(X))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training sentiment distribution: {np.bincount(y_train)}")
print(f"Test sentiment distribution: {np.bincount(y_test)}")

# Build improved model
print("Building improved model...")
model = Sequential([
    Embedding(input_dim=6000, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Train the model
print("Training improved model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,  # Reduced for faster training
    validation_split=0.2,
    verbose=1
)

# Evaluate on test set
print("Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Test predictions on sample data
print("\nTesting predictions on sample texts:")
test_texts = [
    "This movie was great and amazing",
    "This movie was terrible and awful", 
    "This movie was not good",
    "This movie was not bad",
    "terrible awful horrible",
    "good great excellent"
]

for text in test_texts:
    processed = preprocess_text_improved(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model.predict(padded, verbose=0)[0][0]
    sentiment = "positive" if pred > 0.5 else "negative"
    print(f"'{text}' -> {pred:.3f} ({sentiment})")

# Test different thresholds
print("\nTesting different thresholds:")
test_predictions = model.predict(X_test, verbose=0).flatten()

for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred_binary = (test_predictions > threshold).astype(int)
    accuracy = np.mean(pred_binary == y_test)
    pos_ratio = np.mean(pred_binary)
    print(f"Threshold {threshold}: Accuracy={accuracy:.3f}, Positive Ratio={pos_ratio:.3f}")

# Save the improved model
print("Saving improved model...")
model.save('models/sentiment_analysis_model_v2.h5')

# Save the improved tokenizer
with open('models/tokenizer_v2.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save max_length
with open('models/max_length_v2.txt', 'w') as f:
    f.write(str(max_length))

print("Improved model training completed!")
print(f"Model saved as: sentiment_analysis_model_v2.h5")
print(f"Test accuracy: {test_accuracy:.4f}")
