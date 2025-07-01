#!/usr/bin/env python3
"""
Improved sentiment analysis model training script
Addresses positive bias and improves overall performance
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the datasets
print("Loading datasets...")
imdb_df = pd.read_csv('data/imdb_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])
yelp_df = pd.read_csv('data/yelp_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])
amazon_df = pd.read_csv('data/amazon_cells_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])

# Combine the datasets
combined_df = pd.concat([imdb_df, yelp_df, amazon_df], ignore_index=True)

print(f"Total samples: {len(combined_df)}")
print(f"Sentiment distribution:\n{combined_df['sentiment'].value_counts()}")

# Improved text preprocessing
def preprocess_text_improved(text):
    """
    Improved preprocessing that preserves important negation context
    """
    # Convert to lowercase
    text = text.lower()
    
    # Handle negations better - preserve "not", "no", "never", etc.
    negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'nor']
    
    # Remove special characters but keep some punctuation for context
    text = re.sub(r'[^a-zA-Z\s!?.]', '', text)
    
    # Split into words
    words = text.split()
    
    # Simple negation handling - mark words after negation
    processed_words = []
    negate_next = False
    
    for word in words:
        if word in negation_words:
            processed_words.append(word)
            negate_next = True
        elif negate_next and word not in ['!', '?', '.']:
            processed_words.append(f"NOT_{word}")
            negate_next = False
        else:
            processed_words.append(word)
            if word in ['!', '?', '.']:
                negate_next = False
    
    return ' '.join(processed_words)

# Apply improved preprocessing
print("Applying improved preprocessing...")
combined_df['processed_review'] = combined_df['review'].apply(preprocess_text_improved)

# Tokenization with larger vocabulary
print("Creating tokenizer...")
tokenizer = Tokenizer(num_words=8000, oov_token='<OOV>')  # Increased vocab size
tokenizer.fit_on_texts(combined_df['processed_review'])

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(combined_df['processed_review'])

# Find optimal max length (95th percentile)
seq_lengths = [len(seq) for seq in sequences]
max_length = int(np.percentile(seq_lengths, 95))
print(f"Using max_length: {max_length}")

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare data
X = padded_sequences
y = combined_df['sentiment'].values

# Split data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")

# Check class balance
print(f"Training sentiment distribution: {np.bincount(y_train)}")
print(f"Validation sentiment distribution: {np.bincount(y_val)}")
print(f"Test sentiment distribution: {np.bincount(y_test)}")

# Build improved model
print("Building improved model...")
model = Sequential([
    Embedding(input_dim=8000, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(64, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)),
    Bidirectional(LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile with class weights to handle any remaining imbalance
class_weight = {0: 1.0, 1: 1.0}  # Equal weights since data is balanced

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Model architecture:")
model.summary()

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=0.0001,
        verbose=1
    )
]

# Train the model
print("Training improved model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Evaluate on test set
print("Evaluating on test set...")
test_predictions = model.predict(X_test)
test_predictions_binary = (test_predictions > 0.5).astype(int).flatten()

test_accuracy = accuracy_score(y_test, test_predictions_binary)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, test_predictions_binary, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, test_predictions_binary)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix - Improved Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('improved_training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Test bias by checking predictions on different thresholds
print("\nBias Analysis:")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for threshold in thresholds:
    pred_binary = (test_predictions > threshold).astype(int).flatten()
    acc = accuracy_score(y_test, pred_binary)
    pos_ratio = np.mean(pred_binary)
    print(f"Threshold {threshold}: Accuracy={acc:.3f}, Positive Ratio={pos_ratio:.3f}")

# Save the improved model
print("Saving improved model...")
model.save('models/sentiment_analysis_model_improved.h5')

# Save the improved tokenizer
with open('models/tokenizer_improved.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save max_length
with open('models/max_length_improved.txt', 'w') as f:
    f.write(str(max_length))

print("Improved model training completed!")
print(f"Model saved as: sentiment_analysis_model_improved.h5")
print(f"Test accuracy: {test_accuracy:.4f}")
