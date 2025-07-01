# Sentiment Analysis Model Training Script
# This script performs data preprocessing, model training, and evaluation

import pandas as pd
import numpy as np
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle  # Added for saving tokenizer
import os

# MLflow imports
import mlflow
import mlflow.tensorflow

# Set up MLflow tracking
mlflow.set_tracking_uri(f"sqlite:///{os.getcwd()}/mlflow.db")
mlflow.set_experiment("sentiment-analysis")

# Start MLflow run
mlflow.start_run()

# Load the datasets
imdb_df = pd.read_csv('imdb_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])
yelp_df = pd.read_csv('yelp_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])
amazon_df = pd.read_csv('amazon_cells_labelled.txt', delimiter='\t', header=None, names=['review', 'sentiment'])

# Combine the datasets
combined_df = pd.concat([imdb_df, yelp_df, amazon_df], ignore_index=True)

# Get all unique characters from the reviews
all_chars = set(''.join(combined_df['review']))

# Define the set of allowed characters (26 letters)
allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Find characters that are not in the allowed set
unusual_chars = all_chars - allowed_chars

print("Unusual characters found:")
print(sorted(unusual_chars))

def get_words(text):
    return re.findall(r'\w+', text.lower())

all_words = [word for review in combined_df['review'] for word in get_words(review)]
vocabulary = set(all_words)

print(f"Vocabulary size: {len(vocabulary)}")

# Log dataset parameters to MLflow
mlflow.log_param("total_samples", len(combined_df))
mlflow.log_param("vocabulary_size", len(vocabulary))
mlflow.log_param("imdb_samples", len(imdb_df))
mlflow.log_param("yelp_samples", len(yelp_df))
mlflow.log_param("amazon_samples", len(amazon_df))

embedding_dim = int(len(vocabulary) ** 0.25)
print(f"Suggested word embedding length: {embedding_dim}")

# Log embedding dimension
mlflow.log_param("embedding_dim", embedding_dim)

review_lengths = combined_df['review'].apply(lambda x: len(get_words(x)))

print(f"Mean review length: {review_lengths.mean():.2f}")
print(f"Median review length: {review_lengths.median():.2f}")
print(f"95th percentile of review length: {review_lengths.quantile(0.95):.2f}")
print(f"99th percentile of review length: {review_lengths.quantile(0.99):.2f}")

# Text preprocessing and tokenization

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def expand_contractions(text):
    # Dictionary of common contractions
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "i'd": "I would",
        "i'll": "I will",
        "i'm": "I am",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "mightn't": "might not",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "we'd": "we would",
        "we're": "we are",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who would",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }
    
    # Tokenize the text
    words = text.split()
    
    # Replace contractions with their expanded forms
    expanded_words = [contractions_dict.get(word.lower(), word) for word in words]
    
    return ' '.join(expanded_words)

def normalize_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Expand contractions
    text = expand_contractions(text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

# Apply normalization to the dataset
combined_df['normalized_tokens'] = combined_df['review'].apply(normalize_text)

# Create a vocabulary and convert words to integers
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(combined_df['normalized_tokens'].apply(' '.join))

# Convert tokens to sequences of integers
combined_df['integer_tokens'] = combined_df['normalized_tokens'].apply(lambda x: tokenizer.texts_to_sequences([' '.join(x)])[0])

# Sequence padding to standardize input length

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
print(f"Vocabulary size: {vocab_size}")

# Calculate max_length (use the 95th percentile or a fixed value, whichever is smaller)
max_length = min(100, int(combined_df['integer_tokens'].apply(len).quantile(0.95)))
print(f"Max sequence length: {max_length}")

# Log sequence parameters to MLflow
mlflow.log_param("max_sequence_length", max_length)
mlflow.log_param("padding_strategy", "post")
mlflow.log_param("truncating_strategy", "post")

# Apply padding to all sequences in our dataset
combined_df['padded_sequence'] = keras.preprocessing.sequence.pad_sequences(combined_df['integer_tokens'].tolist(), 
                                               maxlen=max_length, 
                                               padding='post', 
                                               truncating='post').tolist()

# Example of a single padded sequence
example_sequence = combined_df['padded_sequence'][0]
print("Example padded sequence:")
print(example_sequence)

# Print the words corresponding to the example sequence
example_words = [tokenizer.index_word.get(i, '') for i in example_sequence if i != 0]
print("Corresponding words:")
print(example_words)

# Data splitting into training, validation, and test sets

# Shuffle the dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate features and labels
X = np.array(combined_df['padded_sequence'].tolist())
y = np.array(combined_df['sentiment'])

# First, split off the test set (15% of the data)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Then, split the remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)  # 0.1765 of 85% is 15% of the total

# Log data split parameters to MLflow
mlflow.log_param("train_samples", len(X_train))
mlflow.log_param("validation_samples", len(X_val))
mlflow.log_param("test_samples", len(X_test))
mlflow.log_param("train_split_ratio", 0.70)
mlflow.log_param("validation_split_ratio", 0.15)
mlflow.log_param("test_split_ratio", 0.15)
mlflow.log_param("random_state", 42)

# Save prepared datasets

# Save the datasets
combined_df.to_csv('combined_dataset.csv', index=False)

# Saving training set
train_df = pd.DataFrame(X_train)
train_df['sentiment'] = y_train
train_df.to_csv('train_data.csv', index=False)

# Saving validation set
val_df = pd.DataFrame(X_val)
val_df['sentiment'] = y_val
val_df.to_csv('validation_data.csv', index=False)

# Saving test set
test_df = pd.DataFrame(X_test)
test_df['sentiment'] = y_test
test_df.to_csv('test_data.csv', index=False)

print("Datasets saved as CSV files successfully.")

# Model architecture definition

embedding_dim = int(vocab_size ** 0.25)
max_length = X_train.shape[1]  # Should be the same as the max_length we used for padding

print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")
print(f"Max sequence length: {max_length}")

# Log model architecture parameters to MLflow
mlflow.log_param("model_type", "LSTM")
mlflow.log_param("vocab_size", vocab_size)
mlflow.log_param("embedding_dim", embedding_dim)
mlflow.log_param("lstm_units", 128)
mlflow.log_param("dense_units", 64)
mlflow.log_param("dropout_rate", 0.5)
mlflow.log_param("activation_dense", "relu")
mlflow.log_param("activation_output", "sigmoid")
mlflow.log_param("optimizer", "adam")
mlflow.log_param("loss_function", "binary_crossentropy")

# Create the model
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.LSTM(128, return_sequences=False),  # or use GRU(128)
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.build((None, max_length))  # Specify the input shape
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Model training with early stopping

# Define the maximum number of epochs
max_epochs = 100

# Log training parameters to MLflow
mlflow.log_param("max_epochs", max_epochs)
mlflow.log_param("batch_size", 32)
mlflow.log_param("early_stopping_patience", 5)
mlflow.log_param("early_stopping_monitor", "val_loss")
mlflow.log_param("restore_best_weights", True)

# Define early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor
    patience=5,          # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=max_epochs,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Print the final epoch
final_epoch = len(history.history['loss'])
print(f"Training stopped at epoch {final_epoch}")

# Print final training and validation metrics
final_train_loss = history.history['loss'][-1]
final_train_accuracy = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_accuracy = history.history['val_accuracy'][-1]

print(f"Final training loss: {final_train_loss:.4f}")
print(f"Final training accuracy: {final_train_accuracy:.4f}")
print(f"Final validation loss: {final_val_loss:.4f}")
print(f"Final validation accuracy: {final_val_accuracy:.4f}")

# Log training results to MLflow
mlflow.log_metric("epochs_trained", final_epoch)
mlflow.log_metric("final_train_loss", final_train_loss)
mlflow.log_metric("final_train_accuracy", final_train_accuracy)
mlflow.log_metric("final_val_loss", final_val_loss)
mlflow.log_metric("final_val_accuracy", final_val_accuracy)

# Log all epoch metrics to MLflow
for epoch in range(len(history.history['loss'])):
    mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
    mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)

# Training visualization

def plot_training_history(history):
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy values
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='lower right')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper right')

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Model evaluation and testing

# Generate predictions
y_pred_proba = model.predict(X_test)

# Convert probabilities to binary classes
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Log test accuracy to MLflow
mlflow.log_metric("test_accuracy", test_accuracy)

# Print classification report
classification_rep = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Log classification metrics to MLflow
mlflow.log_metric("precision_negative", classification_rep['0']['precision'])
mlflow.log_metric("recall_negative", classification_rep['0']['recall'])
mlflow.log_metric("f1_negative", classification_rep['0']['f1-score'])
mlflow.log_metric("precision_positive", classification_rep['1']['precision'])
mlflow.log_metric("recall_positive", classification_rep['1']['recall'])
mlflow.log_metric("f1_positive", classification_rep['1']['f1-score'])
mlflow.log_metric("macro_avg_precision", classification_rep['macro avg']['precision'])
mlflow.log_metric("macro_avg_recall", classification_rep['macro avg']['recall'])
mlflow.log_metric("macro_avg_f1", classification_rep['macro avg']['f1-score'])

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Log confusion matrix values to MLflow
mlflow.log_metric("true_negatives", int(conf_matrix[0][0]))
mlflow.log_metric("false_positives", int(conf_matrix[0][1]))
mlflow.log_metric("false_negatives", int(conf_matrix[1][0]))
mlflow.log_metric("true_positives", int(conf_matrix[1][1]))

# Save trained model and preprocessing components

# Save the entire model
model.save('sentiment_analysis_model.h5')

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the max_length value
with open('max_length.txt', 'w') as f:
    f.write(str(max_length))

print("Model, tokenizer, and max_length saved successfully.")

# Log model and artifacts to MLflow
mlflow.tensorflow.log_model(model, "model")
mlflow.log_artifact("tokenizer.pickle")
mlflow.log_artifact("max_length.txt")

# Log model file size
model_size = os.path.getsize('sentiment_analysis_model.h5') / 1024  # Size in KB
mlflow.log_metric("model_size_kb", model_size)

# Verify saved components
try:
    # Load the tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)
    
    # Load the max_length
    with open('max_length.txt', 'r') as f:
        loaded_max_length = int(f.read().strip())
    
    print("\nVerification:")
    print(f"Tokenizer vocabulary size: {len(loaded_tokenizer.word_index) + 1}")
    print(f"Loaded max_length: {loaded_max_length}")
    
    # Test tokenization of a sample text
    sample_text = "This movie was fantastic! I really enjoyed it."
    
    # Normalize the text
    normalized_tokens = normalize_text(sample_text)
    normalized_text = ' '.join(normalized_tokens)
    
    # Convert to sequence using the loaded tokenizer
    sequence = loaded_tokenizer.texts_to_sequences([normalized_text])[0]
    
    # Pad the sequence
    padded_sequence = keras.preprocessing.sequence.pad_sequences([sequence], 
                                                              maxlen=loaded_max_length, 
                                                              padding='post', 
                                                              truncating='post')[0]
    
    print(f"\nSample text: '{sample_text}'")
    print(f"Normalized tokens: {normalized_tokens}")
    print(f"Sequence: {sequence}")
    print(f"Padded sequence: {padded_sequence}")
    
    print("\nTokenizer saved and loaded successfully!")
except Exception as e:
    print(f"\nError verifying tokenizer: {str(e)}")

# End MLflow run
mlflow.end_run()

print("\n" + "="*50)
print("MLflow tracking completed!")
print("To view the experiment results, run:")
print("mlflow ui --backend-store-uri sqlite:///mlflow.db")
print("Then open http://localhost:5000 in your browser")
print("="*50)
