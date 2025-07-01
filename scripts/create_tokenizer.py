import os
import re
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow import keras
import pandas as pd

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def normalize_text(text):
    # Lowercase the text
    text = text.lower()
    
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

def main():
    # Load the datasets
    print("Loading datasets...")
    imdb_path = 'data/imdb_labelled.txt'
    yelp_path = 'data/yelp_labelled.txt'
    amazon_path = 'data/amazon_cells_labelled.txt'
    
    imdb_df = pd.read_csv(imdb_path, delimiter='\t', header=None, names=['review', 'sentiment'])
    yelp_df = pd.read_csv(yelp_path, delimiter='\t', header=None, names=['review', 'sentiment'])
    amazon_df = pd.read_csv(amazon_path, delimiter='\t', header=None, names=['review', 'sentiment'])
    
    # Combine the datasets
    combined_df = pd.concat([imdb_df, yelp_df, amazon_df], ignore_index=True)
    
    # Apply normalization to the dataset
    print("Normalizing text...")
    combined_df['normalized_tokens'] = combined_df['review'].apply(normalize_text)
    
    # Create a vocabulary and convert words to integers
    print("Creating tokenizer...")
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(combined_df['normalized_tokens'].apply(' '.join))
    
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    print(f"Vocabulary size: {vocab_size}")
    
    # Calculate max_length (use the 95th percentile or a fixed value, whichever is smaller)
    combined_df['integer_tokens'] = combined_df['normalized_tokens'].apply(lambda x: tokenizer.texts_to_sequences([' '.join(x)])[0])
    max_length = min(100, int(combined_df['integer_tokens'].apply(len).quantile(0.95)))
    print(f"Max sequence length: {max_length}")
    
    # Save the tokenizer
    print("Saving tokenizer...")
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the max_length value
    with open('models/max_length.txt', 'w') as f:
        f.write(str(max_length))
    
    print("Tokenizer and max_length saved successfully.")
    
    # Test the tokenizer
    print("\nTesting tokenizer...")
    sample_text = "This movie was fantastic! I really enjoyed it."
    normalized_tokens = normalize_text(sample_text)
    normalized_text = ' '.join(normalized_tokens)
    sequence = tokenizer.texts_to_sequences([normalized_text])[0]
    padded_sequence = keras.preprocessing.sequence.pad_sequences([sequence], 
                                                             maxlen=max_length, 
                                                             padding='post', 
                                                             truncating='post')[0]
    
    print(f"Sample text: '{sample_text}'")
    print(f"Normalized tokens: {normalized_tokens}")
    print(f"Sequence: {sequence}")
    print(f"Padded sequence: {padded_sequence}")

if __name__ == "__main__":
    main()
