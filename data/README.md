# Sentiment Analysis Dataset

This directory contains datasets used for training the sentiment analysis model.

## Dataset Description

The dataset consists of sentences labeled with sentiment (positive or negative) from three different sources:

1. **Amazon Product Reviews** (`amazon_cells_labelled.txt`)
2. **IMDb Movie Reviews** (`imdb_labelled.txt`)
3. **Yelp Restaurant Reviews** (`yelp_labelled.txt`)

## File Format

Each file contains sentences and their sentiment labels in the following format:
```
[sentence]\t[label]
```

Where:
- `[sentence]` is the text of the review
- `[label]` is either 1 (positive) or 0 (negative)

## Dataset Statistics

- **Amazon**: 1000 sentences (500 positive, 500 negative)
- **IMDb**: 1000 sentences (500 positive, 500 negative)
- **Yelp**: 1000 sentences (500 positive, 500 negative)
- **Total**: 3000 sentences (1500 positive, 1500 negative)

## Usage

This dataset is used to train the LSTM model in `train_model.py`. The model learns to classify text as either positive or negative sentiment.

## Source

This dataset is from the "Sentiment Labelled Sentences Data Set" from the UCI Machine Learning Repository.

Original source: https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences
