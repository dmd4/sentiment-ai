# Data Sources for Sentiment Analysis API

## Citation

This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al., KDD 2015.
**Please cite the paper if you use this dataset in your work.**

## Overview

The sentiment analysis model in this project is trained on a dataset containing sentences labeled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants. The dataset provides a balanced collection of clearly positive or negative sentences with minimal neutral content.

## Dataset Format

```
sentence \t score \n
```

Where:
- `sentence` is the text of the review
- `score` is either 1 (for positive) or 0 (for negative)
- Each entry is separated by a tab character and ends with a newline

## Dataset Details

The sentences come from three different websites/fields:

1. **IMDb Movie Reviews**
   - Source: imdb.com
   - Size: 500 positive and 500 negative sentences
   - File Path: `data/imdb_labelled.txt`
   - Original Dataset: Maas et. al., 2011 'Learning word vectors for sentiment analysis'

2. **Amazon Product Reviews**
   - Source: amazon.com
   - Size: 500 positive and 500 negative sentences
   - File Path: `data/amazon_cells_labelled.txt`
   - Original Dataset: McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating dimensions with review text'

3. **Yelp Reviews**
   - Source: yelp.com
   - Size: 500 positive and 500 negative sentences
   - File Path: `data/yelp_labelled.txt`
   - Original Dataset: Yelp dataset challenge http://www.yelp.com/dataset_challenge

The sentences were selected randomly from larger datasets of reviews, with the goal of including only clearly positive or negative sentences and avoiding neutral content.

## Data Handling in This Project

The application handles data in the following ways:

1. **Data Loading**: The `load_data()` function in `train_model.py` attempts to load the datasets from the specified file paths.

2. **Sample Data Generation**: If the datasets are not found, the application automatically generates sample data for demonstration purposes. This ensures the API can run without requiring external data downloads.

3. **Data Preprocessing**: Before training, the text data undergoes preprocessing:
   - Conversion to lowercase
   - Removal of special characters and digits
   - Tokenization
   - Stop word removal
   - Stemming

4. **Data Combination**: The three datasets are combined to create a larger, more diverse training set.

## Using Your Own Data

You can replace the default datasets with your own data by:

1. Creating text files in the same format (text reviews with tab-separated binary labels)
2. Placing them in the `data/` directory with the expected filenames
3. Running the training script: `python train_model.py`

## References

1. Kotzias, D., Denil, M., De Freitas, N., & Smyth, P. (2015). From Group to Individual Labels using Deep Features. In Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 597-606).

2. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies (pp. 142-150).

3. McAuley, J., & Leskovec, J. (2013). Hidden factors and hidden topics: understanding rating dimensions with review text. In Proceedings of the 7th ACM Conference on Recommender Systems (pp. 165-172).

4. Yelp Dataset Challenge. http://www.yelp.com/dataset_challenge
