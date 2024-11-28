# CS4083-NLP

## Description
This repository contains assignments and labs for the CS4083 - Text Mining and NLP course.

## A- Assignments

### [`Assignment_0`](https://github.com/GDHadeel/CS4083-NLP/blob/main/assignments/assignment_0.ipynb): NumPy Basics
This assignment helps you revise the basics of NumPy, focusing on array creation, manipulation, and basic operations.

#### Goals
- Create and manipulate arrays.
- Perform basic operations like sorting, indexing, and reshaping.
- Use essential NumPy functions.

#### Submission Format
- **NumPy**: The main library used for array operations.
- **Python**: The code is written in Python.

To import NumPy, use:
```
import numpy as np
```

#### Requirements
- Python 3.x
- Numpy
  
#### Key Concepts Covered
1. **Array Creation:** **`np.array()`**, **`np.zeros()`**, **`np.ones()`**, etc.
2. **Array Operations:** Indexing, slicing, reshaping, and sorting.
3. **Basic Array Operations:** Addition, subtraction, and other mathematical functions.
4. **Random Numbers:** Generating random arrays using **`np.random`**.
5. **Array Statistics:** Using **`.sum()`**, **`.min()`**, **`.max()`**, etc.


### [`Assignment_1`](https://github.com/GDHadeel/CS4083-NLP/blob/main/assignments/Assignment_1.ipynb): 10 Minutes to Pandas
This assignment introduces you to basic Pandas operations for data manipulation and analysis.

#### Goals
- Learn how to create and manipulate Pandas objects (Series and DataFrames).
- Understand how to select, filter, and apply operations to data.


#### Submission Format
Notebook: **`.ipynb`** format or Python script **`.py`**

To import Pandas, use:
```
import pandas as pd
```

#### Requirements
- Python 3.x
- Pandas
- Numpy
- Jupyter Notebook or Python IDE

#### Key Concepts Covered
1. Creating Pandas Objects:
**`Series`** and **`DataFrame`** creation.

2. Viewing Data:
**`head()`**, **`tail()`**, **`index`**, **`columns`**, **`to_numpy()`**, etc.

3. Selection:
Selecting by label (**`loc`**), position (**`iloc`**), and boolean indexing.

4. Handling Missing Data:
Using **`isna()`**, **`fillna()`**, **`dropna()`**.

5. Basic Operations:
Arithmetic operations, applying functions (**`agg`**, **`transform`**), and string methods.

6. Sorting:
Sorting data by index and values using **`sort_index()`** and **`sort_values()`**.

7. Aggregating & Transforming Data:
Using aggregation functions (**`mean()`**, **`sum()`**) and transformations.

---

## B- Labs

### [`Lab 1`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Data%20analysis%20with%20pandas.ipynb): Data Analysis with Pandas
This lab focuses on data cleaning and exploratory data analysis (EDA) using the pandas library. The dataset contains information about approximately 6,000 books from Goodreads, including details like book ratings, review counts, and genres. The objective is to clean the data, parse and transform it, and perform basic EDA.

#### Goals
- Load and clean datasets with missing values.
- Parse and transform data columns.
- Group data by specific features and perform aggregation.

#### 1. Loading and Cleaning with Pandas
The first step is loading the **`goodreads.csv`** dataset and cleaning it. This includes:

- Checking for missing values and handling them (e.g., removing or filling missing data).
- Ensuring each column has the correct data type.
- Parsing columns like **`author_url`** and **`genre_urls`** to extract more useful information.

###### Cleaning Steps:
1. Load the dataset into a pandas DataFrame.
2. Examine and clean the data (handle missing values, correct data types).
3. Parse relevant columns to extract additional information (e.g., split full author names or genre URLs).

###### Example Code:
  ```python
     import pandas as pd
     df = pd.read_csv('goodreads.csv')
     df.columns = ["rating", 'review_count', 'isbn', 'booktype', 'author_url', 'year', 'genre_urls', 'dir', 'rating_count', 'name']
     df.head()
  ```

##### 2. Parsing and Completing the Dataframe 
Extract author names from **`author_url`** and genres from **`genre_urls`** using string operations.

###### Example Code:
  ```python
     def get_author(url):
       return url.split('/')[-1].split('.')[1]

     df['author'] = df.author_url.map(get_author)
  ```

##### 3. Grouping
Group data by author or year and calculate aggregates (e.g., average rating).

###### Example Code:
  ```python
     dfgb_author = df.groupby('author')
     dfgb_author['author'].count()
  ```

##### Dataset: [`goodreads`](https://github.com/GDHadeel/CS4083-NLP/blob/main/dataset/goodreads.csv)



### [`Lab 2`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Data%20analysis%20with%20pandas.ipynb): Text Data Preprocessing (NLP pipline)
In this lab, we explore how to preprocess tweets for sentiment analysis using the NLTK package. By the end of the lab, you will understand how to build an NLP pipeline to process and analyze Twitter datasets.

#### Goals 
1. Load and explore a Twitter dataset.
2. Perform basic text preprocessing tasks.
3. Use techniques like Bag of Words and TF-IDF to represent text.

#### Setup
To get started, we will use the NLTK package and the sample Twitter dataset included in it. The dataset consists of 5000 positive and 5000 negative tweets.

To install the necessary libraries, run:
```
pip install nltk matplotlib
```
#### Import Libraries:
```
import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random
```

#### 1. Loading the Twitter Dataset
The sample dataset includes positive and negative tweets. Use the following code to load the data:
```
nltk.download('twitter_samples')
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')
```

#### 2. Preprocessing the Text
1. Lowercasing text.
2. Removing stop words, punctuation, and irrelevant content (URLs, hashtags).
3. Tokenizing the text.
4. Stemming the words.
   
###### Example of how we clean and process a tweet:
  ```python
import re
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Example tweet
tweet = "I love NLP! #excited :) http://example.com"

# Clean tweet
tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)  # remove URLs
tweet = re.sub(r'#', '', tweet)  # remove hashtags
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
tokens = tokenizer.tokenize(tweet)

# Remove stop words and punctuation
stop_words = stopwords.words('english')
tokens_clean = [word for word in tokens if word not in stop_words and word not in string.punctuation]

# Stemming
stemmer = PorterStemmer()
tokens_stemmed = [stemmer.stem(word) for word in tokens_clean]

print(tokens_stemmed)
  ```

#### 3. Data Representation (Vectorization)
We use vectorization to convert text into numerical data. We can use techniques like Bag of Words (BoW) and TF-IDF.

###### Example of Bag of Words:
  ```python
     from sklearn.feature_extraction.text import CountVectorizer
     vectorizer = CountVectorizer()
     X = vectorizer.fit_transform([tweet])
     print(vectorizer.get_feature_names_out())
  ```

###### Example of TF-IDF:
  ```python
     from sklearn.feature_extraction.text import TfidfVectorizer
     tfidf_vectorizer = TfidfVectorizer()
     tfidf = tfidf_vectorizer.fit_transform([tweet])
     print(tfidf.toarray())
  ```



### [`Lab 3`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab3_Word_Embedding.ipynb): Regex & ArabicNLP Embedding
This lab introduces Natural Language Processing (NLP) techniques, including Regex for text processing and various Word Embedding methods. Students will explore these methods through Python implementations, visualizations, and evaluations of their effectiveness.

#### Goals
- Implement Regex for automated bill generation.
- Learn and apply multiple **Word Embedding techniques**:
  - One Hot Encoding
  - Bag of Words
  - TF-IDF
  - Word2Vec (CBOW and Skip Gram)
  - FastText
- Visualize and evaluate embeddings using t-SNE and discuss their performance.

#### Pre-requisites

1. Install required libraries:
 ```
pip install fasttext
pip install gensim
pip install scikit-learn
 ```
2. Download pre-trained models for **Word2Vec (CBOW)**, **Word2Vec (Skip Gram)**, and FastText from the following links and place them in the same directory as the notebook:

- Word2Vec CBOW: [`Download`](https://github.com/mmdoha200/ArWordVec)
- Word2Vec Skip Gram: [`Download`](https://github.com/mmdoha200/ArWordVec)
- FastText: [`Download`](https://blackboard.effatuniversity.edu.sa/ultra/courses/_21032_1/cl/outline)

##### Part 1: Rule-Based NLP with Regex
- Objective: Implement Regex to parse a given text and extract product names, quantities, and prices for bill generation.
- Example:
  - Input: "I bought three Samsung smartphones at $150 each, four kilos of fresh bananas for $1.2 a kilogram..."
  - Output: A table displaying the product, quantity, unit price, and total price.

##### Part 2: Word Embedding Techniques
- **Objective:** Explore and visualize different word embedding methods.
  - **One Hot Encoding:** Each word is represented by a vector with a single '1' indicating the presence of the word.
  - **Bag of Words:** Counts word frequencies without considering grammar or order.
  - **TF-IDF:** Evaluates the importance of a word in a document relative to a corpus.
  - **Word2Vec (CBOW):** Predicts words based on surrounding context words.
  - **Word2Vec (Skip Gram):** Predicts surrounding words based on a target word.
  - **FastText:** Represents each word as a bag of character n-grams for better handling of out-of-vocabulary words.

#### Visualizations
t-SNE plots to visualize word embeddings and the semantic relationships between words.

#### How to Use
1. Download the **`word.pkl`** file containing the Arabic words dataset.
2. Download the pre-trained models for Word2Vec and FastText.
3. Run the Jupyter notebook. The code will:
  - Generate a bill from a sample shopping description using Regex.
  - Apply and visualize word embeddings using t-SNE.
4. Evaluate and compare the different word embedding techniques based on their visualizations and effectiveness.



### [`Lab 4`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab_4_Sentiment_Analysis.ipynb): Sentiment Analysis



### Lab 5
#### [`Topic Modeling Using LDA`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Introduction_to_Topic_Modeling.ipynb)



#### [`Topic Modeling with BERTopic`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Topic_Modeling_with_BERTopic.ipynb).



#### [`Evaluate Topic Models`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Evaluate_Topic_Models.ipynb).


 
### Lab 6
#### [`Generative AI Use Case: Summarize Dialogue`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab_6_summarize_dialogue.ipynb).







