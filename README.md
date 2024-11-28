# CS4083-NLP

## Description
This repository contains assignments and labs for the CS4083 - Text Mining and NLP course.

---
## A- Assignments
---
### [`Assignment_0`](https://github.com/GDHadeel/CS4083-NLP/blob/main/assignments/assignment_0.ipynb): NumPy Basics
This assignment helps you revise the basics of NumPy, focusing on array creation, manipulation, and basic operations.

#### Goals
- Create and manipulate arrays.
- Perform basic operations like sorting, indexing, and reshaping.
- Use essential NumPy functions.

#### Requirements
- Numpy

To import NumPy, use:
```
import numpy as np
```
  
#### Key Concepts Covered
1. **Array Creation:** **`np.array()`**, **`np.zeros()`**, **`np.ones()`**, etc.
2. **Array Operations:** Indexing, slicing, reshaping, and sorting.
3. **Basic Array Operations:** Addition, subtraction, and other mathematical functions.
4. **Random Numbers:** Generating random arrays using **`np.random`**.
5. **Array Statistics:** Using **`.sum()`**, **`.min()`**, **`.max()`**, etc.

---
### [`Assignment_1`](https://github.com/GDHadeel/CS4083-NLP/blob/main/assignments/Assignment_1.ipynb): 10 Minutes to Pandas
This assignment introduces you to basic Pandas operations for data manipulation and analysis.

#### Goals
- Learn how to create and manipulate Pandas objects (Series and DataFrames).
- Understand how to select, filter, and apply operations to data.


#### Requirements
- Pandas
- Numpy
- Jupyter Notebook or Python IDE

To import Pandas, use:
```
import pandas as pd
```

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
---
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


---
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


---
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


---
### [`Lab 4`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab_4_Sentiment_Analysis.ipynb): Sentiment Analysis
This lab focuses on understanding the difference between data-centric and model-centric to solving machine learning problems. It demonstrates how improving the quality of the data can lead to better performance, even when the model remains unchanged.

#### Goals
- Train a baseline model.
- Evaluate the model's performance.
- Identify and address data issues (e.g., HTML tags, incorrect labels).
- Improve the model performance by cleaning the data.

#### Example Review

- Review: Excellent! I look forward to every issue. I had no idea just how much I didn't know.
  - Label: ⭐️⭐️⭐️⭐️⭐️ (Good)

- Review: My son waited and waited, it took the 6 weeks to get delivered, but when it got here he was so disappointed.
  - Label: ⭐️ (Bad)

#### Requirements
To run the lab, you’ll need the following Python libraries:
-**`scikit-learn`**
-**`pandas`**

You can install them using the following command:
```
!pip install scikit-learn pandas
```

#### Steps

###### 1. Load the Data
First, load the training and test datasets:
  ```python
     import pandas as pd
     train = pd.read_csv('reviews_train.csv')
     test = pd.read_csv('reviews_test.csv')

     test.sample(5)
  ```

###### 2. Train a Baseline Model
We use a simple pipeline with **`CountVectorizer`**, **`TfidfTransformer`**, and a linear classifier (**`SGDClassifier`**):
  ```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

sgd_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier())
])

sgd_clf.fit(train['review'], train['label'])
```

###### 3. Evaluate the Model
After training, evaluate the accuracy of the model:
  ```python
from sklearn import metrics

def evaluate(clf):
    pred = clf.predict(test['review'])
    acc = metrics.accuracy_score(test['label'], pred)
    print(f'Accuracy: {100*acc:.1f}%')

evaluate(sgd_clf)
  ```

###### 4. Improve the Model
Try a Different Model: You can experiment with other classifiers, such as Naive Bayes or Random Forests, to see if they perform better.

- Tune Hyperparameters: Adjust hyperparameters to improve accuracy.
- Ensemble Methods: Combine multiple models for better performance.
  -Example of training a Naive Bayes classifier:
    ```python
    from sklearn.naive_bayes import MultinomialNB

    NB_clf = Pipeline([
      ('vect', CountVectorizer()),
      ('tfidf', TfidfTransformer()),
      ('clf', MultinomialNB())  
    ])
    NB_clf.fit(train['review'], train['label'])
    evaluate(NB_clf)
    ```
  
###### 5. Data Cleaning
On inspection, the data contains some issues, such as HTML tags and incorrect labels. We clean the data by filtering out these noisy examples.

 ```python
def is_bad_data(review: str) -> bool:
    return ('<' in review and '>' in review) or '&' in review

train_clean = train[~train['review'].map(is_bad_data)]
```
After cleaning the data, retrain the model and evaluate its performance:
 ```python
sgd_clf_clean = clone(sgd_clf)
sgd_clf_clean.fit(train_clean['review'], train_clean['label'])
evaluate(sgd_clf_clean)
```

#### Dataset [`reviews_train`](https://github.com/GDHadeel/CS4083-NLP/blob/main/dataset/reviews_train.csv), [`reviews_test`](https://github.com/GDHadeel/CS4083-NLP/blob/main/dataset/reviews_test.csv)

---
#### [`Lab5.A`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Introduction_to_Topic_Modeling.ipynb): Topic Modeling Using LDA
In this lab, we implement Topic Modeling using Latent Dirichlet Allocation (LDA) to uncover hidden topics in a collection of text data. We focus on understanding the process of preparing text data for LDA, training the model, and analyzing the results.

#### Goals
- Understand how LDA works as a generative probabilistic model.
- Apply LDA for topic modeling on a set of NeurIPS papers.
- Visualize the topics to interpret and analyze them effectively.

#### Requirements
- **`pandas`**
- **`gensim`** for LDA modeling
- **`nltk`** for stopwords
- **`pyLDAvis`** for visualization
- **`wordcloud`** for word cloud generation

- You can install the necessary dependencies by running:
 ```
pip install gensim nltk pyLDAvis wordcloud
```

#### Steps

###### 1. Loading the Data
The dataset used in this project contains research papers from the NeurIPS (NIPS) conference, spanning from 1987 to 2016. The papers are loaded from a zipped CSV file.
 ```python
import zipfile
import pandas as pd

# Open the zip file and extract the content
with zipfile.ZipFile("NIPS Papers.zip", "r") as zip_ref:
    zip_ref.extractall("temp")

# Load the CSV data into a DataFrame
papers = pd.read_csv("temp/NIPS Papers/papers.csv")
```

###### 2. Data Cleaning
We remove unnecessary columns and perform basic preprocessing, like removing punctuation and converting text to lowercase.
 ```python
import re

# Remove punctuation and convert text to lowercase
papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())
```

###### 3. Exploratory Data Analysis
A WordCloud is generated to visualize the most common words in the dataset, ensuring the text is ready for topic modeling.
```python
from wordcloud import WordCloud

long_string = ','.join(list(papers['paper_text_processed'].values))
wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')
wordcloud.generate(long_string)
wordcloud.to_image()

```

###### 4. Preparing the Data for LDA
We tokenize the text and remove stopwords, then prepare the data in a format suitable for training the LDA model.
```python
import gensim
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

stop_words = stopwords.words('english')

# Tokenize and remove stopwords
data_words = list(sent_to_words(papers['paper_text_processed'].values.tolist()))
data_words = remove_stopwords(data_words)
```

###### 5. Training the LDA Model
The LDA model is built using **`gensim`**'s **`LdaMulticore`**, specifying the number of topics to extract from the dataset. For this tutorial, we use 10 topics.
```python
import gensim.corpora as corpora
from pprint import pprint

# Create a Dictionary and Corpus
id2word = corpora.Dictionary(data_words)
corpus = [id2word.doc2bow(text) for text in data_words]

# Build the LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus, id2word=id2word, num_topics=10)

# Print the topics discovered by LDA
pprint(lda_model.print_topics())
```
---
#### [`Lab5.B`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Topic_Modeling_with_BERTopic.ipynb): Topic Modeling with BERTopic

In this lab, we explore BERTopic, a deep learning-based approach for topic modeling. Topic modeling is the process of discovering themes within a large collection of text documents, similar to organizing a bookstore by book topics. The BERTopic model uses BERT embeddings, clustering, and TF-IDF to find and represent topics in text data.

#### Goals
- Understand how BERTopic works for extracting topics from a set of documents.
- Implement BERTopic on a dataset of 20 newsgroup articles.
- Evaluate and visualize the discovered topics.
- Analyze the performance of BERTopic in comparison to traditional methods like LDA.

#### Approach
1. **Document Embeddings:** We convert the text data into numerical representations using BERT embeddings.
2. **Dimensionality Reduction:** We apply UMAP to reduce the dimensions of the embeddings for efficient clustering.
3. **Clustering:** We use HDBSCAN for clustering the reduced embeddings into topics.
4. **TF-IDF:** Class-based TF-IDF is used to extract the most relevant words for each topic.
5- **Topic Visualization:** We visualize the topics and evaluate their coherence.

#### Steps

###### 1. Install Required Libraries:
```
pip install bertopic
```

###### 2.Preprocessing:
- Load and clean the dataset.
- Tokenize and lemmatize the text to prepare it for modeling.

###### 3.Topic Modeling:
- Extract document embeddings using the SentenceTransformer.
- Reduce dimensions using UMAP.
- Cluster the embeddings with HDBSCAN.
- Use TF-IDF to refine topic-word distributions.

###### 4.Evaluation:
- Visualize topics using interactive plots.
- Evaluate topic coherence with different measures.

#### Requirements
Libraries: **`bertopic`**, **`sentence-transformers`**, **`umap`**, **`hdbscan`**, **`nltk`**, **`pandas`**, **`gensim`**, **`sklearn`**, etc.


---
#### [`Lab5.C`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Evaluate_Topic_Models.ipynb): Evaluate Topic Models
In this lab, we explore Latent Dirichlet Allocation (LDA) for topic modeling using Python and Gensim. The main objective is to evaluate topic models through topic coherence and optimize LDA parameters. We use a dataset of NeurIPS (NIPS) conference papers published from 1987 to 2016.

#### Goals
- Understand and evaluate topic models using coherence metrics.
- Implement LDA topic modeling on a text dataset (NeurIPS papers).
- Optimize the LDA model by tuning hyperparameters (e.g., number of topics, alpha, and beta).
- Visualize the results using pyLDAvis.

#### Requirements
- **`Gensim`**
- **`Pandas`**
- **`NITK`**
- **`Spacy`**

To install the required libraries, run:
```
pip install gensim pandas nltk spacy
```
  
#### Steps
1. **Loading Data:** Load a dataset (e.g., NeurIPS conference papers) to perform topic modeling.
2. **Data Cleaning:** Preprocess the text by removing punctuation, converting text to lowercase, and tokenizing.
3. **Phrase Modeling: **Create bigrams and trigrams using Gensim's **`Phrases`** model to improve topic representation.
4. **Data Transformation:** Create a dictionary and corpus for the LDA model.
5. **Base Model:** Train the base LDA model using the corpus and dictionary.
6. **Hyper-parameter Tuning:** Fine-tune the model's parameters for better topic coherence.
7. **Evaluation:** Use topic coherence measures to evaluate the model.
8. **Visualization:** Visualize the topics and their coherence.

#### Example Code 
```python
import gensim
import pandas as pd
import re
from nltk.corpus import stopwords
import spacy

# Data loading and cleaning
papers = pd.read_csv('papers.csv')
papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x).lower())

# Tokenization
def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

data_words = list(sent_to_words(papers['paper_text_processed'].values.tolist()))

# Bigrams and trigrams
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
```

#### Running the Model
Once you've preprocessed the data, you can train the LDA model using Gensim:
```python
from gensim import corpora
from gensim.models import LdaModel

# Create Dictionary and Corpus
id2word = corpora.Dictionary(data_words)
corpus = [id2word.doc2bow(text) for text in data_words]

# Train LDA Model
lda_model = LdaModel(corpus, num_topics=5, id2word=id2word, passes=15)

# Print Topics
topics = lda_model.print_topics(num_words=5)
for topic in topics:
    print(topic)
```

#### Evaluation Metrics
We evaluate the model using coherence measures such as **`C_v`** and **`C_p`**. You can visualize the coherence scores to choose the best model.
```python
from gensim.models import CoherenceModel

# Coherence Evaluation
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'Coherence Score: {coherence_lda}')
```

---
#### [`Lab 6`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab_6_summarize_dialogue.ipynb): Generative AI Use Case: Summarize Dialogue
In this lab, we will explore the use of generative AI for the task of dialogue summarization. We will examine how different types of prompt engineering (zero-shot, one-shot, and few-shot) can influence the output of a pre-trained Large Language Model (LLM), FLAN-T5. 

#### Goals
- Summarize dialogues using a generative AI model.
- Explore different inference types: zero-shot, one-shot, and few-shot.
- Learn prompt engineering techniques to improve the output of the model.

#### Lab Breakdown
1. **Set up Kernel and Dependencies**
Install required libraries such as PyTorch and Hugging Face transformers.

2. **Summarize Dialogue without Prompt Engineering**
Use FLAN-T5 to summarize dialogues from the DialogSum dataset without any specific instruction.

3. **Summarize Dialogue with an Instruction Prompt**
Explore zero-shot inference by providing the model with a task-specific prompt to guide its summarization.

4. **Summarize Dialogue with One-Shot and Few-Shot Inference**
Provide one or more example dialogue-summary pairs to the model and compare the results.

5. **Generative Configuration Parameters**
Adjust the model’s generation parameters to control the output style and quality.

#### Code Setup
To run the lab, you’ll need to install the following dependencies:
```
# Install PyTorch
pip install --upgrade pip
pip install --disable-pip-version-check torch==1.13.1 torchdata==0.5.1 --quiet

# Install Hugging Face libraries
pip install transformers==4.27.2 datasets==2.11.0 --quiet
```

#### Dataset
The dataset used in this lab is the **`DialogSum`** dataset, which contains 10,000+ dialogues along with manually labeled summaries.
```python
from datasets import load_dataset
dataset = load_dataset("knkarthick/dialogsum")
```

#### Example
how to use the FLAN-T5 model to summarize a dialogue:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the model and tokenizer
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Example dialogue
dialogue = "Person A: Hello, how are you? Person B: I'm good, thanks! How about you?"
prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary:"

# Tokenize the input
inputs = tokenizer(prompt, return_tensors='pt')

# Generate the summary
summary = tokenizer.decode(model.generate(inputs["input_ids"], max_new_tokens=50)[0], skip_special_tokens=True)
print("Generated Summary:", summary)
```

#### Prompt Engineering Examples
1. **Zero-Shot Inference:** Provide a direct instruction to summarize the dialogue.
2. **One-Shot Inference:** Provide one example dialogue-summary pair to guide the model.
3. **Few-Shot Inference:** Provide multiple examples to further enhance the model's understanding of the task.



