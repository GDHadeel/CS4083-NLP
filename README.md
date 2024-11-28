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

#### Loading and Cleaning with Pandas
The first step is loading the **`goodreads.csv`** dataset and cleaning it. This includes:

- Checking for missing values and handling them (e.g., removing or filling missing data).
- Ensuring each column has the correct data type.
- Parsing columns like **`author_url`** and **`genre_urls`** to extract more useful information.

##### Cleaning Steps:



### Lab 2
#### [`Text Data Preprocessing (NLP pipline)`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab2NLPpipeline.ipynb).

### Lab 3
#### [`Regex & ArabicNLP Embedding`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab3_Word_Embedding.ipynb).

### Lab 4
#### [`Sentiment Analysis`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab_4_Sentiment_Analysis.ipynb).

### Lab 5
#### [`Topic Modeling Using LDA`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Introduction_to_Topic_Modeling.ipynb).

#### [`Topic Modeling with BERTopic`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Topic_Modeling_with_BERTopic.ipynb).

#### [`Evaluate Topic Models`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab5_Evaluate_Topic_Models.ipynb).

### Lab 6
#### [`Generative AI Use Case: Summarize Dialogue`](https://github.com/GDHadeel/CS4083-NLP/blob/main/Labs/Lab_6_summarize_dialogue.ipynb).







