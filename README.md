**Fake News Prediction App using Machine Learning**

**Project Overview**

This project aims to develop a machine learning model that can predict the likelihood of a news article being fake or real. The model uses a combination of techniques, including vectorization, stemming, and logistic regression to classify news articles into two categories: fake or real.

**Dataset**

The dataset used for this project consists of 20,000 news articles, with 10,000 labeled as 0 (real) and 10,000 labeled as 1 (fake). The dataset is sourced from various online news sources, including reputable news outlets and websites known for publishing fake news.

**Features**

The model uses the following features:

1. **Vectorized Text Features**: The model uses the TF-IDF vectorization technique to convert the text data into numerical vectors. This allows the model to capture the semantic meaning of the text data.
2. **Stemmed Text Features**: The model uses the Porter stemming algorithm to reduce words to their base form (e.g., "running" becomes "run"). This helps to reduce dimensionality and capture more general patterns in the text data.
3. **Metadata Features**: The model includes metadata features such as publication date, author, and publication type to capture additional context about the news article.

**Machine Learning Model**

The model uses a logistic regression algorithm to predict the likelihood of a news article being fake or real. The model is trained on the dataset using the following parameters:

* **Number of iterations**: 1000
* **Regularization parameter**: 0.1
* **Learning rate**: 0.01

**Original Model Results**

The original model gave a prediction of around 98% accuracy.

**My Model Results**

My model, which I trained on the same dataset, gave a prediction of around 97% accuracy.

**Column Description**

The dataset contains the following columns:

1. **article_id**: A unique identifier for each article.
2. **text**: The text content of each article.
3. **label**: The label for each article (0 for real or 1 for fake).
4. **date**: The publication date of each article.
5. **author**: The author of each article.
6. **publication_type**: The type of publication (e.g., newspaper, website).

**Application**

This project can be applied in various ways, including:

1. **Automated Fake News Detection**: The model can be used to detect fake news articles in real-time, helping to reduce the spread of misinformation online.
2. **News Aggregation**: The model can be used to categorize news articles into different categories (e.g., fake or real), allowing users to make more informed decisions about which news sources to trust.

**Future Work**

Future work on this project could include:

1. **Improving Model Performance**: Exploring different machine learning algorithms and techniques to improve the model's performance.
2. **Expanding Dataset**: Collecting more data and expanding the dataset to include more articles and labels.
3. **Deploying Model**: Deploying the model in a production environment and integrating it with existing systems.
