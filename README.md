# Dr. Lee Challenge - Amazon Reviews

## Description
This repository contains the exploration and implementation of machine learning models designed to predict ratings based on music reviews from Amazon. In our experimental process, we tested two models to challenge to the provided baseline results:
- Lasso regression: To challenge the baseline MSE. To handle our text variables, we transformed "summary" and "reviewText" into numerical values using TfidfVectorizer. TfidfVectorizer converts text to a matrix of TF-IDF features, capturing the importance of words in relation to the documents.
- Neural Collaborative Filtering network with associated features: To challenge the most robust result provided. To process our text features, we utilized the Universal Sentence Encoder (USE) from Google to process text inputs. USE creates high-dimensional vectors for sentences, capturing their semantic meanings for improved model performance.

We also created features based on feature-engineering techniques to enhance model performance and tested them against baseline results.

Repo structure:
- `src/experiments.ipynb`: Experiments and results
- `app`: Web application to test our model passing (reviewerID, itemID). Link to web app: https://ferdmartin-recommenderamazonreviews.hf.space

## Results
### Benchmarks provided by challenge guidelines
- Naive baseline: 1.02111
- Robust Baseline: 0.60756

### Our performance
- Naive baseline (Lasso) without feature engineering: 1.004149
- Baseline (Lasso) with feature-engineered variables: 0.9845879
- Robust model (Neural Collaborative Filtering): 0.591156

## Requirements
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- tensorflow-hub

![Best Architecture - Neural Collaborative Filtering](https://github.com/ferdmartin/esalee-challenge/blob/main/misc/Amazon%20Reviews%20Architecture.png)
