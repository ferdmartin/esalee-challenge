# Dr. Lee Challenge - Amazon Reviews

## Description
This repository contains the exploration and implementation of machine learning models designed to predict ratings based on music reviews from Amazon. In our experimental process, we tested two models to compare to the provided baseline results:
- Lasso regression (to challenge baseline MSE)
- Neural Collaborative Filtering network with associated features

We also created features based on feature-engineering techniques to enhance model performance and tested them against baseline results.

Repo structure:
- `src/experiments.ipynb`: Experiments and results
- `app`: Web application to test our model passing (reviewerID, itemID)

## Requirements
- numpy
- pandas
- matplotlib
- scikit-learn
- tensorflow
- tensorflow-hub


![Best Architecture - Neural Collaborative Filtering](https://github.com/ferdmartin/esalee-challenge/blob/main/misc/Amazon%20Reviews%20Architecture.png)