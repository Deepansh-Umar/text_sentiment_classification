# Phrase Sentiment Analysis and Machine Learning Pipeline

## Project Overview
This project delivers an end-to-end natural language processing (NLP) and machine 
learning classification engine built to predict sentiment profiles from text phrases. 
The system features customized metadata imputation routines, high-dimensional n-gram 
feature extractions, and a multi-model voting ensemble strategy designed to capture 
both syntactic structure and deeper semantic meaning.

## Dataset and Structural Handling
The source dataset consists of individual phrase snippets accompanied by discrete 
sentiment labels. It incorporates specific numeric structural attributes that occasionally 
contain missing values. 

To overcome these data challenges without losing critical data matrices, the system 
implements programmatic metadata extraction engines:
- Feature 1 Imputation: Calculates total word lengths per phrase sequence.
- Feature 2 Imputation: Extracts counts of capitalized structural tokens.
- Feature 3 Imputation: Isolates and counts individual punctuation frequencies.

## Pipeline Architecture
The analytical pipeline handles categorical text data and scaled numeric data 
simultaneously using scikit-learn's ColumnTransformer:

1. Word-Level TF-IDF: Extracts word tokens ranging from unigrams to trigrams, capping 
   features at 15,000 components using sublinear term frequency scaling.
2. Character-Level TF-IDF: Analyzes character sequences spanning from 2-grams to 5-grams 
   up to a limit of 10,000 components to robustly capture root spellings and sub-words.
3. Continuous Preprocessing: Subjects numerical features to a MinMaxScaler step to 
   standardize input weightings uniformly for distance-sensitive and linear estimators.

## Experimental Framework
A large range of algorithm types were trained, evaluated, and cross-validated:
- Linear Models: Logistic Regression, Ridge Classifier, Passive Aggressive Classifier, 
  Stochastic Gradient Descent (SGD)
- Generative Models: Multinomial Naive Bayes
- Discriminative Boundaries: Linear Support Vector Classification (LinearSVC)
- Ensemble Gradient Boosters: LightGBM Classifier, XGBoost Classifier

## Ensemble Strategy and Refitting
The final model architecture deploys a hard-voting ensemble (VotingClassifier) that 
harmonizes predictions across Logistic Regression, Naive Bayes, LinearSVC, LightGBM, 
and XGBoost. 

After using 5-fold stratified cross-validation on an 80/20 train-validation split to 
confirm performance, the structural ensemble was refitted across 100 percent of the 
primary data matrix to generate optimized test label predictions.

## Requirements
- Python 3.12+
- numpy
- pandas
- matplotlib
- scikit-learn
- lightgbm
- xgboost
