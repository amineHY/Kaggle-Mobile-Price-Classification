# Kaggle challenge: [Mobile Price Classification](https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv)

In this kaggle challenge to goal is to determine the price range of a mobile phone. It is a classification problem where the target take four class: 0 to 3. The 20 input features characterizes the phones spec and are used to train a classification model to predict the price range.

# Kaggle Dataset

The dataset is hosted in kaggle website and can be downloaded using this link: [https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv](https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv)

## Install python packages

```
pip3 install -r requirements.txt
```

# Run the code

```
python3 src/train.py
```

# Setup

- Random Forest Classifier
- 5-fold Cross Validation
- Random Grid Search
- Evaluation metric: accuracy

- Best parameters for the classifier are:
  - criterion: entropy
  - max_depth: 12
  - n_estimators: 400
