# rf_grid_search.py
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
import config

if __name__ == "__main__":
    # read the training data
    df = pd.read_csv(config.TRAINING_FILE)

    # features are all columns without price_range
    # note that there is no id column in this dataset

    # Here we have training features
    X = df.drop("price_range", axis=1).values
    # and the targets
    y = df.price_range.values

    print("define the model here")
    # i am using random forest with n_jobs=-1
    # n_jobs=-1 => use all cores
    classifier = ensemble.RandomForestClassifier(n_jobs=-1, random_state=0)

    print("[INFO] define a grid of parameters")
    # this can be a dictionary or a list of
    # dictionaries
    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"]}

    # initialize grid search
    # estimator is the model that we have defined
    # param_grid is the grid of parameters
    # we use accuracy as our metric. you can define your own
    # higher value of verbose implies a lot of details are printed
    # cv=5 means that we are using 5 fold cv (not stratified)
    # model = model_selection.GridSearchCV(
    #     estimator=classifier,
    #     param_grid=param_grid,
    #     scoring="accuracy",
    #     verbose=10,
    #     n_jobs=1,
    #     cv=5
    # )

    # # fit the model and extract best score
    # model.fit(X, y)
    # print(f"Best score: {model.best_score_}")

    # print("Best parameters set:")
    # best_parameters = model.best_estimator_.get_params()
    # for param_name in sorted(param_grid.keys()):
    #     print(f"\t{param_name}: {best_parameters[param_name]}")

    # parameter search
    param_grid = {
        "n_estimators": np.arange(100, 1500, 100),
        "max_depth": np.arange(1, 31),
        "criterion": ["gini", "entropy"]}

    model_rand = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model_rand.fit(X, y)
    print(f"Best score: {model_rand.best_score_}")
    print("Best parameters set:")
    best_parameters = model_rand.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")

    # Get best model
    best_model = model_rand.best_estimator_

    df = pd.read_csv("../input/test.csv")
    X_valid = df.drop('id', axis=1).values
    preds = best_model.predict(X_valid)
    print(preds)
