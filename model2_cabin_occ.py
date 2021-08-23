# Has cabin alphabet(cabin_a) values and occupation (occ) code in preprocess

# Import basics
from typing import Union

import numpy as np
import pandas as pd

# Import estimators
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Import GridSearch for parameter optimization
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Import scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing the features in the DataFrame using scalers for better model predictions

    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with processed columns
    """
    df["Fare"] = StandardScaler().fit_transform(df[["Fare"]])
    df["Age"] = MinMaxScaler().fit_transform(df[["Age"]])
    df["Pclass"] = MinMaxScaler().fit_transform(df[["Pclass"]])
    df["Parch"] = MinMaxScaler().fit_transform(df[["Parch"]])
    df["SibSp"] = MinMaxScaler().fit_transform(df[["SibSp"]])
    return df


def add_cabina(df: pd.DataFrame) -> pd.DataFrame:
    """Addition of cabin column to DataFrames
    Extracts the first letter in "Cabin"

    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with addition of occupation cabin 
    """
    df["cabin_name"] = df.Cabin.apply(lambda x: str(x)[0])
    cabin_has_alph = [0 if x == "n" else 1 for x in df["cabin_name"]]
    df["cabin_a"] = cabin_has_alph


def add_occ(df: pd.DataFrame) -> pd.DataFrame:
    """Addition of occupation column to DataFrames
    Extracts the second word in "Name" and returns it as the occupation (e.g. Allen, Cpt)

    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with addition of occupation column 
    """
    df["occ"] = [name.split(",")[1].split(".")[0] for name in df["Name"]]
    special_occ = []
    for x in df["occ"]:
        if x == " Mr":
            special_occ.append(0)
        elif x == " Miss":
            special_occ.append(0)
        elif x == " Mrs":
            special_occ.append(0)
        else:
            special_occ.append(1)
    df["occ"] = special_occ


def concat_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Creates dummies for "Sex" and "Embarked" to have only numeric data in the DataFrame

    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with dummy columns for "Sex" and "Embarked"
    """
    dummies = pd.get_dummies(df[["Sex", "Embarked"]])
    df = pd.concat([df, dummies], axis=1)
    return df


def fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Fill null values in "Age" and "Fare"
    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with no na values
    """
    df["Age"].fillna(df.Age.mean(), inplace=True)
    df["Fare"].fillna(df.Fare.median(), inplace=True)
    return df


def modeling_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model: Union[
        LogisticRegression,
        GaussianNB,
        tree.DecisionTreeClassifier,
        KNeighborsClassifier,
        RandomForestClassifier,
        SVC,
    ],
    **kwargs
) -> np.ndarray:
    """Takes each model and fits the DataFrames for cross validation, then creates a classifer for the final prediction

    Args:
        X (pd.DataFrame): Features from train data
        y (pd.Series): Survived column from train data
        model (Union[ LogisticRegression, GaussianNB, tree.DecisionTreeClassifier, 
        KNeighborsClassifier, RandomForestClassifier, SVC, ]): estimators for the data


    Returns:
        np.ndarray: classifier with the different estimators in it
    """
    clf = model(kwargs)
    cv = cross_val_score(clf, X, y, cv=5)
    print(cv)
    print(cv.mean())
    return clf


def create_submission(clf: VotingClassifier, df: pd.DataFrame) -> None:
    """Uses the VotingClassifier to predict the Survival of the individuals in the train set
    Saves a .csv file in the same directory for kaggle submission

    Args:
        clf (VotingClassifier): [description]
        df (pd.DataFrame): [description]
    """
    y_hat_base_vc = clf.predict(df).astype(int)
    basic_submission = {"PassengerId": df.PassengerId, "Survived": y_hat_base_vc}
    base_submission = pd.DataFrame(data=basic_submission)
    base_submission.to_csv("base_submission.csv", index=False)


def main():

    # Load
    train = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/train.csv")
    test = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/test.csv")

    # Preprocess
    for dfs in [test, train]:
        dfs = scale_numeric_features(dfs)
        dfs = add_occ(dfs)
        dfs = fillna(dfs)
        dfs = concat_dummies(dfs)
        dfs.drop(
            columns=["Ticket", "Name", "Cabin", "Embarked", "Sex", "cabin_name"],
            inplace=True,
        )

    # Split X & y
    X_train = train.drop(columns="Survived").drop(columns="PassengerId")
    y_train = train["Survived"]

    # Modeling & CV
    lr = modeling_cv(X_train, y_train, LogisticRegression, max_iter=2000)
    gnb = modeling_cv(X_train, y_train, GaussianNB)
    dt = modeling_cv(X_train, y_train, tree.DecisionTreeClassifier, random_state=1)
    knn = modeling_cv(X_train, y_train, KNeighborsClassifier)
    rf = modeling_cv(X_train, y_train, RandomForestClassifier, random_state=1)
    svc = modeling_cv(X_train, y_train, SVC, probability=True)

    # Ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("knn", knn),
            ("rf", rf),
            ("gnb", gnb),
            ("svc", svc),
            ("dt", dt),
        ],
        voting="soft",
    )
    cv = cross_val_score(voting_clf, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())
    voting_clf.fit(X_train, y_train)

    # Create .csv file to submit
    create_submission(voting_clf, test)


if __name__ == "__main__":
    main()
