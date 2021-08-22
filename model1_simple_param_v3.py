# import basics
from typing import Union

import numpy as np
import pandas as pd

# import estimators
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# import GridSearch for parameter optimization
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# import scaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Fare"] = StandardScaler().fit_transform(df[["Fare"]])
    df["Age"] = MinMaxScaler().fit_transform(df[["Age"]])
    df["Pclass"] = MinMaxScaler().fit_transform(df[["Pclass"]])
    df["Parch"] = MinMaxScaler().fit_transform(df[["Parch"]])
    df["SibSp"] = MinMaxScaler().fit_transform(df[["SibSp"]])
    return df


def concat_dummies(df: pd.DataFrame) -> pd.DataFrame:
    dummies = pd.get_dummies(df[["Sex", "Embarked"]])
    df = pd.concat([df, dummies], axis=1)
    return df


def fillna(df: pd.DataFrame) -> pd.DataFrame:
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
    clf = model(kwargs)
    cv = cross_val_score(clf, X, y, cv=5)
    print(cv)
    print(cv.mean())
    return clf


def create_submission(clf: VotingClassifier, df: pd.DataFrame) -> None:
    y_hat_base_vc = clf.predict(df).astype(int)
    basic_submission = {"PassengerId": df.PassengerId, "Survived": y_hat_base_vc}
    base_submission = pd.DataFrame(data=basic_submission)
    base_submission.to_csv("base_submission.csv", index=False)


def main():

    # Load
    train = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/train.csv")
    test = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/test.csv")

    # Preprocess
    train = scale_numeric_features(train)
    train = fillna(train)
    train = concat_dummies(train)
    train.drop(columns=["Ticket", "Name", "Cabin", "Embarked", "Sex"], inplace=True)
    test = scale_numeric_features(test)
    test = fillna(test)
    test = concat_dummies(test)
    test.drop(columns=["Ticket", "Name", "Cabin", "Embarked", "Sex"], inplace=True)

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

    create_submission(voting_clf, test)


if __name__ == "__main__":
    main()
