    """Scripts for titanic kaggle competition. Differences from previous models: Added family and is_alone column, Less model tuning,
    Changed preprocess from for loop to previous set up because it kept returnig errors, Drops SibSp and Parch.

    Returns:
        csv: csv file with survival prediction for test dataset.
    """


# Import basics

import numpy as np
import pandas as pd

# Import estimators
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# Import GridSearch for parameter optimization
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Import scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.svm import SVC


def add_family(df: pd.DataFrame) -> pd.DataFrame:
    """Adds column family to classify people if people have family.
    
    Args:
        df(pd.DataFrame): train and test.
    
    Returns:
        df(pd.DataFrame): train and test with family.
    """
    df["family"] = np.add([x for x in df["SibSp"]], [x for x in df["Parch"]])
    return df


def add_is_alone(df: pd.DataFrame) -> pd.DataFrame:
    """Adds column alone to classify people who don't have family.
    
    Args:
        df(pd.DataFrame): train and test.
    
    Returns:
        df(pd.DataFrame): train and test with is_alone.
    """
    df["is_alone"] = [1 if x == 0 else 0 for x in df["family"]]
    return df


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing the features in the DataFrame using scalers for better model predictions.

    Args:
        df (pd.DataFrame): train and test.

    Returns:
        pd.DataFrame: train and test with processed columns.
    """
    sex = []
    for x in df["Sex"].values:
        if x == "male":
            sex.append(1)
        else:
            sex.append(0)
    df["Sex"] = sex
    #     df['TicketHeadCount']=df['Ticket'].map(df['Ticket'].value_counts())
    #     df.Fare = df.Fare / df.TicketHeadCount
    df["Fare"] = RobustScaler().fit_transform(df[["Fare"]])
    df["Fare"] = StandardScaler().fit_transform(df[["Fare"]])
    df["Age"] = MinMaxScaler().fit_transform(df[["Age"]])

    return df


def concat_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """Creates dummies for "Sex" and "Embarked" to have only numeric data in the DataFrame.

    Args:
        df (pd.DataFrame): train and test.

    Returns:
        pd.DataFrame: train and test with dummy columns for "Sex" and "Embarked".
    """
    dummies = pd.get_dummies(df[["Embarked"]])
    df = pd.concat([df, dummies], axis=1)
    return df


def fillna(df: pd.DataFrame) -> pd.DataFrame:
    """Fill na values in "Age" and "Fare".
    Args:
        df (pd.DataFrame): train and test.

    Returns:
        pd.DataFrame: train and test with no na values.
    """
    df["Age"].fillna(df.Age.median(), inplace=True)
    df["Fare"].fillna(df.Fare.mean(), inplace=True)
    return df


def lr_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for LogisticRegression.

    Args:
        X (pd.DataFrame): X_train.
        y (pd.Series): y_train.

    Returns:
        object: lr which is LogisticRegression with the optimized parameters found from GridSearchCV.
    """
    lr = (
        GridSearchCV(
            estimator=LogisticRegression(),
            param_grid={"C": [x for x in range(1, 8, 1)], "random_state": [3],},
            cv=5,
            verbose=True,
            scoring="accuracy",
        )
        .fit(X, y)
        .best_estimator_
    )
    return lr


def knn_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for KNeighborsClassifier.

    Args:
        X (pd.DataFrame): X_train.
        y (pd.Series): y_train.

    Returns:
        object: knn which is LogisticRegression with the optimized parameters found from GridSearchCV.
    """
    knn = (
        GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid={
                "n_neighbors": [4, 5, 6, 7],
                "leaf_size": [x for x in range(5, 50, 5)],
                "p": [1, 2],
                "weights": ["uniform", "distance"],
            },
            cv=5,
            verbose=True,
            scoring="accuracy",
        )
        .fit(X, y)
        .best_estimator_
    )
    return knn


def dt_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for tree.DecisionTreeClassifier.

    Args:
        X (pd.DataFrame): X_train.
        y (pd.Series): y_train.

    Returns:
        object: dt which is LogisticRegression with the optimized parameters found from GridSearchCV.
    """
    dt = (
        GridSearchCV(
            estimator=tree.DecisionTreeClassifier(),
            param_grid={
                "splitter": ["random"],
                "min_samples_leaf": [2, 3, 4, 5, 6],
                "max_features": ["auto", "sqrt", "log2"],
                "random_state": [3],
            },
            cv=5,
            verbose=True,
            n_jobs=-1,
            scoring="accuracy",
        )
        .fit(X, y)
        .best_estimator_
    )
    return dt


def svc_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for SVC.

    Args:
        X (pd.DataFrame): X_train.
        y (pd.Series): y_train.

    Returns:
        object: svc which is LogisticRegression with the optimized parameters found from GridSearchCV.
    """
    svc = (
        GridSearchCV(
            estimator=SVC(probability=True),
            param_grid={
                "C": [2, 3, 4, 5, 6],
                "shrinking": [True, False],
                "decision_function_shape": ["ovo", "ovr"],
            },
            cv=5,
            verbose=True,
            n_jobs=-1,
            scoring="accuracy",
        )
        .fit(X, y)
        .best_estimator_
    )
    return svc


def rf_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for RandomForestClassifier.

    Args:
        X (pd.DataFrame): X_train.
        y (pd.Series): y_train.

    Returns:
        object: rf which is LogisticRegression with the optimized parameters found from GridSearchCV.
    """
    rf = (
        GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid={
                "n_estimators": [50, 100, 150, 200, 300],
                "criterion": ["gini", "entropy"],
                "min_samples_leaf": [2, 3, 4, 5],
                "random_state": [3],
            },
            cv=5,
            verbose=True,
            n_jobs=-1,
            scoring="accuracy",
        )
        .fit(X, y)
        .best_estimator_
    )
    return rf


def create_submission(clf: VotingClassifier, df: pd.DataFrame) -> None:
    """Uses the VotingClassifier to predict the Survival of the individuals in the train set.
    Saves a .csv file in the same directory for kaggle submission.

    Args:
        clf (VotingClassifier): [description]
        df (pd.DataFrame): [description]
    """
    test = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/test.csv")
    train_survived = clf.predict(df).astype(int)
    basic_submission = {"PassengerId": test.PassengerId, "Survived": train_survived}
    base_submission = pd.DataFrame(data=basic_submission)
    base_submission.to_csv("base_submission21.csv", index=False)


def main():

    # Load
    train = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/train.csv")
    test = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/test.csv")

    # Preprocess

    #     train = drop_fare_outliers(train)
    train = add_family(train)
    train = add_is_alone(train)
    train = scale_numeric_features(train)
    train = fillna(train)
    train = concat_dummies(train)
    train.drop(
        columns=["Ticket", "Name", "Cabin", "Embarked", "SibSp", "Parch",], inplace=True
    )

    test = add_family(test)
    test = add_is_alone(test)
    test = scale_numeric_features(test)
    test = fillna(test)
    test = concat_dummies(test)
    test.drop(
        columns=["Ticket", "Name", "Cabin", "Embarked", "SibSp", "Parch",], inplace=True
    )

    # Split X & y
    X_train = train.drop(columns="Survived").drop(columns="PassengerId")
    y_train = train["Survived"]
    X_test = test.drop(columns="PassengerId")

    print(X_train.describe(), X_test.describe())

    # Model Tuning
    lr = lr_parameters(X_train, y_train)
    knn = knn_parameters(X_train, y_train)
    dt = dt_parameters(X_train, y_train)
    svc = svc_parameters(X_train, y_train)
    rf = rf_parameters(X_train, y_train)

    # Ensemble
    voting_clf = VotingClassifier(
        estimators=[("lr", lr), ("knn", knn), ("rf", rf), ("svc", svc), ("dt", dt),],
        voting="soft",
    )
    cv = cross_val_score(voting_clf, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())
    voting_clf.fit(X_train, y_train)

    # Create .csv file to submit
    create_submission(voting_clf, X_test)
    print("Your file has been saved!")


if __name__ == "__main__":
    main()
