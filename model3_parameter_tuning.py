# Differences:
# Used RobustScaler for "Fare"
# Removed cabin alphabet
# Used GridSearchCV to tune parameters

# Import basics
import pandas as pd

# Import estimators
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# Import GridSearch for parameter optimization
from sklearn.model_selection import GridSearchCV


# Import scaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.svm import SVC


def scale_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing the features in the DataFrame using scalers for better model predictions
    Uses RobustScaler for "Fare" and MinMaxScaler for other numeric columns

    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with processed columns
    """
    df["Fare"] = RobustScaler().fit_transform(df[["Fare"]])
    df["Age"] = MinMaxScaler().fit_transform(df[["Age"]])
    df["Pclass"] = MinMaxScaler().fit_transform(df[["Pclass"]])
    df["Parch"] = MinMaxScaler().fit_transform(df[["Parch"]])
    df["SibSp"] = MinMaxScaler().fit_transform(df[["SibSp"]])
    return df


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
    """Fill na values in "Age" and "Fare"
    Args:
        df (pd.DataFrame): train and test

    Returns:
        pd.DataFrame: train and test with no na values
    """
    df["Age"].fillna(df.Age.mean(), inplace=True)
    df["Fare"].fillna(df.Fare.median(), inplace=True)
    return df


def lr_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for LogisticRegression

    Args:
        X (pd.DataFrame): X_train
        y (pd.Series): y_train

    Returns:
        object: lr which is LogisticRegression with the optimized parameters found from GridSearchCV
    """
    lr = (
        GridSearchCV(
            estimator=LogisticRegression(),
            param_grid={
                "max_iter": [1000, 2000, 3000, 4000],
                "C": [x for x in range(1, 5, 1)],
                "random_state": [1, 2, 3, 4, 5],
            },
            cv=5,
            verbose=True,
        )
        .fit(X, y)
        .best_estimator_
    )
    return lr


def knn_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for KNeighborsClassifier

    Args:
        X (pd.DataFrame): X_train
        y (pd.Series): y_train

    Returns:
        object: knn which is LogisticRegression with the optimized parameters found from GridSearchCV
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
        )
        .fit(X, y)
        .best_estimator_
    )
    return knn


def dt_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for tree.DecisionTreeClassifier

    Args:
        X (pd.DataFrame): X_train
        y (pd.Series): y_train

    Returns:
        object: dt which is LogisticRegression with the optimized parameters found from GridSearchCV
    """
    dt = (
        GridSearchCV(
            estimator=tree.DecisionTreeClassifier(),
            param_grid={
                "splitter": ["random"],
                "min_samples_split": [1, 2, 3],
                "min_samples_leaf": [3, 4, 5, 6],
                "max_features": ["auto", "sqrt", "log2"],
                "random_state": [2, 3, 4, 5],
                "max_leaf_nodes": [4, 5, 6],
                "class_weight": ["balanced"],
            },
            cv=5,
            verbose=True,
            n_jobs=-1,
        )
        .fit(X, y)
        .best_estimator_
    )
    return dt


def svc_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for SVC

    Args:
        X (pd.DataFrame): X_train
        y (pd.Series): y_train

    Returns:
        object: svc which is LogisticRegression with the optimized parameters found from GridSearchCV
    """
    svc = (
        GridSearchCV(
            estimator=SVC(),
            param_grid={
                "C": [1, 2, 3, 4, 5],
                "degree": [1, 2, 3, 4, 5],
                "shrinking": [True, False],
                "max_iter": [-3, -2, -1, 1, 2, 3],
                "decision_function_shape": ["ovo", "ovr"],
                "kernel": ["lienar", "poly", "rbf", "sigmoid"],
            },
            cv=5,
            verbose=True,
            n_jobs=-1,
        )
        .fit(X, y)
        .fit(X, y)
        .best_estimator_
    )
    return svc


def rf_parameters(X: pd.DataFrame, y: pd.Series) -> object:
    """Uses GridSearchCV to find the best parameters for RandomForestClassifier

    Args:
        X (pd.DataFrame): X_train
        y (pd.Series): y_train

    Returns:
        object: rf which is LogisticRegression with the optimized parameters found from GridSearchCV
    """
    rf = (
        GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid={
                "n_estimators": [50, 100, 150, 200, 300],
                "criterion": ["gini", "entropy"],
                "min_samples_split": [1, 2, 3, 4, 5],
                "min_samples_leaf": [1, 2, 3, 4, 5],
                "max_leaf_nodes": [1, 2, 3, 4, 5],
                "random_state": [1, 2, 3],
            },
            cv=5,
            verbose=True,
            n_jobs=-1,
        )
        .fit(X, y)
        .best_estimator_
    )
    return rf


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
    test.drop(columns=("PassengerId"), inplace=True)

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
    create_submission(voting_clf, test)


if __name__ == "__main__":
    main()
