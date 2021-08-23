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


def lr_parameters(X: pd.DataFrame, y: pd.Series) -> dict:
    GridSearchCV(
        estimator=LogisticRegression(),
        param_grid={
            "max_iter": [1000, 2000, 3000, 4000],
            "C": [x for x in range(1, 5, 1)],
            "random_state": [1, 2, 3, 4, 5],
        },
        cv=5,
        verbose=True,
    ).fit(X, y).best_params_


def knn_parameters(X: pd.DataFrame, y: pd.Series) -> dict:
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
    ).fit(X, y).best_params_


def dt_parameters(X: pd.DataFrame, y: pd.Series) -> dict:
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
    ).fit(X, y).best_params_


def svc_parameters(X: pd.DataFrame, y: pd.Series) -> dict:
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
    ).fit(X, y).best_params_


def rf_parameters(X: pd.DataFrame, y: pd.Series) -> dict:
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
    ).fit(X, y).best_params_


def main():

    # Load
    train = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/train.csv")
    test = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/test.csv")

    # Preprocess
    train = scale_numeric_features(train)
    train = add_occ(train)
    train = fillna(train)
    train = concat_dummies(train)
    train.drop(
        columns=["Ticket", "Name", "Cabin", "Embarked", "Sex", "cabin_name"],
        inplace=True,
    )
    test = scale_numeric_features(test)
    test = add_occ(test)
    test = fillna(test)
    test = concat_dummies(test)
    test.drop(
        columns=["Ticket", "Name", "Cabin", "Embarked", "Sex", "cabin_name"],
        inplace=True,
    )

    # Split X & y
    X_train = train.drop(columns="Survived").drop(columns="PassengerId")
    y_train = train["Survived"]

    # Modeling & CV
    lr = modeling_cv(X_train, y_train, LogisticRegression)
    gnb = modeling_cv(X_train, y_train, GaussianNB)
    dt = modeling_cv(X_train, y_train, tree.DecisionTreeClassifier, random_state=1)
    knn = modeling_cv(X_train, y_train, KNeighborsClassifier)
    rf = modeling_cv(X_train, y_train, RandomForestClassifier, random_state=1)
    svc = modeling_cv(X_train, y_train, SVC, probability=True)

    # Model Tuning
    lr_parameters(X_train, y_train)
    knn_parameters(X_train, y_train)
    dt_parameters(X_train, y_train)
    svc_parameters(X_train, y_train)
    rf_parameters(X_train, y_train)

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
