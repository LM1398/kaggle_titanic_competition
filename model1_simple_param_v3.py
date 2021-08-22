def main():

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns

    titanic = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/train.csv")

    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    titanic["Fare"] = StandardScaler().fit_transform(titanic[["Fare"]])
    titanic["Age"] = MinMaxScaler().fit_transform(titanic[["Age"]])
    titanic["Pclass"] = MinMaxScaler().fit_transform(titanic[["Pclass"]])
    titanic["Parch"] = MinMaxScaler().fit_transform(titanic[["Parch"]])
    titanic["SibSp"] = MinMaxScaler().fit_transform(titanic[["SibSp"]])

    titanic = titanic.dropna(subset=["Age"])
    titanic = titanic.dropna(subset=["Embarked"])

    dummies = pd.get_dummies(titanic[["Sex", "Embarked"]])
    titanic = pd.concat([titanic, dummies], axis=1)

    titanic.drop(columns=["Ticket", "Name", "Cabin", "Embarked", "Sex"], inplace=True)

    test = pd.read_csv("/Users/leo/samurai/kaggle/titanic/data/test.csv")

    test["Fare"] = StandardScaler().fit_transform(test[["Fare"]])
    test["Age"] = MinMaxScaler().fit_transform(test[["Age"]])
    test["Pclass"] = MinMaxScaler().fit_transform(test[["Pclass"]])
    test["Parch"] = MinMaxScaler().fit_transform(test[["Parch"]])
    test["SibSp"] = MinMaxScaler().fit_transform(test[["SibSp"]])

    test["Age"].fillna(X_test.Age.mean(), inplace=True)
    test["Fare"].fillna(X_test.Fare.median(), inplace=True)

    dummies2 = pd.get_dummies(test[["Sex", "Embarked"]])
    test = pd.concat([test, dummies2], axis=1)

    test.drop(columns=["Ticket", "Name", "Cabin", "Embarked", "Sex"], inplace=True)

    X_train = titanic.drop(columns="Survived").drop(columns="PassengerId")
    y_train = titanic["Survived"]
    X_test = test.drop(columns="PassengerId")

    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    lr = LogisticRegression(max_iter=2000,)
    cv = cross_val_score(lr, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    gnb = GaussianNB()
    cv = cross_val_score(gnb, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    dt = tree.DecisionTreeClassifier(random_state=1)
    cv = cross_val_score(dt, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    knn = KNeighborsClassifier()
    cv = cross_val_score(knn, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    rf = RandomForestClassifier(random_state=1)
    cv = cross_val_score(rf, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    svc = SVC(probability=True)
    cv = cross_val_score(svc, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    from sklearn.ensemble import VotingClassifier

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

    from sklearn.model_selection import GridSearchCV

    mod = GridSearchCV(
        estimator=voting_clf,
        param_grid={"max_iter": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]},
        cv=5,
    )
    mod.fit(X_train, y_train)

    cv = cross_val_score(voting_clf, X_train, y_train, cv=5)
    print(cv)
    print(cv.mean())

    voting_clf.fit(X_train, y_train)
    y_hat_base_vc = voting_clf.predict(X_test).astype(int)
    basic_submission = {"PassengerId": test.PassengerId, "Survived": y_hat_base_vc}
    base_submission = pd.DataFrame(data=basic_submission)
    base_submission.to_csv("base_submission.csv", index=False)

if __name__ == "__main__":
    main()