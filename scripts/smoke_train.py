from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from firstdataset.data import split_qsar_biodegradation


def main() -> None:
    split = split_qsar_biodegradation()
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(split.X_train, split.y_train)
    predictions = model.predict(split.X_test)
    print(classification_report(split.y_test, predictions))


if __name__ == "__main__":
    main()
