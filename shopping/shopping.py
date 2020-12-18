import csv
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    load_data(sys.argv[1])
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    """

    month = {'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'June': 5, 'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9,
             'Nov': 10, 'Dec': 11}
    weekend = {'FALSE': 0, 'TRUE': 1}
    revenue = {'FALSE': 0, 'TRUE': 1}

    with open('shopping.csv', mode='r',newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        evidence =[]
        labels =[]

        next(csv_reader)

        for row in csv_reader:
            row[0] = int(row[0])
            row[1] = float (row[1])
            row[2] = int(row[2])
            row[3] = float(row[3])
            row[4] = int(row[4])
            row[5:10] = [float(row[i]) for i in range(5,10)]

            row[10] = month[row[10]]

            if row[15] == 'Returning_Visitor':
                row[15] = 0
            else:
                row[15] = 1

            row[16] = weekend[row[16]]
            row[17] = revenue[row[17]]

            evidence.append(row[:17])

            labels.append(row[17])

    evidence = np.array(evidence).astype(np.float32)
    labels = np.array(labels).astype(np.float32)

    return (evidence,labels)



def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(evidence,labels)

    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    """
    tn, fp, fn, tp = confusion_matrix(labels,predictions).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    return (sensitivity, specificity)

if __name__ == "__main__":
    main()
