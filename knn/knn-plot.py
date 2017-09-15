import argparse
from collections import defaultdict

import numpy
from sklearn.neighbors import BallTree


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another datastructure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y value for
        # these indices
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

        # If only one neighbor, the majority is definitely its label
        if len(item_indices) == 1:
            return self._y[item_indices[0]]

        labels = [self._y[x] for x in item_indices]

        u = numpy.unique(labels, return_counts=True)  # tuple, first element is the labels, second element is the count

        sorted_counts = numpy.argsort(u[1])  # last element (i.e., sorted_counts[-1]) is the index of highest, etc

        majority_labels = []

        for i in range(len(sorted_counts)):
            if u[1][sorted_counts[i]] == u[1][sorted_counts[-1]]:
                majority_labels.append(u[0][sorted_counts[i]])

        return numpy.median(majority_labels)

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the value.

        dist, ind = self._kdtree.query([example], self._k)

        return self.majority(ind[0])

        # return self.majority(list(random.randrange(len(self._y)) \
        #                           for x in range(self._k)))

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            data_index += 1
            our_label = self.classify(xx)
            d[yy][our_label] = d.get(yy, {}).get(our_label, 0) + 1
            # if data_index % 100 == 0:
            #     print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


def get_accuracy_per_training_points():
    for limit in range(10, 500, 10):
        knn = Knearest(data.train_x[:limit], data.train_y[:limit], args.k)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        accuracy = knn.accuracy(confusion)
        print (limit, "\t", accuracy)


def get_accuracy_per_k():
    limit = 500

    for k in range(1, 30):
        knn = Knearest(data.train_x[:limit], data.train_y[:limit], k)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        accuracy = knn.accuracy(confusion)
        print (k, "\t", accuracy)


def get_accuracy_per_k_large():
    limit = 500

    for k in range(1, 500, 5):
        knn = Knearest(data.train_x[:limit], data.train_y[:limit], k)
        confusion = knn.confusion_matrix(data.test_x, data.test_y)
        accuracy = knn.accuracy(confusion)
        print (k, "\t", accuracy)


def get_confusion():
    knn = Knearest(data.train_x[:500], data.train_y[:500], 3)

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("\t".join(str(confusion[ii].get(x, 0))
                                       for x in range(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    # get_accuracy_per_training_points()
    # get_accuracy_per_k_large()
    get_confusion()