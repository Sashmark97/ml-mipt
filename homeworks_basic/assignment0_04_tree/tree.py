import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005

    # YOUR CODE HERE
    '''
    entropy = 0
    classes_set = set()
    decoded = one_hot_decode(y)
    for x in decoded:
        classes_set.add(x[0])

    for class_sample in classes_set:
        count = 0
        for sample in decoded:
            if sample == class_sample:
                count += 1
        probability = count / len(y)
        entropy -= probability * np.log(probability + EPS)

    return entropy
    '''
    probas = np.mean(y, axis=0)

    return -np.sum(probas * np.log(probas + EPS))


def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    '''
    classes_set = set()
    decoded = one_hot_decode(y)
    for x in decoded:
        classes_set.add(x[0])

    gini = 1.0
    for class_sample in classes_set:
        count = 0
        for sample in decoded:
            if sample[0] == class_sample:
                count += 1
        gini -= (count / len(y)) ** 2
    return gini
    '''
    probas = np.mean(y, axis=0)

    return 1 - np.sum(probas ** 2)


def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    
    return 0.

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    
    return 0.


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE

        features = X_subset[..., feature_index]
        mapping = features < threshold
        X_left = X_subset[mapping]
        X_right = X_subset[np.invert(mapping)]

        y_left = y_subset[mapping]
        y_right = y_subset[np.invert(mapping)]
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        features = X_subset[..., feature_index]
        mapping = features < threshold

        y_left = y_subset[mapping]
        y_right = y_subset[np.invert(mapping)]
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Greedily select the best feature and best threshold w.r.t. selected criterion
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        """
        # YOUR CODE HERE
        best_threshold = 0
        best_feature = 0
        min_criterion = np.inf
        current_uncertanty = self.criterion(y_subset)
        for feature_id in range(X_subset.shape[1]):
            for treshold in np.unique(X_subset[..., feature_id]):
                y_left, y_right = self.make_split_only_y(feature_id, treshold, X_subset, y_subset)
                prob = len(y_left) / len(X_subset)
                weighted_criterion = current_uncertanty - prob * self.criterion(y_left) - \
                                     (1 - prob) * self.criterion(y_right)
                if weighted_criterion < min_criterion:
                    best_feature = feature_id
                    best_threshold = treshold
                    min_criterion = weighted_criterion
        return best_feature, best_threshold
    
    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree
        
        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels or target values for corresponding subset
        
        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        new_node = Node(feature_index, threshold)
        self.depth += 1

        if self.depth < self.max_depth and X_subset.shape[0] >= self.min_samples_split:
            (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold,
                                                                   X_subset, y_subset)
            new_node.left_child = self.make_tree(X_left, y_left)
            new_node.right_child = self.make_tree(X_right, y_right)
        else:
            if self.classification:
                new_node.value = np.argmax(np.sum(y_subset, axis=0))
                new_node.probas = np.mean(y_subset, axis=0)
            elif self.criterion_name == 'variance':
                self.value = np.mean(y_subset)
            else:
                self.value = np.median(y_subset)

        self.depth -= 1
        return new_node

        
    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification 
                   of type float with shape (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)

    def recursive_predict(self, node, X, indices, y_predicted, probas=False):
        if node.left_child is None:
            if probas:
                y_predicted[indices] = node.probas
            else:
                y_predicted[indices] = node.value
        else:
            (X_left, left_indices), (X_right, right_indices) = self.make_split(node.feature_index,
                                                                               node.value, X,
                                                                               indices)
            self.recursive_predict(node.left_child, X_left, left_indices, y_predicted, probas)
            self.recursive_predict(node.right_child, X_right, right_indices, y_predicted, probas)

    def predict(self, X):
        """
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        """

        # YOUR CODE HERE
        n_objects = X.shape[0]
        y_predicted = np.zeros(n_objects)
        self.recursive_predict(self.root, X, np.arange(n_objects), y_predicted)

        return y_predicted
        
    def predict_proba(self, X):
        """
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        """
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        n_objects = X.shape[0]
        y_predicted_probs = np.zeros((n_objects, self.n_classes))
        self.recursive_predict(self.root, X, np.arange(n_objects), y_predicted_probs, probas=True)

        return y_predicted_probs
