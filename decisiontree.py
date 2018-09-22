import pickle
from preprocessing import *
from random import randint
import random
import math
from f1_accuracy import calculate_macro_f1_score
import matplotlib.pyplot as plt

MAX_DEPTH = 18
LABELS_LEN = 10


class Node:

    def __init__(self, attr, attr_value, predicted_label):

        self.left = None
        self.right = None
        self.attr = attr
        self.attr_value = attr_value
        self.predicted_label = predicted_label


    def PrintTree(self):
        
        if self.left:
            self.left.PrintTree()
        print self.attr, " ", self.attr_value, " ", self.predicted_label
        if self.right:
            self.right.PrintTree()

def find_entropy(count_array, total_count):
    entropy = 0;
    sum_prob = 0
    if total_count <= 0:
        return entropy
    for i in range(LABELS_LEN):
        prob_i = count_array[i]/(1.0*total_count)
        if prob_i != 0:
            sum_prob += prob_i
            entropy -= prob_i * math.log(prob_i)
    # print entropy
    # print sum_prob
    return entropy

def calculate_entropy(labels):
    count = [0 for i in range(LABELS_LEN)]
    for i in labels:
        count[i] += 1
    return find_entropy(count, len(labels))

    
def count_left_branch(attr, split_value, features, labels):
    total_examples_left_branch = 0
    count = [0 for i in range(LABELS_LEN)]
    for i in range(len(labels)):
        if features[i][attr] < split_value:
            count[labels[i]] += 1
            total_examples_left_branch += 1
    return count, total_examples_left_branch

def count_right_branch(attr, split_value, features, labels):
    total_examples_right_branch = 0
    count = [0 for i in range(LABELS_LEN)]
    for i in range(len(labels)):
        if features[i][attr] >= split_value:
            count[labels[i]] += 1
            total_examples_right_branch += 1
    return count, total_examples_right_branch

def calculate_information_gain(attr, split_value, features, labels, entropy_sample):
    count_array_left, total_eg_left = count_left_branch(attr, split_value, features, labels)
    count_array_right, total_eg_right = count_right_branch(attr, split_value, features, labels)
    entropy_left = find_entropy(count_array_left, total_eg_left)
    entropy_right = find_entropy(count_array_right, total_eg_right)

    info_gain = entropy_sample - (entropy_left*total_eg_left + entropy_right*total_eg_right)/(total_eg_left+total_eg_right)
    # print info_gain
    return info_gain

def predict_label(labels):
    count = [0 for i in range(LABELS_LEN)]
    max_label=0
    max_count=0
    for label in labels:
        count[label] += 1
        if count[label] > max_count:
            max_count = count[label]
            max_label = label

    return max_label, 100*max_count/(1.0*len(labels))

def create_decision_tree(features, labels, depth):
    
    predicted_label, confidence = predict_label(labels)
    
    if depth > MAX_DEPTH or confidence == 100:
        # print predicted_label, " with confidence ", confidence, " depth is ", depth
        return Node(None, None, predicted_label)
    
    #implement logic to prune decision tree if no of samples are less than a threshold
    if len(labels) <= 10 and confidence >= 90:
        return Node(None, None, predicted_label)

    entropy_sample = calculate_entropy(labels)
    max_information_gain = -100
    max_split_attr = -100
    max_split_value = -100
    for attr in range(0, len(features[0])):
        split_times = 0
        split_value = 0
        while split_times < 150 and split_value <= 1000:
        # for i in range(100):
            split_times += 1
            split_value += 7
            # split_value = randint(0,1000)
            # split_value = features[randint(0,len(features)-1)][attr]
            # split_value = random.uniform(0,1)
            info_gain = calculate_information_gain(attr, split_value, features, labels, entropy_sample)
            if info_gain > max_information_gain:
                max_information_gain = info_gain
                max_split_attr = attr
                max_split_value = split_value
        # print max_information_gain, " attr ", max_split_attr, " split value is ", max_split_value

    left_features, left_labels = [], []
    right_features, right_labels = [], []
    for i in range(len(features)):
        if features[i][max_split_attr] < max_split_value:
            left_features.append(features[i])
            left_labels.append(labels[i])
        else:
            right_features.append(features[i])
            right_labels.append(labels[i])

    node = Node(max_split_attr, max_split_value, predicted_label)

    if len(left_features) > 0:
        node.left = create_decision_tree(left_features, left_labels, depth+1)
    if len(right_features) > 0:
        node.right = create_decision_tree(right_features, right_labels, depth+1)
    return node
        # break

def get_label_decision_tree(root, test_point):
    if root.attr is None:
        return root.predicted_label

    if test_point[root.attr] < root.attr_value:
        if root.left is not None:
            return get_label_decision_tree(root.left, test_point)
    if root.right is not None:
        return get_label_decision_tree(root.right, test_point)
    return root.predicted_label

def get_inference(root, test_points, test_labels):
    correct_points = 0
    predictions = []
    true_labels = []
    for i, test_point in enumerate(test_points):
        predicted_label = get_label_decision_tree(root, test_point)
        
        true_label = test_labels[i]
        if true_label == predicted_label:
            correct_points += 1
        predictions.append(predicted_label)
        true_labels.append(true_label)

    f1_score = calculate_macro_f1_score(predictions, true_labels)
    accuracy = correct_points*100.0/len(test_points)
    return f1_score, accuracy


def main():
    #shuffled the data and stored it into file shuffled_data_wine
    # uncomment the line below to work on original file
    # shuffle_data()
    with open ('shuffled_data_wine', 'rb') as fp:
        data = pickle.load(fp)

    #no shuffling data
    # with open('winequality-white.csv', 'rb') as csvfile:
    #     csvreader = csv.reader(csvfile, delimiter = ';')
    #     # print type(csvreader)
    #     data = []
    #     for i, row in enumerate(csvreader):
    #         if i != 0:
    #             row = [float(i) for i in row]
    #             data.append(row)
    
    
    # print len(data)
    no_folds = 4
    folds = divide_folds(no_folds, data)

    '''
    depth_axis and folds_axis are for plotting graph using matplotlib
    '''
    start_depth = 18
    end_depth = 18
    depth_axis = [i for i in range(start_depth,end_depth+1)]
    folds_axis = [[-100 for i in range(start_depth,end_depth+1)] for i in range(4)]

    for depth in range(start_depth, end_depth+1):
        global MAX_DEPTH
        MAX_DEPTH = depth
        print "Hyper-parameters:"
        print "Max-Depth: ", MAX_DEPTH
        print ""
        
        average_training_f1 = 0
        average_training_accuracy = 0
        average_validation_f1 = 0
        average_validation_accuracy = 0
        average_test_f1 = 0
        average_test_accuracy = 0
        for i in range(no_folds):
            print "Fold-", i+1, ":"
            train_data, test_data = split_folds_train_test(i, folds)
            # shuffle(train_data)
            train, validation = split_train_validation(train_data)
            minValue, maxValue = find_normalization_params(train_data);

            #feature engineering
            # for i in range(11):
            scale = 1000
            isInt = False
            ignored_feature = []
            train_norm, train_labels = normalize(train, minValue, maxValue, ignored_feature, scale, isInt)
            valid_norm, valid_labels = normalize(validation, minValue, maxValue, ignored_feature, scale, isInt)
            test_norm, test_labels = normalize(test_data, minValue, maxValue, ignored_feature, scale, isInt)

            '''
            create decision tree
            uncomment the below line to create decision tree
            '''
            # root = create_decision_tree(train_norm, train_labels,0)


            with open('decision_tree_depth' + str(depth) + '_fold' + str(i+1), 'wb') as fp:
                pickle.dump(root, fp)

            '''
            using the saved decision tree from pickle file
            uncommnent the below 2 lines if using decision tree from scratch
            '''
            # with open ('decision_tree_depth' + str(depth) + '_fold' + str(i+1), 'rb') as fp:
            #     root = pickle.load(fp)

            # root.PrintTree()
            f1_score, accuracy = get_inference(root, train_norm, train_labels)
            average_training_f1 += f1_score
            average_training_accuracy += accuracy
            print "Training: F1 Score: ", f1_score, ", Accuracy: ", accuracy

            f1_score, accuracy = get_inference(root, valid_norm, valid_labels)
            print "Validation: F1 Score: ", f1_score, ", Accuracy: ", accuracy
            average_validation_f1 += f1_score
            average_validation_accuracy += accuracy

            '''
            populate the y axis for plotting f1 score for validation data set in folds_axis
            '''
            folds_axis[i][depth-start_depth] = accuracy

            f1_score, accuracy = get_inference(root, test_norm, test_labels)
            print "Test: F1 Score: ", f1_score, ", Accuracy: ", accuracy
            average_test_f1 += f1_score
            average_test_accuracy += accuracy

            print ""
        print "Average:"
        print "Training: F1 Score: ", average_training_f1/4.0, ", Accuracy: ", average_training_accuracy/4.0
        print "Validation: F1 Score: ", average_validation_f1/4.0, ", Accuracy: ", average_validation_accuracy/4.0
        print "Test: F1 Score: ", average_test_f1/4.0, ", Accuracy: ", average_test_accuracy/4.0

    # plot on matplot lib
    # plot_graph(depth_axis, folds_axis)

def plot_graph(depth_axis, folds_axis):
    plt.plot(depth_axis, folds_axis[0], marker='*')
    plt.plot(depth_axis, folds_axis[1], marker='*')
    plt.plot(depth_axis, folds_axis[2], marker='*')
    plt.plot(depth_axis, folds_axis[3], marker='*')

    plt.legend(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'], loc='upper right')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy in validation set')
    plt.title('Decision Tree Accuracy v/s Max Depth')
    plt.show()       


if __name__ == '__main__':
    main()