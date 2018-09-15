import csv
import math
from random import shuffle
import pickle

def euclidean_distance(point1, point2):
    sum = 0;
    for i in range(len(point1)):
        sum += (point1[i]-point2[i])*(point1[i]-point2[i])

    sum = math.sqrt(sum)
    return sum

def cosine_distance(point1, point2):
    x = 0;
    y = 0;
    xy = 0;
    for i in range(len(point1)):
        xy += point1[i]*point2[i]
        x += point1[i]*point1[i]
        y += point2[i]*point2[i]
    ans = xy/(math.sqrt(x*y))
    return ans

def manhattan_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += abs(point1[i]-point2[i])
    return distance

def find_distance(point1, point2):
    distance = euclidean_distance(point1, point2)
    # distance = manhattan_distance(point1, point2)
    # distance = cosine_distance(point1, point2)
    # print distance
    return distance

def divide_folds(k, data):
    if k < 0:
        return [data]

    fold_size = len(data)/k
    folds = [data[x:x+fold_size] for x in xrange(0, len(data), fold_size)]
    if len(folds) == 5:
        folds[3] += folds[4]
        del folds[4]
    return folds

def split_folds(test_index, folds):
    test_fold = folds[test_index]
    train_fold = []
    for x in range(len(folds)):
        if x != test_index:
            train_fold += folds[x]

    return train_fold, test_fold

def split_train(train_data):
    validation_len = len(train_data)/5
    validation = train_data[:validation_len]
    train = train_data[validation_len:]
    # print validation[validation_len-1]
    # print train[0]
    return train, validation

def findNormParams(train_data):
    minValue = [x for x in train_data[0]]
    maxValue = [x for x in train_data[0]]
    for example in train_data:
        for i in range(len(example)):
            if example[i]>maxValue[i]:
                # print "max ", example[i], " ", maxValue[i]
                # break
                maxValue[i]=example[i]
            if example[i]<minValue[i]:
                minValue[i]=example[i]

    return minValue, maxValue

def min_max_norm(value, minValue, maxValue, scale):
    value = (value-minValue)/(maxValue-minValue)*scale
    if value < 0:
        value = 0
    if value > scale:
        value = scale
    return value


def normalize(data, minValue, maxValue, ignored_feature):
    features, labels = [],[]
    for example in data:
        normalized_example = []
        for j in range(len(minValue)-1):
            if j not in ignored_feature:
                normalized_example.append(min_max_norm(example[j], minValue[j], maxValue[j], 1))
        labels.append(int(example[len(minValue)-1]))
        features.append(normalized_example)

    return features, labels



def knn_output(test_point, train_points, train_labels, k, is_weighting):
    distance = []
    for i, point in enumerate(train_points):
        tup = (find_distance(test_point, point), train_labels[i])
        # print tup
        distance.append(tup)
    distance.sort(key=lambda tup: tup[0])
    # print distance[:k]
    label = [0 for x in range(10)]
    for i in range(k):
        if is_weighting:
            if distance[i][0] == 0:
                label[distance[i][1]] += float("inf")
            else:
                label[distance[i][1]] += 1.0/distance[i][0]
        else:
            label[distance[i][1]] += 1
    maxValue = 0
    predicted_label = 0
    predicted_label_2 = 0
    max_value_2 = 0
    # print label
    for i in range(10):
        if label[i] > maxValue:
            max_value_2 = maxValue
            predicted_label_2 = predicted_label
            maxValue = label[i]
            predicted_label = i
        elif label[i] > max_value_2:
            max_value_2 = label[i]
            predicted_label_2 = i

    return predicted_label, predicted_label_2


def shuffle_data():
    with open('winequality-white.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ';')
        # print type(csvreader)
        data = []
        for i, row in enumerate(csvreader):
            if i != 0:
                row = [float(i) for i in row]
                data.append(row)
    shuffle(data)
    with open('shuffled_data_wine', 'wb') as fp:
        pickle.dump(data, fp)

def main():

    #shuffled the data and stored it into file shuffled_data_wine
    # uncomment the line below to work on original file
    # shuffle_data()
    
    with open ('shuffled_data_wine', 'rb') as fp:
        data = pickle.load(fp)
    # print len(data)
    folds = divide_folds(4, data)
    
    for i in range(4):
        # print len(folds)
        train_data, test_data = split_folds(i, folds)
        # print len(train_data)
        # print len(test_data)
        # shuffle(train_data)
        # print train_data[0]
        train, validation = split_train(train_data)
        # print len(train)
        # print len(validation)
        minValue, maxValue = findNormParams(train_data);
        # print "min value is ", minValue
        # print "max value is ", maxValue
        # print "no of features are ", len(minValue)-1

        

        #feature engineering
        # for i in range(11):
        ignored_feature = [2]
        train_norm, train_labels = normalize(train, minValue, maxValue, ignored_feature)
        valid_norm, valid_labels = normalize(validation, minValue, maxValue, ignored_feature)
        test_norm, test_labels = normalize(test_data, minValue, maxValue, ignored_feature)
        # print norm_train[20]
        # print validation_labels[25]
        # print valid_norm[25]
        # print train_data[1225], " ", test_data[0]

        for k in range(12,13):
            #validation data
            # print "k is ", k, "ignored feature is ", ignored_feature
            print "accuracy for validation ", get_results(train_norm, train_labels, valid_norm, valid_labels, k)

            #test data
            print "accuracy for test ", get_results(train_norm, train_labels, test_norm, test_labels, k)
        


    # print len(folds[0])

def get_results(train, train_labels, test, test_labels, k):
    count = 0
    correct = 0
    incorrect = []
    for i, test_point in enumerate(test):
        count += 1
        true_label = test_labels[i]
        is_weighting = True
        predicted_label, predicted_label_2 = knn_output(test_point, train, train_labels, k, is_weighting)
        # print predicted_label, " ", true_label
        if predicted_label == true_label:
            correct += 1
        else:
            incorrect.append((predicted_label, true_label))
        # break

    return 100.0*correct/count
    # print incorrect
        # break

if __name__ == '__main__':
    main()