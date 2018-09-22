import pickle
from distances import find_distance
from preprocessing import *
from f1_accuracy import calculate_macro_f1_score
# import matplotlib.pyplot as plt

def knn_output(test_point, train_points, train_labels, k, is_weighting):
    distance = []
    for i, point in enumerate(train_points):
        tup = (find_distance(test_point, point), train_labels[i])
        # print tup
        distance.append(tup)

    """
    Since the number of points is not that huge i.e. in the range of few thousands.
    Thus sorting the distances and picking the top k distances also has similar performance
    as using a heap to fetch top k distances. I am using sorting here.
    """
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

def get_results(train, train_labels, test, test_labels, k):
    count = 0
    correct = 0
    incorrect = []
    predictions = []
    true_labels = []
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

        predictions.append(predicted_label)
        true_labels.append(true_label)

    accuracy = 100.0*correct/count
    f1_score = calculate_macro_f1_score(predictions, true_labels)

    return f1_score, accuracy
    # print incorrect
        # break

def main():
    #shuffled the data and stored it into file shuffled_data_wine
    # uncomment the line below to work on original file
    # shuffle_data()
    
    with open ('shuffled_data_wine_knn', 'rb') as fp:
        data = pickle.load(fp)
    # print len(data)
    no_folds = 4

    """"
    The portion of code which divides into 4 folds for cross validation.
    The function divide_folds is in preprocessing.py file.
    """
    folds = divide_folds(no_folds, data)

    start_k = 12
    end_k = 12
    # folds_axis_f1 = [[-100 for i in range(start_k,end_k+1)] for i in range(1)]
    # folds_axis_accuracy = [[-100 for i in range(start_k,end_k+1)] for i in range(1)]

    """
    this for loop was used to get scores for a range of k values 
    and then select the best ones on the validation data set.
    """
    for k in range(start_k,end_k+1):
        print "Hyper-parameters:"
        print "K: ", k
        print "Distance measure: ", "Manhattan Distance"
        print ""
        average_validation_f1 = 0
        average_validation_accuracy = 0
        average_test_f1 = 0
        average_test_accuracy = 0
        for i in range(no_folds):
            print "Fold-", i+1, ":"
            """
            The function split_folds_train_test takes out the ith fold each time as the test_data which
            is not touched until after the model is tuned.
            """
            train_data, test_data = split_folds_train_test(i, folds)

            """
            The train_data is split into training and validation using the function spit_train_validation.
            The implementation of this function is in preprocessing.py
            """
            train, validation = split_train_validation(train_data)

            """
            finds the normalization params for the training data
            """
            minValue, maxValue = find_normalization_params(train_data);

            #feature engineering
            # for i in range(11):

            """
            scale and isInt were tuning parameters which allowed me to change the scale from 0 to 1 to any other scale i.e. 10, 100 etc.
            isInt is a flag to either trim the values of the features to floor of the double value. This is meaningful only if the scale is reasonably spread
            out i.e. 100 or 1000 otherwise there will be a lot of loss of information from feature values.
            """
            scale = 1
            isInt = False
            """
            I did feature engineering by using ignored_feature list. This list indicates the features which I ignore in my final model.
            The practice which I did was ignoring each feature one by one and seeing the impact on f1_score. This tells me the contribution of
            each feature in knowing the correct label. After this experiment, I found that ignoring feature 2 i.e. citric acid content gave me
            better results than those including it.
            """
            ignored_feature = [2]
            train_norm, train_labels = normalize(train, minValue, maxValue, ignored_feature, scale, isInt)
            valid_norm, valid_labels = normalize(validation, minValue, maxValue, ignored_feature, scale, isInt)
            test_norm, test_labels = normalize(test_data, minValue, maxValue, ignored_feature, scale, isInt)


            
            
            #validation data
            # print "k is ", k, "ignored feature is ", ignored_feature
            f1_score, accuracy = get_results(train_norm, train_labels, valid_norm, valid_labels, k)
            print "Validation: F1 Score: ", f1_score, ", Accuracy: ", accuracy
            average_validation_f1 += f1_score
            average_validation_accuracy += accuracy

            # folds_axis_f1[i][k-start_k] = f1_score
            

            #test data
            f1_score, accuracy = get_results(train_norm, train_labels, test_norm, test_labels, k)
            print "Test: F1 Score: ", f1_score, ", Accuracy: ", accuracy
            average_test_f1 += f1_score
            average_test_accuracy += accuracy
            print ""

        print "Average:"
        print "Validation: F1 Score: ", average_validation_f1/4.0, ", Accuracy: ", average_validation_accuracy/4.0
        print "Test: F1 Score: ", average_test_f1/4.0, ", Accuracy: ", average_test_accuracy/4.0
        # folds_axis_accuracy[0][k-start_k] = accuracy
        # folds_axis_f1[0][k-start_k] = f1_score
    
    # with open('knn_cosine_folds_axis_accuracy', 'wb') as fp:
    #             pickle.dump(folds_axis_accuracy, fp)

    # with open('knn_cosine_folds_axis_f1', 'wb') as fp:
    #             pickle.dump(folds_axis_f1, fp)
    

    '''
    Code to plot the matplotlib graph
    '''
    # k_axis = [i for i in range(2,14+1)]
    # folds_axis_f1 = []
    # folds_axis_accuracy = []
    # with open ('knn_cosine_folds_axis_f1', 'rb') as fp:
    #     folds_axis_f1.append(pickle.load(fp))
    # with open ('knn_manhattan_folds_axis_f1', 'rb') as fp:
    #     folds_axis_f1.append(pickle.load(fp))
    # with open ('knn_euclidean_folds_axis_f1', 'rb') as fp:
    #     folds_axis_f1.append(pickle.load(fp))


    # with open ('knn_cosine_folds_axis_accuracy', 'rb') as fp:
    #     folds_axis_accuracy.append(pickle.load(fp))
    # with open ('knn_manhattan_folds_axis_accuracy', 'rb') as fp:
    #     folds_axis_accuracy.append(pickle.load(fp))
    # with open ('knn_euclidean_folds_axis_accuracy', 'rb') as fp:
    #     folds_axis_accuracy.append(pickle.load(fp))

    # # print len(folds_axis_f1[0][0])

    # # print "dump complete for knn cosine"
    # plot_graph(k_axis, folds_axis_f1, folds_axis_accuracy)

def plot_graph(depth_axis, folds_axis_f1, folds_axis_accuracy):
    plt.plot(depth_axis, folds_axis_f1[0][0], marker='*')
    plt.plot(depth_axis, folds_axis_f1[1][0], marker='*')
    plt.plot(depth_axis, folds_axis_f1[2][0], marker='*')
    # plt.plot(depth_axis, folds_axis_f1[3], marker='*')

    plt.legend(['Cosine Distance', 'Manhattan Distance', 'Euclidean Distance'], loc='upper right')
    plt.xlabel('Value of k')
    plt.ylabel('%f1 score in validation set')
    plt.title('KNN F1 Score v/s k')
    plt.show()


    plt.plot(depth_axis, folds_axis_accuracy[0][0], marker='*')
    plt.plot(depth_axis, folds_axis_accuracy[1][0], marker='*')
    plt.plot(depth_axis, folds_axis_accuracy[2][0], marker='*')
    # plt.plot(depth_axis, folds_axis_accuracy[3], marker='*')

    plt.legend(['Cosine Distance', 'Manhattan Distance', 'Euclidean Distance'], loc='upper right')
    plt.xlabel('Value of k')
    plt.ylabel('Accuracy score in validation set')
    plt.title('KNN Accuracy v/s k')
    plt.show()

if __name__ == '__main__':
    main()