import matplotlib.pyplot as plt
from random import randint

def main():
    depth_axis = [i for i in range(4,25)]
    folds_axis = [[-100 for i in range(4,25)] for i in range(4)]
    for depth in range(4,25):
        # fold_points = []
        for i in range(4):
            folds_axis[i][depth-4] = randint(0,100)


    # print depth_axis
    # print folds_axis[0]
    # print folds_axis[1]
    plt.plot(depth_axis, folds_axis[0], marker='*')
    plt.plot(depth_axis, folds_axis[1], marker='*')
    plt.plot(depth_axis, folds_axis[2], marker='*')
    plt.plot(depth_axis, folds_axis[3], marker='*')

    plt.legend(['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4'], loc='upper right')
    plt.xlabel('Max Depth')
    plt.ylabel('%f1 score in validation set')
    plt.title('Decision Tree F1 Score v/s Max Depth')
    plt.show()


if __name__ == '__main__':
    main()