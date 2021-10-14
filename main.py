from Point import Point
from Line import Line
from Adaboost import Adaboost
import numpy as np
from sklearn.model_selection import train_test_split

# Create all points and labels from rectangle.txt
points = []

f = open("rectangle.txt", "r")
for line in f:
    words = line.split()
    points.append(Point(float(words[0]), float(words[1]), float(words[2])))
f.close()

# Run Adaboost 100 times. Each run of Adaboost is of 8 iterations
num_iters = 100
num_rules = 8
train_errors = np.zeros(shape=num_rules)
test_errors = np.zeros(shape=num_rules)

for i in range(num_iters):
    # randomly split point to half train and half test
    train, test = train_test_split(points, test_size=0.5, shuffle=True)
    train = np.array(train)
    test = np.array(test)

    # Create every possible rule from every 2 points ןn train
    rules = []
    num_points = len(train)
    for j in range(num_points):
        for k in range(j + 1, num_points):
            rules.append(Line(train[j], train[k], direction=False))
            rules.append(Line(train[j], train[k], direction=True))
    rules = np.array(rules)

    # get best rules and their weight using adaboost algorithm
    adaboost = Adaboost()
    (best_rules, weights) = adaboost.train(train, num_rules, rules)

    for j in range(num_rules):
        train_errors[j] += adaboost.calculate_error(best_rules[:j + 1], weights[:j + 1], train)
        test_errors[j] += adaboost.calculate_error(best_rules[:j + 1], weights[:j + 1], test)

train_errors /= num_iters
test_errors /= num_iters

# print results
print("train mean error per rule: \n", train_errors)
print("train mean error: ", np.average(train_errors), "\n")

print("test mean error per rule: \n", test_errors)
print("test mean error: ", np.average(test_errors))
