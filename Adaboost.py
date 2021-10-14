import numpy as np
from Point import Point
from Line import Line

class Adaboost:
  def __init__(self):
    self.weights = None
    self.best_rules = None

  def train(self, points, iters, rules):
    """
    Inputs:
    - points and rules
    - iters: numer of iterations

    Returns:
    - best_rules: set of rules the size of the number of iterations
    - weights: weight for each rule
    """
    num_points = len(points)
    num_rules = len(rules)

    # init numpy arrays
    D = np.zeros(shape=(iters+1, num_points))
    self.best_rules = np.zeros(shape=iters, dtype=object)
    self.weights = np.zeros(shape=iters)

    # Adaboost algorithm
    D[0] = np.ones(shape=num_points) / num_points
    for i in range(iters):
      errors = np.zeros(shape=num_rules)
      for (j, rule) in enumerate(rules):
        for (p, point) in enumerate(points):
          if rule.classify(point) != point.label:
            errors[j] += D[i][p]
      
      # Find rule with min error and calculate its weight
      minErrorIdx = np.argmin(errors)
      minError = errors[minErrorIdx]
      minErrorRule = rules[minErrorIdx]
      self.weights[i] =  0.5 * np.log((1 - minError) / minError)
      self.best_rules[i] = minErrorRule

      # update next iteration points weight 
      for (p, point) in enumerate(points):
        D[i+1][p] = D[i][p] * (np.exp(-1*self.weights[i]*minErrorRule.classify(point)*point.label))
      z = np.sum(D[i+1])
      D[i+1] /= z

    return (self.best_rules, self.weights)

  """
  Receives a point and returns the classification of that point 
  according to adaboost classification
  """
  def predict(self, point):
    sumPred = 0
    for (i, rule) in enumerate(self.best_rules):
      sumPred  += self.weights[i]*rule.classify(point)
      
    if(sumPred > 0):
      return 1
    else:
      return -1
  
  # Calculates the mean error on a set of points
  def calculate_error(self, rules, weights, points):
    """
    Inputs:
    - rules and their weights
    - points: set of points

    Returns: mean error of the rules on the set of points
    """
    sumError = 0
    
    # calculate the rules prediction on any point
    for point in points:
      sumPred = 0
      for (i, rule) in enumerate(rules):
        sumPred += weights[i]*rule.classify(point)
      if(sumPred > 0):
        pred = 1
      else:
        pred = -1
      
      # sum the number of misclassified points
      if(pred != point.label):
        sumError += 1
    
    return (sumError/len(points))
