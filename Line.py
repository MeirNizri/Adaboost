
class Line:
  def __init__(self, p1, p2, direction):
    """
    Inputs:
    - p1, p2: points to make line equasion
    - direction: if true, Points above the line will be classified as -1, otherwise 1
    """
    self.direction = direction
    if (p1.x == p2.x):
        self.parallel_to_y = True
        self.m = 0
        self.n = p1.x
    else:
      self.parallel_to_y = False
      self.m = (p1.y - p2.y) / (p1.x - p2.x)
      self.n = p1.y - (self.m * p1.x)

  """
  Receives a point and returns the classification of that point 
  according to the rule defined by the line
  """
  def classify(self, p):
    if self.direction:
      aboveLine = -1
    else:
      aboveLine = 1

    if (not self.parallel_to_y):
      y = (self.m * p.x) + self.n
      if p.y >= y:
        return aboveLine
      else:
        return -aboveLine
    else:
      if p.x >= self.n :
        return aboveLine
      else:
        return -aboveLine
