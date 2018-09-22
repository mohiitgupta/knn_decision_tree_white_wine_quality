import math

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
    # distance = euclidean_distance(point1, point2)
    distance = manhattan_distance(point1, point2)
    # distance = cosine_distance(point1, point2)
    # print distance
    return distance