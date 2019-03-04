class Coordinate():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return "("+str(self.x)+","+str(self.y)+")"

class BoundingBox():
    def __init__(self, coord1, coord2):
        self.coord1, self.coord2 = coord1, coord2

    def __str__(self):
        return "BoundingBox: " + str(self.coord1) + " -> " + str(self.coord2)

    def area(self):
        return abs((self.coord2.x - self.coord1.x) * (self.coord2.y - self.coord1.y))


def intersect_over_union(box1, box2):
    overlap = "lol"
    return overlap/float(box1.area() + box2.area() - overlap)

def overlap(box1, box2):


a = BoundingBox(Coordinate(0,0), Coordinate(2,2))
b = BoundingBox(Coordinate(1,1), Coordinate(3,3))

print(a, a.area())
print(intersect_over_union(a, b))
