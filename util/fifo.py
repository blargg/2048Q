
class Empty:
    """Describes an empty cell. This should only be used by Fifo, as a
    placeholder to identify empty cells"""
    def __init__(self):
        pass


class Fifo:
    """Creates a list of elements with a finite size. When elements are added
    past that size, the earliested added element is removed"""
    def __init__(self, maxSize):
        assert maxSize > 0, "Must have a maxSize greater than 0"
        self.maxSize = maxSize
        self.nextInsertIndex = 0
        self.elements = [Empty() for _ in range(maxSize)]

    def nextAddCycles(self):
        """Returns true if the next add will wrap around in the list"""
        return (not not self.elements) and self.nextInsertIndex == 0

    def modIndex(self, index):
        """Index modulo the max size. Indecies should wrap around."""
        return index % self.maxSize

    def add(self, element):
        """adds an element. If maxSize elements are already in the Fifo, removes
        the oldest element."""
        self.elements[self.nextInsertIndex] = element
        self.nextInsertIndex = self.modIndex(self.nextInsertIndex + 1)

    def getElement(self, index):
        assert index < self.maxSize, "Index out of bounds of maxSize"
        assert type(self.elements[index]) is not Empty,\
            "Index has not been populated yet"
        return self.elements[self.modIndex(self.nextInsertIndex + index)]

    def size(self):
        return len(self.elements)

    def getMaxSize(self):
        return self.maxSize
