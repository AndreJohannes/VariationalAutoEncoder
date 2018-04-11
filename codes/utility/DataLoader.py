class DataIterator:

    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return (data for data, target in self.data)
