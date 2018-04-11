import torch.multiprocessing as mp

class Counter(object):
    '''
    A counter used for multiprocessing, simple wrapper around multiprocessing.Value
    '''
    def __init__(self):
        from torch.multiprocessing import Value
        self.val = Value('i', 0)

    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n

    def reset(self):
        with self.val.get_lock():
            self.val.value = 0

    @property
    def value(self):
        return self.val.value

class Signal(object):
    '''
    a signal used for mutliprocessing, simple wrapper around multiprocessing.Value
    '''
    def __init__(self):
        from torch.multiprocessing import Value
        self.val = Value('i', False)

    def set_signal(self, boolean):
        with self.val.get_lock():
            self.val.value = boolean

    @property
    def value(self):
        return bool(self.val.value)