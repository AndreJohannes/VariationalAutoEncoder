import numpy

def get_number_of_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([numpy.prod(p.size()) for p in model_parameters])
