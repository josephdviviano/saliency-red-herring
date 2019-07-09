
# Utility function to set parameters of functions.
# Useful to define what consist of a model, dataset, etc.
def wrap_setattr(attr, value):
    def foo(func):
        setattr(func, attr, value)
        return func
    return foo

def setmodelname(value):
    return wrap_setattr('_MODEL_NAME', value)

def setdatasetname(value):
    return wrap_setattr('_DG_NAME', value)
