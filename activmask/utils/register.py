def wrap_setattr(attr, value):
    """
    Utility function to set parameters of functions.
    Useful to define what consist of a model, dataset, etc.
    """
    def foo(func):
        setattr(func, attr, value)
        return func
    return foo

def setmodelname(value):
    return wrap_setattr('_MODEL_NAME', value)

def setdatasetname(value):
    return wrap_setattr('_DG_NAME', value)

def setcallbackname(value):
    return wrap_setattr('_CB_NAME', value)
