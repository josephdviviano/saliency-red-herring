import torch.nn as nn
import importlib
import gradmask.utils.register as register

@register.setmodelname("CNNWrapper")
class CNNWrapper(nn.Module):

    def __init__(self, model_name='resnet50', num_class=2, **model_args):
        super(CNNWrapper, self).__init__()

        # Obtain the correct Model from torchvision
        models_from_module = importlib.import_module('torchvision.models')

        if hasattr(models_from_module, model_name):
            obj = getattr(models_from_module, model_name)
        else:
            raise ValueError("{} is not a valid model.".format(model_name))

        self.model = obj(**model_args)

        # Change the last layer to be as versatile as possible.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_class)

    def forward(self, x):
        return self.model(x)
