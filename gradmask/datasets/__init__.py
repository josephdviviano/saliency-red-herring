import os

classes = []
for module in os.listdir(os.path.dirname(__file__)):
    if module != '__init__.py' and module[-3:] == '.py':
        module_name = module[:-3]
        classes.append(module_name)

__all__ = classes
