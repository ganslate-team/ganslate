import pathlib
import importlib
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path

def import_class_from_dir(class_name, dirs):
    for (importer, module_name, _) in iter_modules(dirs):

        file_path =  importer.path / module_name 
        file_path = file_path.resolve().with_suffix('.py')
        
        # Import the module from the file path by creating a spec: 
        # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Import the class from the module and return it 
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isclass(attribute) and class_name == attribute_name:            
                return attribute 