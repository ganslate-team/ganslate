import importlib
import inspect
from pathlib import Path
from pkgutil import iter_modules

def import_class_from_dirs_and_modules(class_name, dirs_modules):
    dirs = []
    for location in dirs_modules:
        if inspect.ismodule(location):
            dir_path = Path(location.__file__).parent
        else:
            dir_path = Path(location).resolve()
        dirs.append(dir_path)
        
    for (importer, module_name, _) in iter_modules(dirs):

        file_path =  importer.path / module_name 
        file_path = file_path.resolve().with_suffix('.py')

        if not file_path.exists():
            continue
        
        # Import the module from the file path by creating a spec: 
        # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Import the class from the module and return it 
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if inspect.isclass(attribute) and class_name == attribute_name:            
                return attribute 
                
    raise ValueError(f"Class with name `{class_name}`` not found in any of the given directories or modules.\
                       If it is located in a project folder, set `project_dir` in config as the project's path.")