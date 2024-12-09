import os

def get_resource_path(relative_path):
    """Return the absolute path for a resource given its relative path from the library root."""
    # Get the directory of this file (assuming it's within the library package)
    this_dir = os.path.dirname(__file__)
    
    # Move up one level to get to the library root (since we are in 'your_library/utility.py')
    library_root = os.path.dirname(this_dir)
    
    # Join the library root with the relative path to the resource
    resource_path = os.path.join(library_root, relative_path)
    
    return resource_path