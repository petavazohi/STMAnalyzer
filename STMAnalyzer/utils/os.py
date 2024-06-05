from pathlib import Path
from typing import List

def check_and_iter(path: Path, file_paths: List = []) -> List:
    """
    Recursively collects and returns paths to .3ds files starting from the given directory or file path.

    Parameters:
    path (Path): The starting file or directory path from which to begin the search.
    file_paths (List, optional): A list to accumulate .3ds file paths. Defaults to an empty list.

    Returns:
    List: A list containing the paths to all .3ds files found.
    """
    if path.suffix == '.3ds':
        # If the given path is a .3ds file, append it to the file_paths list
        file_paths.append(path)
    elif path.is_dir():
        # If the path is a directory, iterate through its contents
        for idir in path.iterdir():
            # Recursively call check_and_iter for each item in the directory
            check_and_iter(idir, file_paths)
    
    return file_paths