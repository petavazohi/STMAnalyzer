import sys
import subprocess
from pathlib import Path
from re import findall
from typing import List, Tuple, Dict

if 'google.colab' in sys.modules:
  from google.colab import drive
  drive.mount('/content/drive')
  base_dir = Path('/content/drive/My Drive/_projects/ML-STS')
  if base_dir.exists():
      # Change the current working directory to base_dir
      %cd "$base_dir"
      repo_path = base_dir / "nanonispy"
      if not repo_path.exists():
        !git clone "https://github.com/underchemist/nanonispy.git"
      if repo_path.as_posix() not in sys.path:
        sys.path.append(repo_path.as_posix())

  else:
      print("Directory does not exist.")
else:
  base_dir = Path('G:\My Drive\_projects\ML-STS')
import nanonispy.read as nap1