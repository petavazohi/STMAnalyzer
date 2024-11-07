import os
import typer
import time
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import subprocess
import platform
import sys
from datetime import datetime
from pathlib import Path
from re import findall
from typing import Dict, List, Tuple

import h5py
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

scripts_path = Path(__file__).resolve().parent
sys.path.append(str(scripts_path.parent))
app = typer.Typer()

import STMAnalyzer
from STMAnalyzer.core import STMScan
from STMAnalyzer.utils.os import check_and_iter, remove_duplicates

def is_matlab_installed():
        # Use 'matlab -batch' for checking MATLAB without opening GUI
        if platform.system() == "Windows":
            # Windows: Check for matlab command
            result = subprocess.run(["matlab", "-batch", "disp('MATLAB is installed')"], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    text=True, shell=True)
        else:
            # Linux: Check for matlab command
            result = subprocess.run(["matlab", "-batch", "disp('MATLAB is installed')"], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                    text=True)
        
        # If MATLAB responds, it's installed
        if result.returncode == 0:
            print(result.stdout)  # Print MATLAB's output
        else:
            raise RuntimeError("MATLAB command not found or not installed. Please install MATLAB to proceed.")

def process_existing_files(directory: Path):
    """Process existing .3ds files in the directory."""
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix == '.3ds':
            print(f"Processing existing file {file_path}")
            process_3ds_file(file_path)

class MyEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        if file_path.suffix != '.3ds':
            return
        print(f"File {event.src_path} has been created")
        if file_path.suffix == '.3ds':
            Thread(target=wait_for_file_completion, args=(file_path,)).start()
        elif file_path.suffix in ['.json', '.h5', '.hdf5']:
            return
        else:
            print(f"Handling files with the '{file_path.suffix}' extension is not implemented yet.")


def wait_for_file_completion(file_path: Path, wait_time:int =1, timeout: int=30):
    last_size = -1
    start_time = time.time()

    while True:
        current_size = file_path.stat().st_size
        if current_size == last_size:
            print(f"File {file_path} writing completed.")
            if file_path.suffix == '.3ds':
                process_3ds_file(file_path)
            break
        last_size = current_size
        time.sleep(wait_time)

        if time.time() - start_time > timeout:
            print(f"Timeout waiting for file {file_path} to complete writing.")
            break

def process_3ds_file(file_path: Path):
    base_dir = file_path.parent
    print(f"Processing file: {file_path}")
    stm_scan = STMScan.from_file(file_path)
    stm_scan.histogram_equalization()
    stm_scan.flatten_topography()
    stm_scan.normalize_to_setpoint()
    stm_scan.plot_dIdV()
    print(stm_scan)
    plt.savefig(file_path.with_suffix('.svg'))
    scan_name = Path(stm_scan.metadata['file_path']).stem.replace('-','_')
    file_path_h5 = base_dir/file_path.with_suffix('.h5')
    with h5py.File(file_path_h5, 'w') as wf:
            ds = wf.create_dataset(scan_name, data=stm_scan.dIdV.reshape(-1, stm_scan.nE))
            for key, value in stm_scan.metadata.items():
                ds.attrs[key] = str(value)
    dimensions = [4, 4]
    settings = {"system": 'FeSeTe',
            "dimensions": dimensions,
            "topologyFcn": "hextop",
            "distanceFcn": "dist",
            "coverSteps": 100,
            "initNeighbor": 3,
            "epochs": 200,
            "type": "ind",
            'filePath': file_path_h5.absolute().resolve().as_posix(),
            'storeFormats': ['.png'], 
            'overWrite': True}
    today_date = datetime.now().strftime('%Y%m%d')
    calc_dir_name = (
        f"{today_date}-{settings['system']}-{settings['dimensions'][0]}x{settings['dimensions'][1]}-{settings['topologyFcn']}-"
        f"{settings['epochs']}Ep-{settings['distanceFcn']}-"
        f"covStp{settings['coverSteps']}-"
        f"iniNei{settings['initNeighbor']}"
    )
    with open(file_path.parent/'settings.json', 'w') as wf:
        json.dump(settings, wf, indent=2)
    command = ['matlab', '-batch', f"addpath('{scripts_path.as_posix()}'); som_loop"]
    subprocess.run(command, cwd=file_path.parent)
    matlab_output = STMAnalyzer.io.read_hdf5(file_path.parent/calc_dir_name/'output.h5')
    for key in matlab_output:
        if key != 'weights':
            STMAnalyzer.plot.hit_histogram(matlab_output[key]['hitHistorgam'], V=stm_scan.V, som_weights=matlab_output[key]['weights'], savefig= file_path.parent/calc_dir_name/'MaxHits.png', offset=0.01)
    for i, key in enumerate(matlab_output):
        if key != 'weights':
            STMAnalyzer.plot.som_didv_topo(matlab_output[key]['hitHistorgam'],
                                        matlab_output[key]['weights'],
                                        matlab_output[key]['clusterIndex'],
                                        stm_scan=stm_scan, 
                                        block_size=(2, 2),
                                        #    dimensions=, (dimensions[0], dimensions[1]),
                                        offset=0.01, savefig=file_path.parent/calc_dir_name/f'proj_topo{i}.png')
    
@app.command()
def main(process_existing: bool = typer.Option(False, help='Process existing .3ds files before monitoring.'),
         directory: Path = typer.Option(Path.cwd(), help='Directory to monitor.')):
    """Monitor a directory and optionally process existing .3ds files."""
    is_matlab_installed()
    if process_existing:
        print("Processing existing .3ds files...")
        process_existing_files(directory)
    print(f"Monitoring directory: {directory}")
    event_handler = MyEventHandler()
    observer = Observer()
    observer.schedule(event_handler, directory, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    app()