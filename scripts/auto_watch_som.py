import os
import time
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class MyEventHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        print(f"File {event.src_path} has been created")
        Thread(target=wait_for_file_completion, args=(event.src_path,)).start()

def wait_for_file_completion(file_path, wait_time=1, timeout=30):
    last_size = -1
    start_time = time.time()

    while True:
        current_size = os.path.getsize(file_path)
        if current_size == last_size:
            print(f"File {file_path} writing completed.")
            process_3ds_file(file_path)
            break
        last_size = current_size
        time.sleep(wait_time)

        if time.time() - start_time > timeout:
            print(f"Timeout waiting for file {file_path} to complete writing.")
            break

def process_3ds_file(file_path):
    print(f"Processing file: {file_path}")
    # Add your file processing logic here

if __name__ == "__main__":
    path = "."  # Directory to monitor
    event_handler = MyEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
