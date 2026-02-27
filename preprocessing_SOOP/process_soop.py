import subprocess
from threading import Thread, Lock
import os
import sys
from math import ceil
from datetime import datetime
import time
import argparse


ROOT_FOLDER = '/Datasets/SOOP'
MNI152_TEMPLATE = '/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz'
preprocessing_script = '/scripts/SOOP-T1_Preprocessing/freesurfer_and_fsl.sh'

LOG_DIR = '/scripts/SOOP-T1_Preprocessing'
LOG_FILE = os.path.join(LOG_DIR, 'log.txt')

log_lock = Lock()

def log_completion(subject, thread_id, node_id, start_time, end_time, duration):
    log_entry = "{}, Node batch file {}, Thread {}, Start: {}, End: {}, Duration: {:.2f} seconds\n".format(
        subject, node_id, thread_id, start_time, end_time, duration)
    with log_lock:
        with open(LOG_FILE, 'a') as f:
            f.write(log_entry)

class Worker(Thread):
    def __init__(self, files, source, destination, node_id, max_images=None):
        # files = batches (batch text file list of images)
        # source = raw data folder
        # destination = freesurfer+fsl folder
        # node_id = batch text file name
        # max images = stop thread after n images 
        super(Worker, self).__init__()
        self.files = files
        self.source = source
        self.destination = destination
        self.node_id = node_id
        self.max_images = max_images

    def run(self):
        count = 0
        for filename in self.files:
            if self.max_images is not None and count >= self.max_images:
                break
            
            if '_T1w' in filename:
            
                subject = filename.replace('_T1w.nii.gz', '')
                
                cmd = ['/usr/bin/bash', preprocessing_script,
                    '-f', os.path.join(self.source, subject, 'anat', filename), # FIRST ISSUE: SOOP structure is different | source/subject/anat/filename
                    '-d', os.path.join(self.destination, subject),
                    '-t', MNI152_TEMPLATE]

                start = datetime.now()
                start_time_str = start.strftime("%Y-%m-%d %H:%M:%S")
                t0 = time.time()

                # subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
                subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)

                print("Processing Subject {} File {} in thread {}".format(subject, filename, self.name))
                sys.stdout.flush()
                end = datetime.now()
                end_time_str = end.strftime("%Y-%m-%d %H:%M:%S")
                t1 = time.time()
                duration = t1 - t0

                log_completion(subject, self.name, self.node_id, start_time_str, end_time_str, duration)
                count += 1

def main():
    parser = argparse.ArgumentParser(description="Process images in batches with multi-threading.")
    parser.add_argument('batch_file', type=str, help='Path to the batch file (list of image filenames)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Maximum number of images each thread should process before stopping (default: all)')
    parser.add_argument('--num_threads', type=int, default=1)
    args = parser.parse_args()

    NUM_THREADS = args.num_threads
    if NUM_THREADS < 1:
        raise ValueError("Number of threads must be at least 1.")
    # Load batch file
    if not os.path.isfile(args.batch_file):
        print("Batch file '{}' not found.".format(args.batch_file))
        sys.exit(1)

    with open(args.batch_file, 'r') as file:
        node_images = [line.strip() for line in file if line.strip()]

    raw_folder = os.path.join(ROOT_FOLDER, 'SOOP_Original')
    freesurfer_fsl_folder = os.path.join(ROOT_FOLDER, 'SOOP-T1w/freesurfer+fsl')

    if not os.path.isdir(ROOT_FOLDER):
        raise ValueError('Root folder does not exist! Check Python Script. Supplied folder path: {}'.format(ROOT_FOLDER))
    if not os.path.isdir(raw_folder):
        raise ValueError('Raw dataset folder does not exist! Check Python Script. Supplied raw folder: {}'.format(raw_folder))
    if not os.path.isfile(preprocessing_script):
        raise ValueError('Unable to find FreeSurfer+FSL preprocessing script at {}'.format(preprocessing_script))

    if not os.path.isdir(freesurfer_fsl_folder):
        os.mkdir(freesurfer_fsl_folder)

    node_images.sort()

    total_files = len(node_images)
    chunk_size = int(ceil(float(total_files) / NUM_THREADS))
    batches = {}

    for i in range(NUM_THREADS):
        start = i * chunk_size
        end = min(start + chunk_size, total_files)
        sublist = node_images[start:end]
        batches[i] = sublist

    
    print("Split {} files into {} threads.".format(total_files, NUM_THREADS))
    print("Starting {} threads with ~{} files each.".format(NUM_THREADS, chunk_size if args.max_images is None else args.max_images))

    threads = []
    for i in range(NUM_THREADS):
        worker = Worker(batches[i], raw_folder, freesurfer_fsl_folder, node_id=args.batch_file, max_images=args.max_images)
        threads.append(worker)

    for worker in threads:
        worker.start()

    print("All threads started. Waiting for completion...")

    for worker in threads:
        worker.join()

    print("All threads have completed their work.")

if __name__ == '__main__':
    main()


# python process_soop_t1w.py /scripts/SOOP-T1_Preprocessing/Node_1.txt --max_images 2 --num_threads 2

