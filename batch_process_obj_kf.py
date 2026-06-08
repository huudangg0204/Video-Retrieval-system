#!/usr/bin/env python3


import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from obj_detect import process_keyframes_batch

def process_video_folder(video_folder, base_keyframes_dir, base_output_dir, mapping_csv_path, model_path):
    input_folder = os.path.join(base_keyframes_dir, video_folder)
    output_folder = os.path.join(base_output_dir, video_folder)
    try:
        process_keyframes_batch(
            input_folder=input_folder,
            output_folder=output_folder,
            mapping_csv_path=mapping_csv_path,
            model_path=model_path
        )
        print(f"Successfully processed {video_folder}")
    except Exception as e:
        print(f"Error processing {video_folder}: {str(e)}")

def batch_process_all_videos(base_keyframes_dir=r"./keyframes", 
                           base_output_dir=r"./detections",
                           mapping_csv_path=r"./oidv7-class-descriptions-boxable.csv",
                           model_path=r"./yolov8x-oiv7.pt",
                           num_worker=4):

    base_keyframes_dir = os.path.abspath(base_keyframes_dir)
    base_output_dir = os.path.abspath(base_output_dir)
    
    if not os.path.exists(base_keyframes_dir):
        print(f"Error: Keyframes directory '{base_keyframes_dir}' does not exist")
        return
    
    video_folders = []
    for item in os.listdir(base_keyframes_dir):
        item_path = os.path.join(base_keyframes_dir, item)
        if os.path.isdir(item_path):
            video_folders.append(item)
    
    video_folders.sort()
    
    if not video_folders:
        print(f"No video folders found in {base_keyframes_dir}")
        return
    
    print(f"Found {len(video_folders)} video folders to process:")
    for folder in video_folders:
        print(f"  - {folder}")

    print(f"\n{'='*60}")
    print("Starting batch processing with multithreading...")
    print(f"{'='*60}")

    with ThreadPoolExecutor(max_workers= num_worker) as executor:
        futures = [
            executor.submit(
                process_video_folder,
                video_folder,
                base_keyframes_dir,
                base_output_dir,
                mapping_csv_path,
                model_path
            )
            for video_folder in video_folders
        ]
        for i, future in enumerate(as_completed(futures), 1):
            # Optionally, you can handle results or exceptions here
            try:
                future.result()
            except Exception as e:
                print(f"Thread {i} raised an exception: {e}")

    print(f"\n{'='*60}")
    print("Batch processing completed!")
    print(f"Results saved to: {base_output_dir}")

if __name__ == "__main__":
    # main()
    batch_process_all_videos()
