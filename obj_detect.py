from ultralytics import YOLO
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import csv
import os
import glob
from pathlib import Path
import argparse
from tqdm import tqdm
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_mid_mapping(mapping_csv_path):
    mid_to_entity = {}
    entity_to_mid = {}
    with open(mapping_csv_path, "r", encoding="utf-8") as f:
        for mid, entity in csv.reader(f):
            mid_to_entity[mid] = entity
            entity_to_mid[entity.lower()] = mid
    return mid_to_entity, entity_to_mid

def process_single_image(image_path, model, entity_to_mid):
    # tắt verbose
    results = model.predict(source=image_path, conf=0.1, max_det=300, iou=0.7, nms_time_limit=10.0, device=device)

    open_images_class_names = model.names
    
    detection_scores = []
    detection_class_names = []      # sẽ lưu MID
    detection_class_entities = []   # sẽ lưu Entity
    detection_boxes = []
    detection_class_labels = []     # lưu class index (số)
    
    # Detect
    for r in results:
        for box in r.boxes:
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            entity_name = open_images_class_names[cls_id]  # Ví dụ: 'Tree'
            
            # Tra MID từ entity
            mid = entity_to_mid.get(entity_name.strip().lower(), "UNKNOWN")
            
            detection_scores.append(score)
            detection_class_labels.append(cls_id)     # numeric ID
            detection_class_names.append(mid)         # MID
            detection_class_entities.append(entity_name)  # Entity
            
            # bounding box: [x1, y1, x2, y2]
            detection_boxes.append([float(x) for x in box.xyxy[0]])
    
    # Tạo JSON
    detection_json = {
        "detection_scores": detection_scores,
        "detection_class_names": detection_class_names,     # MID
        "detection_class_entities": detection_class_entities,  # Entity
        "detection_boxes": detection_boxes,
        "detection_class_labels": detection_class_labels     # class index
    }
    
    return detection_json

def process_keyframes_batch(input_folder, output_folder, mapping_csv_path=r"./oidv7-class-descriptions-boxable.csv", model_path="yolov8x-oiv7.pt"):
    
    model = YOLO(model_path)
    model.overrides['verbose'] = False
    
    mid_to_entity, entity_to_mid = load_mid_mapping(mapping_csv_path)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    image_files.sort()  
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")

    # Progress bar
    for image_path in tqdm(image_files, desc="Processing images", unit="img"):
        try:
            # Process the image
            detection_result = process_single_image(image_path, model, entity_to_mid)
            
            # Create output JSON filename
            image_name = Path(image_path).stem  # Get filename without extension
            json_filename = f"{image_name}.json"
            json_path = os.path.join(output_folder, json_filename)
            
            # Save JSON result
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(detection_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            tqdm.write(f"  -> Error processing {image_path}: {str(e)}")
    
    print(f"\nProcessing completed! Results saved to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description='Process keyframes for object detection')
    parser.add_argument('--input', '-i', required=True, help='Input folder containing keyframes')
    parser.add_argument('--output', '-o', required=True, help='Output folder for JSON results')
    parser.add_argument('--mapping', '-m', default='oidv7-class-descriptions-boxable.csv', 
                       help='Path to MID mapping CSV file')
    parser.add_argument('--model', default='yolov8x-oiv7.pt', help='Path to YOLO model')
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not os.path.exists(args.input):
        print(f"Error: Input folder '{args.input}' does not exist")
        return
    
    # Validate mapping file exists
    if not os.path.exists(args.mapping):
        print(f"Error: Mapping file '{args.mapping}' does not exist")
        return
    
    # Process keyframes
    process_keyframes_batch(args.input, args.output, args.mapping, args.model)

if __name__ == "__main__":
    # Example usage for testing
    if len(os.sys.argv) == 1:
        # Default test with video1
        input_folder = "./keyframes/L21_V001"
        output_folder = "./detections/L21_V001"
        print("Running in test mode with default paths:")
        print(f"Input: {input_folder}")
        print(f"Output: {output_folder}")
        process_keyframes_batch(input_folder, output_folder)
    else:
        main()