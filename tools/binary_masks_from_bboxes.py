import os
import json
import numpy as np

DATA_PATH = '../data/sard_yolo/'

ANN_FILES_PATH = f'{DATA_PATH}ann_files/binary_masks/'

TRAIN_IMAGES_PATH = f'{DATA_PATH}images/train/'
VALID_IMAGES_PATH = f'{DATA_PATH}images/valid/'
TEST_IMAGES_PATH = f'{DATA_PATH}images/test/'

def convert_ann_file(image_paths, annotations_json, prefix): 
    for image_path in image_paths:
        image_filename = os.path.basename(image_path)
        image_id = next(item for item in annotations_json['images'] if item["file_name"] == image_filename)['id']
        image_annotations = [ann for ann in annotations_json['annotations'] if ann['image_id'] == image_id]
        
        for ann in image_annotations:
            bbox = ann['bbox']
            ann['segmentation'] = np.array([[bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3],bbox[0],bbox[1]+bbox[3]]])
            for seg in ann['segmentation']:
                ann['segmentation'] = seg.tolist()
    
    with open(f'{ANN_FILES_PATH}{prefix}_annotations.coco.json', 'w') as f:
        json.dump(annotations_json, f)

with open(f'{ANN_FILES_PATH}_train_annotations.coco.json', 'r') as f:
    train_annotations = json.load(f)
with open(f'{ANN_FILES_PATH}_valid_annotations.coco.json', 'r') as f:
    valid_annotations = json.load(f)
with open(f'{ANN_FILES_PATH}_test_annotations.coco.json', 'r') as f:
    test_annotations = json.load(f)

train_image_paths = [os.path.join(TRAIN_IMAGES_PATH, img['file_name']) for img in train_annotations['images']]
valid_image_paths = [os.path.join(VALID_IMAGES_PATH, img['file_name']) for img in valid_annotations['images']]
test_image_paths = [os.path.join(TEST_IMAGES_PATH, img['file_name']) for img in test_annotations['images']]

convert_ann_file(train_image_paths, train_annotations, '_train')
convert_ann_file(valid_image_paths, valid_annotations, '_valid')
convert_ann_file(test_image_paths, test_annotations, '_test')
