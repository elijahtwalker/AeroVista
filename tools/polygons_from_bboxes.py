import json

def convert_bbox_to_polygon(bbox):
    x = bbox[0]
    y = bbox[1]
    w = bbox[2]
    h = bbox[3]
    polygon = [x,y,(x+w),y,(x+w),(y+h),x,(y+h)]
    return([polygon])

def convert_ann_file(file_path):
    f = open(file_path)
    data = json.load(f)
    f.close()
    for line in data["annotations"]:
        segmentation = convert_bbox_to_polygon(line["bbox"])
        line["segmentation"] = segmentation
    f = open(file_path, 'w')
    f.write(json.dumps(data))
    f.close()

convert_ann_file("../data/sard_yolo/ann_files/polygons/_train_annotations.coco.json")
convert_ann_file("../data/sard_yolo/ann_files/polygons/_valid_annotations.coco.json")
convert_ann_file("../data/sard_yolo/ann_files/polygons/_test_annotations.coco.json")
