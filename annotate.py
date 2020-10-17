import numpy as np
from skimage.measure import find_contours
from shapely.geometry import Polygon, MultiPolygon 
from skimage import measure   
import json
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as plt
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def annotateResult(result, image_name, label):
    n = len(result['class_ids'])
    annotations = []
    annotationId = 1
    for i in range(n):
        if class_names[result['class_ids'][i]] == label:
            annotation = create_sub_mask_annotation(result['masks'][:, :, i], result['rois'][i],
                                                    annotationId, result['class_ids'][i], image_name)
            annotations.append(annotation)
            annotationId += 1
    return annotations


def create_sub_mask_annotation(sub_mask, bounding_box, annotationId, classId, image_name):
    # Find contours (boundary lines) around each sub-mask
    contours = measure.find_contours(sub_mask, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    # Combine the polygons to calculate the bounding box and area
    multi_poly = MultiPolygon(polygons)
    x, y, max_x, max_y = multi_poly.bounds
    width = max_x - x
    height = max_y - y
    bbox = (x, y, width, height)
    # area = multi_poly.area

    annotation = {
        'filename': image_name,
        'id': annotationId,
        'label': str(class_names[classId]),
        'bbox': bbox,
        'segmentation': segmentations,
    }

    return annotation



def writeToJSONFile(path, fileName, data):
    fileName = fileName.split(".")[0]
    filePathNameWExt =  path + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)


def annotateAndSaveAnnotations(r, directory, image_name, label):
    annotationsJson = annotateResult(r, image_name, label)
    writeToJSONFile(directory, image_name, annotationsJson)


def annotateImagesInDirectory(rcnn, directory_path, label):
    for fileName in os.listdir(directory_path):
        if fileName.endswith(".jpg") or fileName.endswith(".jpeg") or fileName.endswith(".png") or fileName.endswith(".tif") or fileName.endswith(".tiff"):
        # load image
            print("Evaluating Image: " + fileName)
            img = load_img(directory_path+"/"+fileName)
            img = img_to_array(img)
            # make prediction
            results = rcnn.detect([img], verbose=0)
            # get dictionary for first prediction
            result = results[0]

            if class_names.index(label) in result['class_ids']:
                print("Label found in image: " + fileName)
                print("Annotating...")
                annotateAndSaveAnnotations(result, directory_path, fileName, label)
                if (args.displayMaskedImages is True):
                    display_instances(img, result['rois'], result['masks'], result['class_ids'],
                                  class_names, class_names.index(label), result['scores'])
            else:
                print("Label not found in image: " + fileName)


ROOT_DIR = os.path.abspath("./")
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "./mask_rcnn_coco.h5")

# Directory to save logs, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description = 'Annotate the object')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'annotateCoco' or 'annotateCustom'")
    parser.add_argument('--image_directory', required=True,
                        metavar="/path/to/the/image/directory/",
                        help='Directory of the images that need to be annotated')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="path_to_weights.h5_file or 'coco_weights'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--label', required=True,
                        metavar="object_label_to_annotate",
                        help='Either COCO dataset labels or custom')
    parser.add_argument('--displayMaskedImages', type=bool,
                        default=False, required=False,
                        help='Display the masked images.')
                        
    args = parser.parse_args()

    # Validate arguments
    if args.command == "annotateCoco":
        assert args.label in class_names, "Label --label does not belong to COCO labels "

    elif args.command == "annotateCustom":
        assert args.label, "Argument --label is required for annotation"

    assert args.image_directory, "Argument --image_directory is required for annotation"
    assert args.weights, "Argument --weights is required for annotation"


    class InferenceCocoConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = "inferenceCoco"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 80
        
    class InferenceCustomConfig(Config):
        NAME = "inferenceCustom"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1


    if args.command == "annotateCoco":
        config = InferenceCocoConfig()
        class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
    else:
        config = InferenceCustomConfig()
        class_names = ['BG']
        class_names.append(args.label)
    config.display()

    # Create model
    model = MaskRCNN(mode="inference", config = config, model_dir = "./")

    # Select weights file to load
    if args.command == "annotateCoco":
        weights_path = COCO_WEIGHTS_PATH

        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights... ", weights_path)
    model.load_weights(weights_path, by_name=True)


    # Annotate
    if args.command == "annotateCoco" or args.command == "annotateCustom":
        annotateImagesInDirectory(model, directory_path=args.image_directory, label = args.label)
    else:
        print("'{}' is not recognized. "
              "Use 'annotateCoco' or 'annotateCustom'".format(args.command))

