# /project/tensorflow.py
#
# Use pre-trained tensorflow model (tested on ssd_mobilenet_v1_coco_11_06_2017)
# to recognise some images and print some interesting properties about them.
#
# See /LICENCE.md for Copyright information

import argparse
import os
import pprint
import sys

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from object_detection.utils import label_map_util
from PIL import Image
from project.proto import string_int_label_map_pb2


def label_map_to_indexed_categories(path):
    """Loads label map protobuf."""
    with tf.gfile.GFile(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()

    try:
        text_format.Merge(label_map_string, label_map)
    except text_format.ParseError:
        label_map.ParseFromString(label_map_string)

    # The label map is 1-indexed, so we need to add a dummy entry to
    # pad it out
    return {
        l.id: l.display_name
        for l in label_map.item
    }


def detect_objects(image_np, sess, detection_graph, threshold):
    """Use the Tensorflow session to detect objects in the image array.

    Note that the image here must have been formatted as a numpy array
    already, you can use Pillow for that. The threshold indicates the scores
    of objects to exclude in the result.

    The class returned for each box is just the ID of the label, you'll need
    to run it through a label map to get string labels back.
    """
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    for box, klass, score in zip(np.squeeze(boxes),
                                 np.squeeze(classes).astype(np.int32),
                                 np.squeeze(scores)):
        if score > threshold:
            yield (list(box), klass, score)


def load_image_into_numpy_array(image):
    """Load a Pillow-encoded image into a numpy array."""
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def setup_detection_graph_from_pretrained_net(net_path):
    """Get an image detection graph ready for use from a pretrained net."""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(net_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def images_to_detection_results(detection_graph, image_paths, tolerance=0.8):
    """Given some images and a detection_graph, get detection results."""
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            for image_path in image_paths:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                for result in detect_objects(image_np,
                                             sess,
                                             detection_graph,
                                             tolerance):
                    yield tuple([image_path] + list(result))


def main(argv=None):
    """Take image on the desktop and use the model."""
    parser = argparse.ArgumentParser("""Cloud Vision Test.""")
    parser.add_argument("images",
                        nargs="+",
                        help="""The images to parse.""",
                        metavar="IMAGE")
    parser.add_argument("--tolerance",
                        type=float,
                        help="""The tolerance for image detection.""",
                        default=0.8,
                        metavar="TOLERANCE")
    parser.add_argument("--labels",
                        type=str,
                        help="""Path to the labels protocol buffer text.""",
                        required=True)
    parser.add_argument("--net",
                        type=str,
                        help="""Path to the labelling network.""",
                        required=True)
    result = parser.parse_args(argv or sys.argv[1:])

    label_map = label_map_to_indexed_categories(result.labels)
    detection_graph = setup_detection_graph_from_pretrained_net(result.net)
    results = [
        {
            "path": path,
            "box": box,
            "label": label_map[klass],
            "score": score
        }
        for path, box, klass, score in
        images_to_detection_results(detection_graph,
                                    result.images,
                                    tolerance=result.tolerance)
    ]
    pprint.pprint(results)
