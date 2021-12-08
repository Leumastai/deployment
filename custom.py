import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath("../../Mask_RCNN")
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_instances
from mrcnn.visualize import display_images
from mrcnn import model as modellib, utils
from mrcnn.model import log
from mrcnn.config import Config

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

CUSTOM_DIR = os.path.join(ROOT_DIR, "crane_handle_")

INFERENCE_IMG_PATH = os.path.join(ROOT_DIR, "crane_handle_/val/33f5a133-516_crane.jpg") #path to the image that will be used for inference

WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_object_0004.h5") #path to the last trained weight

class CustomConfig(Config):
    """
    Building configuration for the crane datasets. 
    from base Config class and overrides some values.
    """

    #configuration name
    NAME = "object"

    IMAGES_PER_GPU = 1 #TODO: Change to 2 for training and 1 for inference

    GPU_COUNT = 1

    NUM_CLASSES = 1 + 1 #background + crane

    #Number of training steps per epoch
    STEPS_PER_EPOCH = 20

    #Skip detection with < 80% confidence 
    DETECTION_MIN_CONFIDENCE = 0.9

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
      
      """
      Load a subset of the crane dataset

      Args:
      dataset_dir: root directory of the dataset
      subset: the subset of the dataset to load either train or val

      Return:
      None
      """

      #Add classes, since we only have oone class
      self.add_class("object", 1, "crane handle")

      #train or validation dataset
      assert subset in ['train', 'val']
      dataset_dir = os.path.join(dataset_dir, subset)

      #Load annotations
      #since we only care about the x and y coordinate of each regions
      annotations1 = json.load(open(os.path.join(dataset_dir, "Crane_json.json")))

      #print annotations
      annotations = list(annotations1.values()) #we don't need the dictionary keys

      #skipping unannotated images
      annotations = [a for a in annotations if a['regions']]

      #add images
      for a in annotations:
        #get the x, y coordinates of points of the polygon that makes up the
        #outline of each object instances

        polygons = [r['shape_attributes'] for r in a['regions']]
        #name is from the header of the labeling tool
        objects = [s['region_attributes']['name'] for s in a['regions']]
        #print("objects", objects)

        name_dict = {"crane handle" : 1}
        num_ids = [name_dict[a] for a in objects]

        #print("numids", num_ids)

        image_path = os.path.join(dataset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        self.add_image(
            "object",
            image_id = a['filename'],
            path = image_path,
            width = width,
            height = height,
            polygons = polygons,
            num_ids = num_ids
        )

    def load_mask(self, image_id):
      """
      It generate instance mask for an image

      Args:
      image_id: 

      Returns:
      mask: a boll array of shape [heaight, width, instance count] with one mask 
      per image
      class_ids: a 1D array of class IDs of the instance masks.
      """

      # if not a crane dataset image, delegate to parent class
      image_info = self.image_info[image_id]
      if image_info['source'] != "object":
        return super(self.__class__, self).load_mask(image_id)

      #convert polygons to a bitmap mask of shape
      info = self.image_info[image_id]
      if info["source"] != "object":
        return super(self.__class__, self).load_mask(image_id)
      num_ids = info['num_ids']
      mask = np.zeros([info['height'], info["width"], len(info['polygons'])],
                      dtype=np.uint8)
      for i, p in enumerate(info['polygons']):
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        mask[rr, cc, i] = 1

      #return mask and array of class IDs of each instances
      num_ids = np.array(num_ids, dtype=np.int32)
      return mask, num_ids

    def image_reference(self, image_id):
      """
      Returns the path of an image
      """

      info = self.image_info[image_id]
      if info['source'] == "object":
        return info['path']
      
      else:
        super(self.__class__, self).images_reference(image_id)


#config = CustomConfig()