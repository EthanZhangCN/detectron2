import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer

import geopandas as gpd
import math
from skimage import io
import json
import os
from shapely.geometry import Polygon
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")

def get_dir_files(path):
    for root,dirs,files in os.walk(path):
        pass
    return files


def get_buildings_dicts(img_dir):
    files = get_dir_files(img_dir)

    dataset_dicts = []
    imgid=0

    for file in tqdm(files):

        imgid=imgid+1
        if file.split('.')[1] != "json":
            continue

        imgName=file.split('.')[0] #"JingDeZhen_02_result_dom0_0_4000_2000"#.json"
        json_file = os.path.join(img_dir, imgName+'.json')
        with open(json_file) as f:
            imgs_anns = json.load(f)

        record = {}
        imgPath =os.path.abspath(os.path.join(img_dir, imgName+'.jpg'))

        height, width = cv2.imread(imgPath).shape[:2]
        record["file_name"] = imgPath
        record["height"] = height
        record["width"] = width
        record["image_id"] = imgid

        objs = []

        for idx, v in enumerate(imgs_anns['features']):

            if (v['geometry']==None):
                continue

            if (v['geometry']['type']=='MultiPolygon'):
                annos = v["geometry"]['coordinates'][0]

            elif (v['geometry']['type']=='Polygon'):
                annos = v["geometry"]['coordinates']

            else :
                continue

            for idt,anno in enumerate(annos):

                (minX,maxY) = (min(anno,key=lambda tup:tup[0])[0],max(anno,key=lambda tup:tup[1])[1])

                (maxX,minY) = (max(anno,key=lambda tup:tup[0])[0],min(anno,key=lambda tup:tup[1])[1])
                
                poly = [p for x in anno for p in x]

                obj = {
                    "bbox": [minX, minY, maxX, maxY],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)

            record["annotations"] = objs

        if "annotations" in record:
            dataset_dicts.append(record)
    return dataset_dicts




def train(datasets_path,regist,total_iter):

    # datasets_path="../../datasets/segBuildings/"
    if not regist:
        for d in ["train", "val"]:
            DatasetCatalog.register("segBuildings_" + d, lambda d=d: get_buildings_dicts(datasets_path + d))
            regist=True

    dic =get_buildings_dicts(datasets_path + 'train')
    segBuildings_metadata = MetadataCatalog.get("segBuildings_train")

    # Train
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("segBuildings_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 14
    # cfg.MODEL.WEIGHTS =cfg.OUTPUT_DIR +'/model_final.pth' #model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = total_iter    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class . (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()

    return regist


def get_dir_files(path):
    for root,dirs,files in os.walk(path):
        pass
    return files


import data_prepare

gisdata = "/home/zhizizhang/Documents/gisdata/"
gisfiles = get_dir_files(gisdata)

regist=False
total_iter = 10000

data_Generate_DIR = './../../datasets/segBuildings/'
data_prepare.del_file(data_Generate_DIR+'val')


for file in tqdm(gisfiles):

    if file.split('.')[1] != "tif":
        continue

    tif_id = file.split('/')[-1].split('.')[0]
    tif_path = file

    imgMat = io.imread(os.path.join(gisdata,tif_path))

    sh_file = os.path.join(gisdata, file.split('.')[0] + '.shp')
    shpData = gpd.read_file(sh_file)

    val_rate=2

    for hscale in tqdm(range(1,8)):

        data_prepare.del_file(data_Generate_DIR+'train')

        for vscale in tqdm(range(1,8)):
            scales=(hscale,vscale)
            steps = (int(1000*scales[0]/2),int(1000*scales[1]/2))
            data_prepare.seg2files(imgMat,shpData,steps,scales,val_rate,tif_id)

        total_iter = 10000+total_iter
        regist = train(data_Generate_DIR,regist,total_iter)


