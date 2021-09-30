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
from tqdm import tqdm


regist=False



# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")


from detectron2.structures import BoxMode

datasets_path="../../datasets/segBuildings/"

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
                print(file,"\n",v['geometry']['type'])
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

# dataset_dicts=get_buildings_dicts(datasets_path+'/train/')
# print(dataset_dicts)
# exit()




if not regist:
    for d in ["train", "val"]:
        DatasetCatalog.register("segBuildings_" + d, lambda d=d: get_buildings_dicts(datasets_path + d))
        regist=True

segBuildings_metadata = MetadataCatalog.get("segBuildings_train")


# ## show img
# dataset_dicts = get_buildings_dicts(datasets_path+"/val/")
# for d in random.sample(dataset_dicts, 30):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=segBuildings_metadata, scale=0.25)
#     out = visualizer.draw_dataset_dict(d)
#     cv2.imshow("trian im",out.get_image()[:, :, ::-1])
#     cv2.waitKey(0)

# cv2.waitKey()
# cv2.destroyAllWindows()

# exit()

# Train

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("segBuildings_train",)
cfg.DATASETS.TEST = ()

cfg.DATALOADER.NUM_WORKERS = 14


cfg.MODEL.WEIGHTS =cfg.OUTPUT_DIR +'/model_final.pth' #model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class . (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
