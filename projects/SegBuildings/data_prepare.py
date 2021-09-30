import geopandas as gpd
import math
from skimage import io
import json
import os
from shapely.geometry import Polygon
from tqdm import tqdm

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL) 


# good mask : JDZ02,
# bad mask :  
BoundSets={
            "ChengDuTianFu05":[104.0243835449218750,30.4391477700309991,104.0516434316744778,30.4591941833496094],
            "ChengDuTianFu06":[104.0242156982421875,30.4553815172624454,104.0577312409211999,30.4763755798339844],
            "ChengDuTianFu08":[104.0700302124023438,30.4475994569148334,104.0932006674850925,30.4766273498535156],
            "ChengDuTianFu10":[104.0635910034179688,30.3564526266847459,104.0784454729670330,30.3872280120849609],
            "JingDeZhen02":[117.2026367187500000,29.2673337837145482,117.2170864490679492,29.2863044738769531],
            "JingDeZhen04":[117.2324981689453125,29.2835041944319627,117.2511978813632965,29.3476047515869141],
            "JingDeZhen05":[117.1748504638671875,29.2645375995160144,117.2665178604774923,29.2989749908447266],
            "JingDeZhen06":[117.1573410034179688,29.2538627906836979,117.1751096767510347,29.3724918365478516],
            "JingDeZhen07":[117.1450042724609375,29.2656117859834488,117.1575315784811266,29.3736267089843750],
            "JingDeZhen09":[117.1182479858398438,29.2548121533758909,117.1338194552440655,29.3603000640869141],
            "JingDeZhen10":[117.1058731079101563,29.2277604574052638,117.1183623201071384,29.3513736724853516],
            "JingDeZhen11":[117.0842666625976563,29.2207530195058069,117.1059948403538868,29.3369560241699219],
            "JingDeZhen13":[117.1851654052734375,29.2826922710019559,117.2028048147170125,29.3350582122802734],
            "JingDeZhen14":[117.2026596069335938,29.2862781339484073,117.2171861179810151,29.3341217041015625],
            "JingDeZhen15":[117.1748352050781250,29.2988069451056035,117.2727815599891841,29.3332614898681641]
        }

def del_file(path_data):

    if not os.path.exists(path_data):
        return

    for i in os.listdir(path_data) :# os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data =os.path.abspath(path_data + '/' + i )#当前文件夹的下面的所有东西的绝对路径

        if os.path.isfile(file_data) == True:#os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

def trans2xy(pos,imgShape,dataBounds):
    lon=pos[0]
    lat=pos[1]
    height = imgShape[0]-1 
    width = imgShape[1]-1 
    (minLon,minLat,maxLon,maxLat) = dataBounds
    scaleX = ((maxLon-minLon) * 360000) / width 
    scaleY =  ((maxLat-minLat) * 360000) / height 
    screenX = (lon-minLon) * 360000 / scaleX
    screenY = (maxLat-lat) * 360000 / scaleY
    return (int(screenX),int(screenY))

def createBoxRegion(box):
    boxpoly = gpd.GeoSeries([Polygon([(box[0],box[1]), (box[2],box[1]), (box[2],box[3]), (box[0],box[3])])])
    boxdf = gpd.GeoDataFrame({'geometry': boxpoly, 'box':[1]})
    return boxdf

def seg2files(imgMat,shpData,steps,scales,val_rate,tif_id,output_DIR):

    assert steps[0]>0 and steps[1]>0,'Steps should be above 0'
    assert scales[0]>0 and scales[1]>0,'scales should be above 0'
    assert val_rate >= 0 and val_rate <100 ,'val_rate should be int between 0 and 100'

    os.makedirs(output_DIR+'/train/', exist_ok=True)
    os.makedirs(output_DIR+'/val/', exist_ok=True)

    dataBounds =BoundSets[tif_id] #shpData.total_bounds
    imgShape = imgMat.shape
    shpData=shpData.set_crs(4326).to_crs(3857)
    sdata=shpData.to_json()
    jsondata=json.loads(sdata)
    pixeldata=jsondata
    trans_dataBounds=createBoxRegion(dataBounds).set_crs(4326).to_crs(3857).total_bounds

    # for idf,region in enumerate(jsondata['features']):
    #     if region['geometry']!=None:
    #         for idp,point in enumerate(region['geometry']['coordinates'][0]):
    #             pixeldata['features'][idf]['geometry']['coordinates'][0][idp]=trans2xy(point,imgShape,dataBounds)

    for idf,region in enumerate(pixeldata['features']):
        if region['geometry']!=None:
            for idh,poly in enumerate(region['geometry']['coordinates']):
                points = poly if region['geometry']['type']=='Polygon' else poly[0]
                for idp,point in enumerate(points):
                    if region['geometry']['type']=='Polygon':
                        pixeldata['features'][idf]['geometry']['coordinates'][idh][idp]=trans2xy(point,imgShape,trans_dataBounds)                     
                    else:
                        pixeldata['features'][idf]['geometry']['coordinates'][idh][0][idp]=trans2xy(point,imgShape,trans_dataBounds) 

    with open("./temp.json",'w',encoding='utf-8') as json_file:
            json.dump(pixeldata,json_file,ensure_ascii=False)
    pData = gpd.read_file('temp.json')
    geoData=pData.drop(['id'],axis=1)
    geoData.crs =None
    os.remove('temp.json')
    segShape =(1000*scales[0],1000*scales[1])

    rowStep = steps[0]
    colStep = steps[1]
    num = 0
    for row in tqdm(range(0,imgShape[0],rowStep)):
        for col in range(0,imgShape[1],colStep):

            rowEnd = row + segShape[0] if row + 2*segShape[0] < imgShape[0] else imgShape[0] 
            colEnd = col + segShape[1] if col + 2*segShape[1] < imgShape[1] else imgShape[1]
            if colEnd <col + segShape[1]:
                continue
            if rowEnd <row + segShape[0]:
                continue
            pic=imgMat[row:rowEnd,col:colEnd,:]   
            boxdf = createBoxRegion((col,row,colEnd,rowEnd))
            # boxdf.buffer(10)
            # geoData.buffer(10)
            try:
                mask=gpd.overlay(boxdf, geoData, how='intersection',keep_geom_type=False)
            except Exception as e:
                # print(e)
                continue
            mask=mask.translate(-col,-row)

            num = num +1

            if val_rate!=0:
                segType = 'train' if (num % int( 100 / val_rate ) ) else 'val'
            else:
                segType = 'train'
            if len(mask) > 0:

                saveName = '/'+segType+'/'+tif_id+'_'+str(row)+'_'+str(col)+'_'+str(rowEnd)+'_'+str(colEnd)
                mask.to_file(output_DIR+saveName+'.json', driver='GeoJSON', encoding="utf-8")
                io.imsave(output_DIR+saveName+'.jpg',pic)



if __name__ == '__main__':
    output_DIR = './../../datasets/segBuildings/'
    tif_path="/home/zhizizhang/Documents/gisdata/JingDeZhen05.tif"

    imgMat = io.imread(tif_path)

    sh_file = "/home/zhizizhang/Documents/gisdata/JingDeZhen05.shp"

    shpData = gpd.read_file(sh_file)

    tif_id = tif_path.split('/')[-1].split('.')[0]

    val_rate=2 

    # del_file(output_DIR+'train')
    # del_file(output_DIR+'val')

    for hscale in range(7,8):
        for vscale in range(3,4):
            scales=(hscale,vscale)
            steps = (int(1000*scales[0]/2),int(1000*scales[1]/2))
            seg2files(imgMat,shpData,steps,scales,val_rate,tif_id,output_DIR)

