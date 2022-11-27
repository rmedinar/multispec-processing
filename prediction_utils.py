"""
Extract contours from predicted tiles, save them as polygons
and write them into a csv file to be load into qgis as a 
layer.


Author:
    Rosario Alejandra Medina Rodríguez - 12.01.2022
"""

import os
import cv2
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import shapes
from shapely.geometry import Point, Polygon, shape
from shapely import wkt
import torch


def draw_convex_hull(mask):
    """
    Function that draws convex hulls contours from image patch prediction
    :param mask: prediction patch
    :return: patch contours
    """
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        hull = cv2.convexHull(c)
        cv2.drawContours(img, [hull], 0, (255, 255, 255),-1)

    return img/255.


def savePredImages(pred,inputImageTileName,outputName):
    """
    Function that saves predicted tiles 
    :param pred:
    :param inputImageTileName:
    :param outputName:
    :return:
    """
    img_data =  rasterio.open(inputImageTileName)
    meta = img_data.meta
    img_data.close()

    ## update geo information
    kwargs = meta
    kwargs.update(count=1) # 1 band
    kwargs.update(nodata=0)

    with rasterio.open(outputName, 'w', **kwargs) as dst:
        dst.write(pred,1)

        
'''
predict patches
'''
def predictPatches(imageTilesPath,modelPath,predMaskOutputPath):
    """
    Network predictions
    :param spotTilesPath:
    :param modelPath:
    :param predMaskOutputPath:
    :return:
    """
    ## create dataset and dataloader from previously cutted tiles
    test_dataset = ImageDataset(imageTilesPath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=12, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_model = model(6,2).to(device)
    test_model.load_state_dict(torch.load(modelPath))
    test_model.eval()

    for batch_i, (x, path) in enumerate(test_dataloader):
        for j in range(len(x)):
        
            with torch.no_grad():
                result = test_model(x.to(device)[j:j+1])

            c1 = torch.exp(result[0][0,:,:]).detach().cpu()
            c2 = torch.exp(result[0][1,:,:]).detach().cpu()
                       
            ############################################
        
            my_threshold = 1.      
            mask = (cv2.threshold(c2.numpy(), my_threshold, 1, cv2.THRESH_BINARY)[1])
            mask = draw_convex_hull(mask.astype(np.uint8))
            num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
            predictions = np.zeros((256, 256), np.float32)
            num = 0
            min_size = 0
            for c in range(1, num_component):
                p = component == c
                if p.sum() > min_size:
                    predictions[p] = 1
                    num += 1
                
            patchName = os.path.basename(path[j])   
            savePredImages(predictions, path[j],predMaskOutputPath+'pred_'+patchName+'.tif')

            
    print('Finished predicting and saving masks')



def generateShapelyPolygonsFromMask(mask_image_name):
    """
    Generate polygons
    :param mask_image_name:
    :return: list of shapely polygons
    """
    with rasterio.Env():
        with rasterio.open(mask_image_name) as src:
            image = src.read(1).astype('uint8') #first band
            results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s,v) in enumerate(shapes(image, image>0, transform=src.transform))) # image>0 to get polygons from class 1
    
    # list of polygons
    geoms = list(results)
    # convert to geometry
    polygons_list = []
    for i in range(len(geoms)):
        polygons_list.append(shape(geoms[i]['geometry']))
        
    return polygons_list



def generatePolygonsFileFromMasks(predTilesPath,outputPath):
    """
    Generate csv with polygons
    :param mask_image_name:
    :return: list of shapely polygons
    """ 
    tmpFolder = '/'.join(outputPath.split('/')[:-2])+'/'
    with open(tmpFolder+'generatedPolygons.csv', 'a') as f:
        pred_mask_list = glob.glob(predTilesPath + '/**/*.tif', recursive=True)
        for tmp_mask_name in tqdm_notebook(pred_mask_list):
            tmp_polygon_list = generateShapelyPolygonsFromMask(tmp_mask_name)
            for poly in tmp_polygon_list:
                txt_line = wkt.dumps(poly) 
                f.write(txt_line+'\n') 



############ CUT IMAGES IN TILES ################### 
if __name__ == "__main__":  
    
