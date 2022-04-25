
import numpy as np 
import os
import rasterio as rio
import matplotlib.pyplot as plt
import glob
from rasterio import windows
import affine
import copy
import argparse



# Modified from https://github.com/szymon-datalions/geoprocessing/blob/master/rasters/clip_into_quadrants.py 
class ImageForClipModel_MS:
    """
    Class for clipping big raster images into smaller parts. Class initialized with image filename.
    The multispectral image is padded according to the tile size as provided as input by the user.
    The clipped images are saved in a folder.
    """

    # Get bands, band shape, dtype, crs and transform values
    def __init__(self, image_path):
        with rio.open(image_path, 'r') as f:
            self.bands = f.read()
            self.crs = f.crs
            self.base_transform = f.transform
            
        self.bands_shape = self.bands.shape
        self.bands_dtype = self.bands.dtype
 
        ## for padding       
        self.padded_bands = None
        
    
    # pad image bands to required tile shape. It adds black pixels to the rows and columns 
    # from last tile to complete the tile size.
    def pad2tiles_size(self,rasterio_image_bands, tile_size_x, tile_size_y):    
        channels,height, width = rasterio_image_bands.shape
    
        pad_x = int(np.ceil(width / tile_size_x) * tile_size_x)  ## width
        pad_y = int(np.ceil(height / tile_size_y) * tile_size_y) ## height
   
        padded_array = np.zeros((channels,pad_y,pad_x))
        padded_array[:channels,:height,:width] = rasterio_image_bands
    
        return padded_array
    

    # Function for clipping bands
    def clip_raster(self, height, width, prefix_img, buffer=0):
        """
        Function clips and saves raster data
        
        INPUT:
        :param height: height of the clipped image in pixels.
        :param width: width of the clipped image in pixels.
        :param prefix_img: default = 'clipped_band_'.
        Prefix of the clipped data if it is saved as the static files.
        :param buffer: default = 0. 
        If buffer is provided then clipped tiles overlaps by number of pixels provided in this
            variable. As example: if width = 1000 and buffer is 10 then the first tile will be extracted from
            columns 1 to 1000, the second tile from 991 to 1990. The formula is:
            where i > 1
            then start_column = 1000 * (i - 1) - (buffer * (i - 1)) + 1
            end_column = start_column + 999
        
        
        OUTPUT
        :save clipped tiles of non black tiles
        """
        
        removed_images = 0
        ## padding image bands
        self.padded_bands = self.pad2tiles_size(self.bands,height,width).astype(self.bands_dtype)        
        padded_bands_shape = self.padded_bands.shape  # shape of padded image bands

        
        row_position = 0
        while row_position <  padded_bands_shape[1]: 
            col_position = 0
            while col_position < padded_bands_shape[2]: 
                
                clipped_image = self.padded_bands[:,row_position:row_position + height,
                                col_position:col_position + width]
            
                # Check if frame is empty (pixels are black)
                if np.mean(clipped_image)!=0 :
                
                    # Positioning
                    tcol, trow = self.base_transform * (col_position, row_position)
                    new_transform = affine.Affine(self.base_transform[0], self.base_transform[1], tcol,
                                              self.base_transform[3], self.base_transform[4], trow)
                
                    image = [clipped_image, 
                         self.crs, 
                         new_transform,
                         clipped_image.shape[1], #height
                         clipped_image.shape[2], #width
                         clipped_image.shape[0], #channels
                         self.band_dtype]
                    
                    # Save into a set
                    filename_img = prefix_img + '_x_' + str(col_position) + '_y_' + str(row_position) + '.tif'
                    ## write ms image
                    with rio.open(filename_img, 'w', driver='GTiff', height=image[3], width=image[4], count=image[5], 
                                  dtype=image[6], crs=image[1], transform=image[2]) as dst:
                        dst.write(image[0])
                            
                else:
                    # count number of black tiles
                    removed_images+=1

                # Update column position
                col_position = col_position + width - buffer

            # Update row position
            row_position = row_position + height - buffer

        print('Tiles saved successfully')
        print('Number of removed tiles: ', removed_images)



############ CUT IMAGES IN TILES ################### 
if __name__ == "__main__":  
    print('Cut ms image into tiles ... ')

    parser = argparse.ArgumentParser(description='Information')
    parser.add_argument('--image2save', dest='image2save', type=str, help='Image Path')
    parser.add_argument('--height', dest='height', type=int, help='Height of Tile')
    parser.add_argument('--width', dest='width', type=int, help='Width of Tile')
    parser.add_argument('--img_prefix', dest='image_prefix', type=str, help='Tiles prefix')
    parser.add_argument('--imgTilesPath', dest='imgTilesPath', type=str, help='Tiles path')
    args = parser.parse_args()

    ## clip image with rasterio    
    clipper= ImageForClipModel_MS(image_path=image2save)
    clipper.clip_raster(height=tile_size_y,
                        width=tile_size_x,
                        prefix_img=imgTilesPath+img_prefix,
                        buffer=0)

    print('Finished!')
