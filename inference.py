import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"

import glob
import tensorflow as tf
from resunet_tf import ResUnet

import keras.backend as K
import numpy as np
from PIL import Image
from tqdm import tqdm



def decode_img(img):
  #img = tf.image.decode_jpeg(img, channels=3) #color images
  img = tf.image.convert_image_dtype(img, tf.float32)
   #convert unit8 tensor to floats in the [0,1]range
  img = tf.cast(img, tf.float32) / 255.0
  return tf.image.resize(img, [256, 256])

def decode_mask(mask):
  #mask = tf.image.decode_jpeg(mask, channels=1) #greyscale images
  #convert from L to to RGB

  #reshape to WxHx1
  width, height = mask.size
  mask = tf.reshape(mask, [width, height , 1])

  mask = tf.image.convert_image_dtype(mask, tf.float32)

  return tf.image.resize(mask, [256, 256])


def refine_tile(  model,  image_in, trimap_in  ):

    image_tensor = decode_img(image_in)
    trimap_tensor = decode_mask(trimap_in)

    image_trimap_in = tf.concat([image_tensor, trimap_tensor], axis=-1)
    #add None to batch
    image_trimap_in = tf.expand_dims(image_trimap_in, 0)
    output = model(image_trimap_in, training=False )

    output = output[0]


    alpha = tf.keras.preprocessing.image.array_to_img(output)

    background = Image.new("RGB", image_in.size, (10, 128,  10))
    white = Image.new("RGB", image_in.size, (255, 255, 255))


    #resize alpha to same size to image_in
    alpha = alpha.resize(image_in.size, Image.BILINEAR)
    #result = Image.composite( image_in, background, alpha )
    result = Image.composite( image_in, background, alpha )

    return result

def crop_is_ok( rect ,  trimap_in ):
    trimap = trimap_in.crop( rect )
    area = trimap.size[0] * trimap.size[1]
    #count zeros and 255
    ts = np.asarray(trimap)
    n0 = np.sum( ts == 0 )

    n255 = np.sum( ts == 255 )
    sc = 8

    if n0  < area//sc or n255 < area//sc:
        return False
    return True

def has_unknow(rect , trimap_in ):
    trimap = trimap_in.crop( rect )
    ts = np.asarray(trimap)
    n_low = np.sum( ts < 10 )
    n_high = np.sum( ts > 245 )
    area = trimap.size[0] * trimap.size[1]
    nmid = area - n_low - n_high
    if nmid  > 0  :
        return True
    return False


def expand_until_ok( rect ,  trimap ):
    if (has_unknow( rect ,trimap ) == False ):
        return None
    rect = (rect[0]-64, rect[1]-64, rect[2]+64, rect[3]+64) # first expand
    for it in range(2):
        if crop_is_ok( rect ,  trimap ):
            return   rect
        rect = (rect[0]-64, rect[1]-64, rect[2]+64, rect[3]+64)
        #check out of bounds
        if rect[0] < 0:
            rect = (0, rect[1], rect[2], rect[3])
        if rect[1] < 0:
            rect = (rect[0], 0, rect[2], rect[3])
        if rect[2] > trimap.size[0]:
            rect = (rect[0], rect[1], trimap.size[0], rect[3])
        if rect[3] > trimap.size[1]:
            rect = (rect[0], rect[1], rect[2], trimap.size[1])
    return  rect

def get_tiles(image, trimap, tile_size= 128, border = 64):
    w,h = image.size
    tiles = []

    for i in range( 0, w, tile_size //3   ):
        for j in range( 0, h,  tile_size//3   ):
            rect = (i,j, i+tile_size, j+tile_size)
            if rect[2] > w:
                rect = (w-tile_size, rect[1], w, rect[3])
            if rect[3] > h:
                rect = (rect[0], h-tile_size, rect[2], h)
            #crop image
            rect_center = (rect[0], rect[1], rect[2], rect[3])
            if has_unknow( rect ,  trimap ) :
                #expand 64 pixels
                rect = (rect[0]-border, rect[1]-border, rect[2]+border, rect[3]+border)
                if crop_is_ok( rect ,  trimap ):
                    image_tile = image.crop( rect )
                    trimap_tile = trimap.crop( rect )
                    tiles.append( (image_tile, trimap_tile, rect ,rect_center ) )
    return tiles



def main(image , trimap, output_filename,  border = 64 ):
    output =  image.copy()
    #check if gpu is available
    print(tf.__version__)
    print("CUDA available: " + str(tf.test.is_gpu_available()))

    unet = ResUnet()
    model =  unet.build_model( )
    model.summary()
    model.compiled_metrics == None

    if os.path.exists("weights/model_32.ckpt.index") :
        print("Load model from file ...")
        model.load_weights("weights/model_32.ckpt")
    else:
        raise ValueError("Model not found")

    N = 256 - border
    tiles =  get_tiles( image, trimap, tile_size= N , border = border )
    print(len(tiles))
    invocations  = 0
    for i, (image_tile, trimap_tile, rect_ex,rect_center ) in tqdm( enumerate( tiles) ):
        #save image and tile
        output_t = refine_tile( model, image_tile, trimap_tile )

        if output_t is not None:
            output_t_center = output_t.crop( (border,border, border+ N , border+ N ) )
            output.paste( output_t_center, rect_center )
            invocations += 1 #count invocations
    output.save( output_filename )

import sys
if __name__ == "__main__":
    image_rgb_filename = sys.argv[1]
    trimap_filename = sys.argv[2]

    output_filename = "output.jpg"
    if len(sys.argv) > 3:
        output_filename = sys.argv[3]

    #try load images
    try:
        image = Image.open(image_rgb_filename)
        trimap = Image.open(trimap_filename)
    except:
        print("Error loading images")
        exit(0)

    main(image , trimap , output_filename)