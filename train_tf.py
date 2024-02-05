import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.0.6"

import glob
import tensorflow as tf
from resunet_tf import ResUnet

import keras.backend as K
from tqdm import tqdm

batch_size = 16
img_height = 256
img_width = 256


sobelFilter = K.variable([[[[1.,  1.]], [[0.,  2.]],[[-1.,  1.]]],
                      [[[2.,  0.]], [[0.,  0.]],[[-2.,  0.]]],
                      [[[1., -1.]], [[0., -2.]],[[-1., -1.]]]])
def expandedSobel(inputTensor):
    #this considers data_format = 'channels_last'
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    #if you're using 'channels_first', use inputTensor[0,:,0,0] above
    return sobelFilter * inputChannels

def sobelLoss(yTrue,yPred):
    #get the sobel filter repeated for each input channel
    filt = expandedSobel(yTrue)
    #calculate the sobel filters for yTrue and yPred
    #this generates twice the number of input channels
    #a X and Y channel for each input channel
    sobelTrue = K.depthwise_conv2d(yTrue,filt)
    sobelPred = K.depthwise_conv2d(yPred,filt)

    #now you just apply the mse:
    return K.mean(K.square(sobelTrue - sobelPred))




def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3) #color images
  img = tf.image.convert_image_dtype(img, tf.float32)
   #convert unit8 tensor to floats in the [0,1]range
  return tf.image.resize(img, [256, 256])


def decode_mask(mask):
  mask = tf.image.decode_jpeg(mask, channels=1) #greyscale images
  mask = tf.image.convert_image_dtype(mask, tf.float32)
  return tf.image.resize(mask, [256, 256])

@tf.function
def load_image_train(image_ifilename ):


    _trimap_ifilename = tf.strings.regex_replace(image_ifilename, "image", "trimap")
    _refine_ifilename = tf.strings.regex_replace(image_ifilename, "image", "refined")

    #replace .jpg by .png
    _trimap_ifilename = tf.strings.regex_replace(_trimap_ifilename, ".jpg", ".png")
    _refine_ifilename = tf.strings.regex_replace(_refine_ifilename, ".jpg", ".png")

    _image = decode_img(tf.io.read_file(image_ifilename))
    _trimap = decode_mask(tf.io.read_file(_trimap_ifilename))
    _refine = decode_mask(tf.io.read_file(_refine_ifilename))


    # Randomly choosing the images to flip right left.
    # We need to split both the input image and the input mask as the mask is in correspondence to the input image.
    if tf.random.uniform(()) > 0.5:
        _image = tf.image.flip_left_right(_image)
        _trimap = tf.image.flip_left_right(_trimap)
        _refine = tf.image.flip_left_right(_refine)

    # radom change hue, saturation, brightness and contrast
    _image = tf.image.random_hue(_image, 0.1)
    _image = tf.image.random_saturation(_image, 0.5, 2)
    _image = tf.image.random_brightness(_image, 0.2)
    _image = tf.image.random_contrast(_image, 0.5, 2)


    # Normalizing the input image.
    _image = tf.cast(_image, tf.float32) / 255.0

    #concat trimap as new channel
    #_image = tf.concat( [ _image, _trimap ], axis=2 )

    # Returning the input_image and the input_mask
    return _image, _trimap, _refine


def train_model(model , data_folder ):
    #get all images in the folder
    #files = glob.glob( data_folder + "/image/*.jpg" )
    dataset = tf.data.Dataset.list_files(data_folder + "/image/*.jpg" , shuffle=True )
    train = dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for image, xmap, refined  in train.take(1):
        print("Image shape: ", image.shape)
        print("Refined shape: ", refined.shape)

    print("Test one image ...", image.shape)

    # add batch dimension
    image = tf.expand_dims(image, 0)

    #result = model.predict(image)
    #print("Result shape: ", result.shape)


    # Compile the model
    optimizer_sgd =  tf.keras.optimizers.legacy.SGD (learning_rate=0.001)
    #optimizer = torch.optim.Adam( model.parameters(), lr=0.0005 , betas=(0.9,0.999),eps=1e-08  )

    optimizer_adam = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08 )

    optimizer = optimizer_adam

    #loss_fn = tf.keras.losses.MeanSquaredError()

    #loss is a sum of mse , abs amd sobel
    loss_mse = tf.keras.losses.MeanSquaredError()
    loss_abs = tf.keras.losses.MeanAbsoluteError()
    #model.compile(optimizer=optimizer, loss_fn='mse', metrics=['accuracy'])
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    mean_loss = None
    total_images = train.cardinality().numpy()
    enum_tqdm =   tqdm(enumerate(train.batch(16)), total=total_images// 16 )

    #only one epoch

    #for step, (x_batch_train, x_trimap, y_batch_train) in enumerate(train.batch(32)):
    for step, (x_batch_train, x_trimap, y_batch_train) in   enum_tqdm  :
        xt_batch_train = tf.concat( [ x_batch_train, x_trimap ], axis=3 )
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:
            y_refined = model(xt_batch_train, training=True)  # y_refined for this minibatch
            loss_value = loss_mse(y_batch_train, y_refined)
            loss_value += loss_abs(y_batch_train, y_refined)
            loss_value += sobelLoss(y_batch_train, y_refined)

            if mean_loss is None:
                mean_loss = loss_value
            else:
                mean_loss = 0.95 * mean_loss + 0.05 * loss_value

            #print("Loss: ", mean_loss)
            mean_loss_value = mean_loss.numpy()
            enum_tqdm.set_description("Loss: %.4f" % mean_loss_value)


            if (step % 50 == 0):
                #save image
                #print("Save image ...")
                #convert to RGB only
                x_rgb = x_batch_train[0]
                #x_rgb = x_rgba[:,:,:3]
                tf.keras.preprocessing.image.save_img( "x_input.jpg", x_rgb)
                #get only alpha channel from y_batch_train
                x_aa = x_trimap[0]
                #add extra channel to make it 256x256x1
                #x_aa = tf.expand_dims(x_aa, -1)

                tf.keras.preprocessing.image.save_img( "x_trimap.jpg",x_aa )
                tf.keras.preprocessing.image.save_img( "x_refined.jpg", y_refined[0] )

            if (step % 500 == 0):
                model.save_weights('weights/model_32.ckpt')

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))



    # Save the model
    model.save_weights('weights/model_32.ckpt')


def main():
    #check if gpu is available
    print(tf.__version__)
    print("CUDA available: " + str(tf.test.is_gpu_available()))

    #abort if not gpu is available
    if not tf.test.is_gpu_available():
        print("No GPU available")
        exit(0)




    unet = ResUnet()
    model =  unet.build_model( )
    model.summary()
    model.compiled_metrics == None
    #exit(0)


    #existe an model.h5 file ?
    if os.path.exists("weights/model_32.ckpt.index") :
        print("Load model from file ...")
        model.load_weights("weights/model_32.ckpt")

    else:
        print("Create new model ...")


    #train_model( model , "H:\\dev\\p3m_tiles\\p3m")
    train_model( model , "/home/astro/ml/p3m/")




if __name__ == "__main__":
    main()