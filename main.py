import tensorflow as tf
import numpy as np
import os
import utils as ut
import models as models
import Spectograms as spectograms
import SoundUtils as SU
import scipy.misc
from PIL import Image

content_sound = "sound/content_fcjf0_SA1.WAV"
style_sound = "sound/style_mcpm0_SA1.WAV"

model_weights_path = '/home/ubuntu/vgg'

# content_weights_path = "/Volumes/Storage/Datasets/TIMIT/Phonemes/fc7/p_weights.npy"
# content_bias_path = "/Volumes/Storage/Datasets/TIMIT/Phonemes/fc7/p_bias.npy"

# style_weights_path = "/Volumes/Storage/Datasets/TIMIT/Speakers/fc7/s_weights.npy"
# style_bias_path = "/Volumes/Storage/Datasets/TIMIT/Speakers/fc7/s_bias.npy"
content_weights_path = "Weights/p_weights.npy"
content_bias_path = "Weights/p_bias.npy"

style_weights_path = "Weights/s_weights.npy"
style_bias_path = "Weights/s_bias.npy"


array_inversion_file = "/Users/Sergey/Thesis/SpectrogramInversion1.02b/invertMe.mat"
array_inversion_shown_file = "/Users/Sergey/Thesis/SpectrogramInversion1.02b/invertMeShown.mat"

full_spect_image = "full_spect.png"
temp_folder = "Temp"
images_folder = "Images"
style_images_folder = "StyleImages"
transformed_images_folder = "TransformedImages"

device="/cpu:0"

windowSize = 256
frameStep = 64

num_iters = 50000

#def ReshapeToVector(input_x):
#    shape = tf.shape(input_x)
#    dim = 1
#    for d in shape[1:]:
#        dim *= d
#    x = tf.reshape(input_x, [-1, dim])
#    return x

def SeparateIntoChunksAndSave(sound, folder):
    #creating sequence of images
    wav_file = SU.PCM2Wav(sound, temp_folder)
    imageRGB, minEl, maxEl = spectograms.SpectogramToImage(wav_file, full_spect_image)

    chunkLength = 11
    totalFeatures = imageRGB.shape[1]
    #The stepLength is 1 therefore the number of chunks is calculated as follows
    numChunks = totalFeatures-chunkLength + 1

    for i in range(numChunks):
        chunk = imageRGB[:,i:i+chunkLength,:]
        chunk = scipy.misc.imresize(chunk, (224, 224))
        pa = folder + "/" + str(i) + ".png"
        spectograms.SaveRGB(chunk, pa)
        

SeparateIntoChunksAndSave(content_sound, images_folder)
SeparateIntoChunksAndSave(style_sound, style_images_folder)

print "Loading parameters"
params = np.load(model_weights_path).item()
content_weights = np.load(content_weights_path)
content_bias = np.load(content_bias_path)
style_weights = np.load(style_weights_path)
style_bias = np.load(style_bias_path)


for chunk_number in range(89,90):
    print
    #print "CHUNK NUMBER: " + str(chunk_number)
    #i_img_path = images_folder + "/" + str(chunk_number) + ".png"
    #img = spectograms.ReadRGB(i_img_path)

    #TEST
    chunk_number = 549
    test_image = images_folder + "/" + "549.png"
    img = spectograms.ReadRGB(test_image)
    
    #style_image_path = style_images_folder + "/" + "579.png"
    #style_img = spectograms.ReadRGB(style_image_path)
    #style_image = ut.process_image(style_img)

    g = tf.Graph()
    content_image = ut.process_image(img)
    wanted_style = np.array([[0,1.0]])
    with g.device(device), g.as_default(), tf.Session(graph=g, config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    
        print "Load content values and calculate style and content softmaxes"
        #content
        cW = tf.constant(content_weights)
        cB = tf.constant(content_bias)
        #style
        sW = tf.constant(style_weights)
        sB = tf.constant(style_bias)
        wanted_style = tf.constant(wanted_style, tf.float32)

        image = tf.constant(content_image)
        model = models.getModel(image, params)
        pool2_image_val = sess.run(model.y())
	#fc7_image_val = sess.run(model.y())
	reshaped_pool2_image_val = tf.reshape(pool2_image_val, [-1, 401408])
#ReshapeToVector(pool2_image_val)
        content_values= tf.nn.softmax(tf.matmul(reshaped_pool2_image_val,cW) + cB)
        ##style_values= tf.nn.softmax(tf.matmul(pool2_image_val,sW) + sB)

        print "Generate noise"
        ##gen_image = tf.Variable(tf.truncated_normal(content_image.shape, stddev=20), trainable=True, name='gen_image')

        gen_image = tf.Variable(tf.constant(np.array(content_image, dtype=np.float32)), trainable=True, name='gen_image')
        ##gen_image = tf.Variable(tf.constant(np.array(style_image, dtype=np.float32)), trainable=True, name='gen_image')
        
        model_gen = models.getModel(gen_image, params)
        pool2_gen_image_val = model_gen.y()
	reshaped_pool2_gen_image_val = tf.reshape(pool2_gen_image_val, [-1, 401408]) 
#ReshapeToVector(pool2_gen_image_val)

        content_transformed_values= tf.nn.softmax(tf.matmul(reshaped_pool2_gen_image_val,cW) + cB)
        style_transformed_values= tf.nn.softmax(tf.matmul(reshaped_pool2_gen_image_val,sW) + sB)

        L = 0.0
        L_Content = 0.0
        L_Style = 0.0
        #L += tf.nn.l2_loss(fc7_gen_image_val - fc7_image_val)


	L_Content += -tf.reduce_sum(content_values*tf.log(tf.clip_by_value(content_transformed_values,1e-10,1.0)))
	L_Style += -tf.reduce_sum(wanted_style*tf.log(tf.clip_by_value(style_transformed_values,1e-10,1.0)))
        #L_Content += tf.nn.l2_loss(content_transformed_values - content_values)
        #L_Style += tf.nn.l2_loss(style_transformed_values - wanted_style)

        # The loss
        L=L_Content + L_Style
    
        # The optimizer
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step, 
                                               decay_steps=100, decay_rate=0.94, staircase=True)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(L, global_step=global_step)

        #tf.scalar_summary("L_content", L_Content)

        print "Start calculation..."
        # The optimizer has variables that require initialization as well
        sess.run(tf.initialize_all_variables())
        for i in range(num_iters):
            if i % 10 == 0:
                print "Iter:", i
                print "L_content", sess.run(L_Content)
                print "L_style", sess.run(L_Style)
                print "L", sess.run(L)
                # Increment summary
                #sess.run(tf.assign(gen_image_addmean, add_mean(gen_image_val)))
                #summary_str = sess.run(summary_op)
                #summary_writer.add_summary(summary_str, i)
            if i % 50 == 0:
                gen_image_val = sess.run(gen_image)
                #saving image
                img = gen_image_val.copy()
                img = ut.add_mean(img)
                img = np.clip(img[0, ...],0,255).astype(np.uint8)
                p = transformed_images_folder + "/" + str(chunk_number) + "_" + str(i) + ".png"
                #save_image(gen_image_val, i, transformed_images_folder)
                spectograms.SaveRGB(img, p)
            sess.run(train_step)
