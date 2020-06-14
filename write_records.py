import tensorflow as tf
print(tf.__version__)
import os
import numpy as np

'''
2d array (spectrogram, mel-spectrogram) for write tfrecords files.

Authors: June-Woo Kim (kaen2891@gmail.com)
'''


def serialize_example(batch, enc_inp, dec_inp, tar_inp):
    filename = "./tf_records_example.tfrecords"
    writer = tf.io.TFRecordWriter(filename)
    
    for i in range(batch):
        feature = {}
        enc = enc_inp[i]
        dec = dec_inp[i]
        tar = tar_inp[i]
        print('enc shape {} dec shape {} tar shape {}'.format(enc.shape, dec.shape, tar.shape)) 
        feature['enc_inp'] = tf.train.Feature(float_list=tf.train.FloatList(value=enc.flatten()))
        feature['dec_inp'] = tf.train.Feature(float_list=tf.train.FloatList(value=dec.flatten()))
        feature['tar_inp'] = tf.train.Feature(float_list=tf.train.FloatList(value=tar.flatten()))
        
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
        print("{}th enc {} dec {} tar {} finished ".format(i, enc_inp[i], dec_inp[i], tar_inp[i]))
        


list_enc_inp = np.random.rand(50, 201, 21)
list_dec_inp = np.random.rand(50, 201, 22)
list_tar_inp = np.random.rand(50, 201, 22)
print('len enc {} dec {} tar {}'.format(len(list_enc_inp), len(list_dec_inp), len(list_tar_inp)))

serialize_example(len(list_enc_inp), list_enc_inp, list_dec_inp, list_tar_inp)
