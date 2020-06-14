import tensorflow as tf

'''
This code is for read form tfrecords file and get batch for training in tensorflow 2.0

In tf_records_example.tfrecords files, dataset is consist of 2d array with 50 batch size. So the enc_inp shape is (50, 201, 21) and the dec_inp and tar_inp shape is (50, 201, 22)

You can use this code for tfrecords file to training

Authors: June-Woo Kim (kaen2891@gmail.com)
'''

spectrum_feature_description = {
    'enc_inp': tf.io.FixedLenSequenceFeature ([], tf.float32, allow_missing=True),
    'dec_inp': tf.io.FixedLenSequenceFeature ([], tf.float32, allow_missing=True),
    'tar_inp': tf.io.FixedLenSequenceFeature ([], tf.float32, allow_missing=True)
}

def _parse_spec_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, spectrum_feature_description)


def input_fn(record_file, batch_size, buffer_size):
    dataset = tf.data.TFRecordDataset(record_file)
    print('raw_dataset', dataset) # ==> raw_dataset <TFRecordDatasetV2 shapes: (), types: tf.string>    

    parsed_spec_dataset = dataset.map(_parse_spec_function)
    print('map', parsed_spec_dataset)
    
    #parsed_spec_dataset = parsed_spec_dataset.cache()
    #print('cache', parsed_spec_dataset)
    
    train_dataset = parsed_spec_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    print('buffer + batch', train_dataset)
    
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    print('train_dataset autotune', train_dataset)
    
    return train_dataset


record_file = './tf_records_example.tfrecords'
batch_size = 20
train_dataset = input_fn(record_file, batch_size=batch_size, buffer_size=10)
print(train_dataset)

for (batch, spec) in enumerate(train_dataset):
    enc_raw = spec['enc_inp'].numpy()
    enc_raw = tf.reshape(enc_raw, [batch_size, 201, 21]) # batch, d_model, seq_len        
             
            
    dec_raw = spec['dec_inp'].numpy()
    dec_raw = tf.reshape(dec_raw, [batch_size, 201, 22])
            
    tar_raw = spec['tar_inp'].numpy()
    tar_raw = tf.reshape(tar_raw, [batch_size, 201, 22])
    
    print('batch = {}, enc_raw = {}, dec_raw = {}, tar_raw = {}'.format(batch, enc_raw.shape, dec_raw.shape, tar_raw.shape))
    
    print(enc_raw[0])

