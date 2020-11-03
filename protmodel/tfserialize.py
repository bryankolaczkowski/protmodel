################################################################################
# library for serializing tensorflow data records
#
# (C) 2020 Bryan Kolaczkowski, University of Florida, Gainesville, FL USA
# Released under GNU General Public License (GPL)
# bryank@ufl.edu
################################################################################

import tensorflow as tf

### tensorflow description of labelled training data instance for 1D data ###
data_feature_description = {
  # ravel-ed array of data
  # Nx11 array, where N is the sequence length; 11 residue features
  'dta' : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True,
                                        default_value=0.0),
  # ravel-ed training label for this data array
  # Nx20 array, where N is the sequence length; 20 residues
  'lbl' : tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True,
                                        default_value=0.0)
}

### list of floats from list of floats ###
def _float_features(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

### BEG USER FUNCTIONS #########################################################
#
### serialize data to byte string ###
def serialize_data(data, label):
  feature = {
    'dta' : _float_features(data.ravel()),
    'lbl' : _float_features(label.ravel()),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature)) \
                          .SerializeToString()

### deserialize data from byte string ###
@tf.function
def deserialize_data(rawdata):
  return tf.io.parse_single_example(rawdata, data_feature_description)
#
### END USER FUNCTIONS #########################################################
