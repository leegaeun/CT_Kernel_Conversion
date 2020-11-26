
from __future__ import absolute_import

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import lmdb
import numpy as np
import caffe
import SimpleITK as sitk
import PIL
from scipy import ndimage, misc


#############################################################
#  Functions
#############################################################

def _write_to_lmdb(db, key, value):
    """
    Write (key, value) to db
    """
    success = False
    while not success:
        txn = db.begin(write=True)
        try:
            txn.put(key, value)
            txn.commit()
            success = True
        except lmdb.MapFullError:
            txn.abort()
            
            # double the map_size
            curr_limit = db.info()['map_size']
            new_limit = curr_limit*2
            db.set_mapsize(new_limit)   # double it

            
def _save_mean(mean, filename):
    """
    Saves mean to file
    Arguments:
    mean -- the mean as an np.ndarray
    filename -- the location to save the image
    """
    if filename.endswith('.binaryproto'):
        blob = caffe.proto.caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels = 1
        blob.height, blob.width = mean.shape
        blob.data.extend(mean.astype(float).flat)
        with open(filename, 'wb') as outfile:
            outfile.write(blob.SerializeToString())
    elif filename.endswith(('.jpg', '.jpeg', '.png')):
        misc.imsave(filename, mean)
    else:
        raise ValueError('unrecognized file extension')



#############################################################
#  Dataset list
#############################################################
"""
Examples of data lists when conversion from B10f to B70f image.
1~8 are train data and 9~10 are test data.
"""

DATA_DIR = "/home/user/data/CT_KernelConversion_noncontrast/"
KERNEL_from = "10"
KERNEL_to = "70"


train_data_list = ["001", 
"002", 
"003", 
"004", 
"005", 
"006", 
"007", 
"008"]

test_data_list = ["009", 
"010"]



#############################################################
#  Train datasets
#############################################################

index = 0

train_image_db = lmdb.open("train_image_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)
train_label_db = lmdb.open("train_label_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)

image_sum = np.zeros((512, 512), 'float64')
image_bias = np.zeros((512, 512), 'uint16')

for data in train_data_list:
    image = sitk.ReadImage(DATA_DIR+data+"_B"+KERNEL_from+".mha")
    label = sitk.ReadImage(DATA_DIR+data+"_B"+KERNEL_to+".mha")
    num_of_slices = image.GetDepth()
    for i in xrange(num_of_slices):
        nda_img = sitk.GetArrayFromImage(image)[i,:,:]
        nda_label = sitk.GetArrayFromImage(label)[i,:,:]
        
        nda_img = nda_img + image_bias
        nda_label = nda_label + image_bias
        nda_label = nda_label - nda_img
        
        image_sum += nda_img
        
        image_datum = caffe.proto.caffe_pb2.Datum()
        image_datum.channels, image_datum.height, image_datum.width = 1, 512, 512
        image_datum.float_data.extend(nda_img.astype(float).flat)
        _write_to_lmdb(train_image_db, str(index+i), image_datum.SerializeToString())

        label_datum = caffe.proto.caffe_pb2.Datum()
        label_datum.channels, label_datum.height, label_datum.width = 1, 512, 512
        label_datum.float_data.extend(nda_label.astype(float).flat)
        _write_to_lmdb(train_label_db, str(index+i), label_datum.SerializeToString())
        
    index += num_of_slices

image_count = index

# close databases
train_image_db.close()
train_label_db.close()

# save mean
mean_image = (image_sum / image_count).astype('int16')
_save_mean(mean_image, "train_image_mean_"+KERNEL_from+"to"+KERNEL_to+".png")
_save_mean(mean_image, "train_image_mean_"+KERNEL_from+"to"+KERNEL_to+".binaryproto")



#############################################################
#  Test datasets
#############################################################

index = 0

test_image_db = lmdb.open("test_image_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)
test_label_db = lmdb.open("test_label_"+KERNEL_from+"to"+KERNEL_to, map_async=True, max_dbs=0)

#image_sum = np.zeros((512, 512), 'float64')
image_bias = np.zeros((512, 512), 'uint16')

for data in test_data_list:
    image = sitk.ReadImage(DATA_DIR+data+"_B"+KERNEL_from+".mha")
    label = sitk.ReadImage(DATA_DIR+data+"_B"+KERNEL_to+".mha")
    num_of_slices = image.GetDepth()
    for i in xrange(num_of_slices):
        nda_img = sitk.GetArrayFromImage(image)[i,:,:]
        nda_label = sitk.GetArrayFromImage(label)[i,:,:]
        
        nda_img = nda_img + image_bias
        nda_label = nda_label + image_bias
        nda_label = nda_label - nda_img
        
        #image_sum += nda_img
        
        image_datum = caffe.proto.caffe_pb2.Datum()
        image_datum.channels, image_datum.height, image_datum.width = 1, 512, 512
        image_datum.float_data.extend(nda_img.astype(float).flat)
        _write_to_lmdb(test_image_db, str(index+i), image_datum.SerializeToString())

        label_datum = caffe.proto.caffe_pb2.Datum()
        label_datum.channels, label_datum.height, label_datum.width = 1, 512, 512
        label_datum.float_data.extend(nda_label.astype(float).flat)
        _write_to_lmdb(test_label_db, str(index+i), label_datum.SerializeToString())
        
    index += num_of_slices

#image_count = index

# close databases
test_image_db.close()
test_label_db.close()

