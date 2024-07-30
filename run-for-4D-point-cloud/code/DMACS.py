import os.path
import time
import numpy as np
import tensorflow as tf
from deepseg20240513 import *
from p1processing import *
from p2processing import *
from meshfitting import *
from motionEstimation import *
from decimation import *

""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('coreNo', 3, 'Number of CPUs.')
#tf.app.flags.DEFINE_string('test_dir', '/vol/medic02/users/jduan/New_BRIDGE_Cases4',
#                           'Path to the test set directory, under which images are organised in '
#                           'subdirectories for each subject.')
#tf.app.flags.DEFINE_string('model_path', '/vol/medic02/users/jduan/HHData/tensorflowFCNCodes/DeepRegionEdgeSegmentation'
#                           '/saver/model/vgg_RE_network/vgg_RE_network.ckpt-50000', 'Path to the saved trained model.')
#tf.app.flags.DEFINE_string('atlas_dir', '/vol/medic02/users/jduan/HHData/3Dshapes_new', 'Path to the atlas.')
#tf.app.flags.DEFINE_string('param_dir', '/vol/medic02/users/jduan/myPatchMatch/par', 'Path to the registration parameters.')
#tf.app.flags.DEFINE_string('template_dir', '/vol/medic02/users/jduan/myPatchMatch/template_wenzhe', 'Path to the template.')
#tf.app.flags.DEFINE_string('template_PH', '/vol/medic02/users/jduan/myPatchMatch/vtks', 'Path to the template.')


CurrentFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
tf.app.flags.DEFINE_integer('coreNo', 8, 'Number of CPUs.')
tf.app.flags.DEFINE_string('test_dir', os.path.join(CurrentFolder, "repair_219_data"),
                           'Path to the test set directory, under which images are organised in '
                           'subdirectories for each subject.')
tf.app.flags.DEFINE_string('model_path',  os.path.join(CurrentFolder,"model/vgg_RE_network.ckpt-50000"), 'Path to the saved trained model.')
tf.app.flags.DEFINE_string('atlas_dir',   os.path.join(CurrentFolder,'refs'), 'Path to the atlas.')
tf.app.flags.DEFINE_string('param_dir',  os.path.join(CurrentFolder,'par'), 'Path to the registration parameters.')
tf.app.flags.DEFINE_string('template_dir',  os.path.join(CurrentFolder,'vtks/1'), 'Path to the template.')
tf.app.flags.DEFINE_string('template_PH',  os.path.join(CurrentFolder,'vtks/2'), 'Path to the template.')
tf.app.flags.DEFINE_boolean('irtk', True, 'use irtk or not')

if __name__ == '__main__':
    
        print('Start evaluating on the test set ...')
        table_time = []
        start_time = time.time()
        # Run the code and calculate the mask labels for ES and ED by pre training the weights
        #################################################################################################################################
        # deeplearningseg(FLAGS.model_path, FLAGS.test_dir, FLAGS.atlas_dir)

        # # Perform 2D non rigid registration, hoping to apply labels to other cardiac cycle phases
        # # multiatlasreg2D(FLAGS.test_dir, FLAGS.atlas_dir, FLAGS.param_dir, FLAGS.coreNo, True, FLAGS.irtk) # parallel, irtk

        #################################################################################################################################
        # Perform 3D non rigid registration, hoping to apply labels to other cardiac cycle phases
        multiatlasreg3D(FLAGS.test_dir, FLAGS.atlas_dir, FLAGS.param_dir, FLAGS.coreNo, True, FLAGS.irtk) # parallel, irtk
        # Generate mesh mesh
        meshCoregstration(FLAGS.test_dir, FLAGS.param_dir, FLAGS.template_dir, FLAGS.coreNo, True, False) # parallel, irtk
        # Motion tracking
        motionTracking(FLAGS.test_dir, FLAGS.param_dir, FLAGS.template_PH, FLAGS.coreNo, True) # parallel

        decimate(FLAGS.test_dir, FLAGS.coreNo, False)

        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s in total for processing all the subjects).'.format(process_time))
import os.path
import time
import numpy as np
import tensorflow as tf
from deepseg20240513 import *
from p1processing import *
from p2processing import *
from meshfitting import *
from motionEstimation import *
from decimation import *

""" Deployment parameters """
FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('coreNo', 3, 'Number of CPUs.')
#tf.app.flags.DEFINE_string('test_dir', '/vol/medic02/users/jduan/New_BRIDGE_Cases4',
#                           'Path to the test set directory, under which images are organised in '
#                           'subdirectories for each subject.')
#tf.app.flags.DEFINE_string('model_path', '/vol/medic02/users/jduan/HHData/tensorflowFCNCodes/DeepRegionEdgeSegmentation'
#                           '/saver/model/vgg_RE_network/vgg_RE_network.ckpt-50000', 'Path to the saved trained model.')
#tf.app.flags.DEFINE_string('atlas_dir', '/vol/medic02/users/jduan/HHData/3Dshapes_new', 'Path to the atlas.')
#tf.app.flags.DEFINE_string('param_dir', '/vol/medic02/users/jduan/myPatchMatch/par', 'Path to the registration parameters.')
#tf.app.flags.DEFINE_string('template_dir', '/vol/medic02/users/jduan/myPatchMatch/template_wenzhe', 'Path to the template.')
#tf.app.flags.DEFINE_string('template_PH', '/vol/medic02/users/jduan/myPatchMatch/vtks', 'Path to the template.')


CurrentFolder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
tf.app.flags.DEFINE_integer('coreNo', 8, 'Number of CPUs.')
tf.app.flags.DEFINE_string('test_dir', os.path.join(CurrentFolder, "repair_219_data"),
                           'Path to the test set directory, under which images are organised in '
                           'subdirectories for each subject.')
tf.app.flags.DEFINE_string('model_path',  os.path.join(CurrentFolder,"model/vgg_RE_network.ckpt-50000"), 'Path to the saved trained model.')
tf.app.flags.DEFINE_string('atlas_dir',   os.path.join(CurrentFolder,'refs'), 'Path to the atlas.')
tf.app.flags.DEFINE_string('param_dir',  os.path.join(CurrentFolder,'par'), 'Path to the registration parameters.')
tf.app.flags.DEFINE_string('template_dir',  os.path.join(CurrentFolder,'vtks/1'), 'Path to the template.')
tf.app.flags.DEFINE_string('template_PH',  os.path.join(CurrentFolder,'vtks/2'), 'Path to the template.')
tf.app.flags.DEFINE_boolean('irtk', True, 'use irtk or not')

if __name__ == '__main__':

        print('Start evaluating on the test set ...')
        table_time = []
        start_time = time.time()
        # Run the code and calculate the mask labels for ES and ED by pre training the weights
        #################################################################################################################################
        # deeplearningseg(FLAGS.model_path, FLAGS.test_dir, FLAGS.atlas_dir)

        # # Perform 2D non rigid registration, hoping to apply labels to other cardiac cycle phases
        # # multiatlasreg2D(FLAGS.test_dir, FLAGS.atlas_dir, FLAGS.param_dir, FLAGS.coreNo, True, FLAGS.irtk) # parallel, irtk

        #################################################################################################################################
        # Perform 3D non rigid registration, hoping to apply labels to other cardiac cycle phases
        multiatlasreg3D(FLAGS.test_dir, FLAGS.atlas_dir, FLAGS.param_dir, FLAGS.coreNo, True, FLAGS.irtk) # parallel, irtk
        # Generate mesh mesh
        meshCoregstration(FLAGS.test_dir, FLAGS.param_dir, FLAGS.template_dir, FLAGS.coreNo, True, False) # parallel, irtk
        # Motion tracking
        motionTracking(FLAGS.test_dir, FLAGS.param_dir, FLAGS.template_PH, FLAGS.coreNo, True) # parallel

        decimate(FLAGS.test_dir, FLAGS.coreNo, False)

        process_time = time.time() - start_time
        print('Including image I/O, CUDA resource allocation, '
              'it took {:.3f}s in total for processing all the subjects).'.format(process_time))
