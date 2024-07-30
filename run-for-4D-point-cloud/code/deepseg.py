import os, time, math
import nibabel as nib, numpy as np
import tensorflow as tf
import glob
from image_utils import *

def deeplearningseg(model_path, test_dir, atlas_dir):           
     
   with tf.Session() as sess:
       
        sess.run(tf.global_variables_initializer())

        # Import the computation graph and restore the variable values
        # 调用和加载分割模型的神经网络权重
        saver = tf.train.import_meta_graph('{0}.meta'.format(model_path))
        saver.restore(sess, '{0}'.format(model_path))
        
        # Process each subject subdirectory
        table_time = []

        # 如果 test_dir目录下已经存在 subjnames.txt 文件则删除该文件以外的所有.txt 文件并确保 subjnames.txt 文件存在；
        # 如果 subjnames.txt 不存在，则创建一个空的 subjnames.txt 文件
        if os.path.exists('{0}/subjnames.txt'.format(test_dir)):
            os.system('rm {0}/*.txt'.format(test_dir))
        os.system('touch {0}/subjnames.txt'.format(test_dir))
        for data in sorted(os.listdir(test_dir)):
            
            print(data)
            
            data_dir = os.path.join(test_dir, data)
            
            if not os.path.isdir(data_dir):
                print('  {0} is not a valid directory, Skip'.format(data_dir))
                continue
            
            file = open('{0}/subjnames.txt'.format(test_dir),'a')
            file.write('{0}\n'.format(data))
            file.close()
            
            if os.path.exists('{0}/PHsegmentation_ED.gipl'.format(data_dir)):
                os.system('rm {0}/*.gipl'.format(data_dir))
            if os.path.exists('{0}/lvsa_.nii.gz'.format(data_dir)):
                os.system('rm {0}/lvsa_*.nii.gz'.format(data_dir))
                os.system('rm {0}/seg_*.nii.gz'.format(data_dir))
            
            originalnii = glob.glob('{0}/*.nii'.format(data_dir))
            # 执行imagePreprocessing
            # 1. 处理nifit文件并保存头文件信息
            # 2. 对图像进行时间轴的对齐
            # 3. 应用了自动对比度调整
            # 4. 用于心脏相位检测检测舒张期末期和收缩期末期
            # 以上预处理操作已经封装在 Docker 镜像的环境中，不能显式调用
            if not originalnii:
                print('  original nifit image does not exist, use lvsa.nii.gz')
                originalnii = glob.glob('{0}/*.nii.gz'.format(data_dir))  
                imagePreprocessing(originalnii[0], data_dir, atlas_dir) 
            else:
                print('  start image preprocessing ...')
                imagePreprocessing(originalnii[0], data_dir, atlas_dir)
            
            # Process ED and ES time frames
            image_ED_name = '{0}/lvsa_{1}.nii.gz'.format(data_dir, 'ED')
            image_ES_name = '{0}/lvsa_{1}.nii.gz'.format(data_dir, 'ES')
   
            if not os.path.exists(image_ED_name) or not os.path.exists(image_ES_name):
                print(' Image {0} or {1} does not exist. Skip.'.format(image_ED_name, image_ES_name))
                continue
            # 判断是否重复运行
            if os.path.exists('{0}/{1}'.format(data_dir, 'dofs')) or \
               os.path.exists('{0}/{1}'.format(data_dir, 'segs')) or \
               os.path.exists('{0}/{1}'.format(data_dir, 'tmps')) or \
               os.path.exists('{0}/{1}'.format(data_dir, 'sizes')) or \
               os.path.exists('{0}/{1}'.format(data_dir, 'motion')) or \
               os.path.exists('{0}/{1}'.format(data_dir, 'vtks')):
                    
                os.system('rm -rf {0}/{1}'.format(data_dir, 'dofs'))
                os.system('rm -rf {0}/{1}'.format(data_dir, 'segs'))
                os.system('rm -rf {0}/{1}'.format(data_dir, 'tmps'))
                os.system('rm -rf {0}/{1}'.format(data_dir, 'sizes'))
                os.system('rm -rf {0}/{1}'.format(data_dir, 'motion'))
                os.system('rm -rf {0}/{1}'.format(data_dir, 'vtks'))
                
                os.mkdir('{0}/{1}'.format(data_dir, 'dofs'))
                os.mkdir('{0}/{1}'.format(data_dir, 'segs'))
                os.mkdir('{0}/{1}'.format(data_dir, 'tmps'))
                os.mkdir('{0}/{1}'.format(data_dir, 'sizes'))
                os.mkdir('{0}/{1}'.format(data_dir, 'motion'))
                os.mkdir('{0}/{1}'.format(data_dir, 'vtks'))
                
            else: 
                
                os.mkdir('{0}/{1}'.format(data_dir, 'dofs'))
                os.mkdir('{0}/{1}'.format(data_dir, 'segs'))
                os.mkdir('{0}/{1}'.format(data_dir, 'tmps'))
                os.mkdir('{0}/{1}'.format(data_dir, 'sizes'))
                os.mkdir('{0}/{1}'.format(data_dir, 'motion'))
                os.mkdir('{0}/{1}'.format(data_dir, 'vtks'))
            # 对ED和ES图像进行深度学习的分割操作
            for fr in ['ED', 'ES']:
                # lvsa_ED or lvsa_ES
                image_name = '{0}/lvsa_{1}.nii.gz'.format(data_dir, fr)

                # Read the image
                print('  Reading {} ...'.format(image_name))
                nim = nib.load(image_name) # dim == [W,H,Z,1]
                image = nim.get_data()

                imageOrg = np.squeeze(image, axis=-1).astype(np.int16) # 移除颜色通道维度 dim == [W,H,Z]
                tmp = imageOrg

                X, Y, Z = image.shape[:3] # 确保取前三个维度
                
                print('  Segmenting {0} frame ...'.format(fr)) # 会执行两次，一次分割ED，一次分割ES
              
                # print('  Segmenting {0} frame {1} ...'.format(fr, slice))
                start_seg_time = time.time()
                
                for slice in range(Z): # 获取每层图像
                    
                    image = imageOrg[:,:,slice]
                    
                    if image.ndim == 2:
                        image = np.expand_dims(image, axis=2) # dim == [W,H,1]
                        
                        # Intensity rescaling
                        image = rescale_intensity(image, (1, 99))
                        # Pad the image size to be a factor of 16 so that the downsample and upsample procedures
                        # in the network will result in the same image size at each resolution level.
                        X2, Y2 = int(math.ceil(X / 16.0)) * 16, int(math.ceil(Y / 16.0)) * 16
                        x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
                        x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
                        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0)), 'constant')
                        
                        # Transpose the shape to NXYC
                        image = np.transpose(image, axes=(2, 0, 1)).astype(np.float32)
                        image = np.expand_dims(image, axis=-1)
                        
                        # Evaluate the network 网络预测
                        prob, pred = sess.run(['probE:0', 'predR:0'], feed_dict={'image:0': image, 'training:0': False})
                        
                        # Transpose and crop the segmentation to recover the original size
                        pred = np.transpose(pred, axes=(1, 2, 0))
                        
                        pred = pred[x_pre:x_pre + X, y_pre:y_pre + Y]
                        pred = np.squeeze(pred, axis=-1).astype(np.int16)
                        tmp[:,:,slice] = pred
                    
                seg_time = time.time() - start_seg_time
                print('  Segmentation time = {:3f}s'.format(seg_time))
                table_time += [seg_time]

                pred = tmp
        
                nim2 = nib.Nifti1Image(pred, nim.affine)
                nim2.header['pixdim'] = nim.header['pixdim']
                nib.save(nim2, '{0}/segs/seg_lvsa_{1}.nii.gz'.format(data_dir, fr))

        print('Average segmentation time = {:.3f}s per frame'.format(np.mean(table_time)))
