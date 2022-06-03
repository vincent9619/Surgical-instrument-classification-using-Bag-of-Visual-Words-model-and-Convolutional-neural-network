import tensorflow as tf
# from tensorflow import keras
from keras.utils.np_utils import to_categorical 
# from keras.applications.resnet50 import ResNet50
# from keras.applications.resnet101 import ResNet101
import keras.applications as kap
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense
from keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
# from keras.applications.resnext import ResNext

import numpy as np

# tf.keras.models.load_model(model_path)
from utils import load_imgs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# vram limit for eficiency
# config = tf.ConfigProto()
config=tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess=tf.compat.v1.Session(config=config)

# define class names
classes = ['1.1mm_K-Wire',
          '6_Babcock_Tissue_Forceps',
           '6_Mayo_Needle_Holder',
            '7_Metzenbaum_Scissors',
             '7_Microvascular_Needle_Holder',
              '8_Babcock_Tissue_Forceps' ,
              '8_Mayo_Needle_Holder' ,
              '8_Microvascular_Needle_Holder' ,
              '9_DeBakey_Dissector' ,
              '9_DeBakey_Needle_Holder',
              '9_Metzenbaum_Scissors' ,
              'Allis_Tissue_Forceps' ,
              'angled-post',
              'Ball___Socket_Towel_Clips' ,
              'Bearing_Plates_3_Hole_7_Peg_Left',
              'Bearing_Plates_3_Hole_7_Peg_Left_Narrow',
              'Bearing_Plates_3_Hole_7_Peg_Right',
              'Bearing_Plates_3_Hole_7_Peg_Right_Narrow',
              'Bearing_Plates_5_Hole_5_Peg_Left',
              'Bearing_Plates_5_Hole_5_Peg_Right',
              'Bearing_Plates_5_Hole_7_Peg_Left',  
              'Bearing_Plates_5_Hole_7_Peg_Right',
              'Bearing_Plates_7_Hole_7_Peg_Left',
              'Bearing_Plates_7_Hole_7_Peg_Right',
              'Bonneys_Non_Toothed_Dissector' ,
              'Bonneys_Toothed_Dissector',
              'chuck-no3',
              'chuck-no4',
              'Clavicle_Plate_3.5_6_Hole_Left',
              'Clavicle_Plate_3.5_6_Hole_Right',  
              'Clavicle_Plate_3.5_7_Hole_Left',
              'Clavicle_Plate_3.5_7_Hole_Right',
              'Clavicle_Plate_3.5_8_Hole_Left',
              'Clavicle_Plate_3.5_8_Hole_Right',
              'Clavicle_Plate_3.5-2.7_3_Hole_Left',
              'Clavicle_Plate_3.5-2.7_3_Hole_Right',
              'Clavicle_Plate_3.5-2.7_4_Hole_Left',  
              'Clavicle_Plate_3.5-2.7_4_Hole_Right',
              'Clavicle_Plate_3.5-2.7_5_Hole_Left',
              'Clavicle_Plate_3.5-2.7_5_Hole_Right',
              'Clavicle_Plate_3.5-2.7_6_Hole_Left',
              'Clavicle_Plate_3.5-2.7_6_Hole_Right',
              'Clavicle_Plate_3.5-2.7_7_Hole_Left',
              'Clavicle_Plate_3.5-2.7_7_Hole_Right',  
              'Clavicle_Plate_3.5-2.7_8_Hole_Left',
              'Clavicle_Plate_3.5-2.7_8_Hole_Right',
              'Clavicle_Plate_Lateral_7_Hole',
              'Clavicle_Plate_Lateral_9_Hole',
              'Clavicle_Plate_Lateral_10_Hole',
              'Clavicle_Plate_Lateral_11_Hole',
              'Clavicle_Plate_Lateral_12_Hole',  
              'Clavicle_Plate_Medial_6_Hole',
              'Clavicle_Plate_Medial_7_Hole',
              'Clavicle_Plate_Medial_8_Hole',
              'compression-distraction-rod',
              'coupling-orange-green',
              'Crile_Artery_Forceps',
              'Curved_Mayo_Scissors',
              'Depth_Measure',
              'Dorsal_Guide',  
              'Dorsal_Hook_Plate_4_Hole',
              'Dorsal_Hook_Plate_7_Hole',
              'Dressing_Scissors',
              'Drill_Sleeve_2.7_Coaxial',
              'Drill_Sleeve_2.7_Conical',
              'drill-bit-2mm',
              'drill-bit-3.2mm',
              'Fixed_Angle_Plates_3_Hole_7_Peg_Left',  
              'Fixed_Angle_Plates_3_Hole_7_Peg_Right',
              'Fixed_Angle_Plates_5_Hole_6_Peg_Right',
              'Fixed_Angle_Plates_5_Hole_7_Peg_Left',    
              'Gillies_Toothed_Dissector',
              'handle-for-guide-block',
              'hoffman-4-hole-guide-block',
              'Hook_Impactor',
              'Hook_Plate_Volar_4_Hole',  
              'Hook_Plate_Volar_7_Hole',
              'Lahey_Forceps',
              'Large_Depth_Gauge',
              'Large_Quick_Guide',
              'Littlewood_Tissue_Forceps',
              'Mayo_Artery_Forceps',
              'multi-pin-clamp-grey-orange',
              'No3_BP_Handles',
              'No4_BP_Handles',
              'No7_BP_Handles',
              'Olive_K-Wire_1.6mm',
              'Peg_Extender_Quick_Release',
              'Peg_Guides',  
              'periarticular-clamp',
              'pin-to-rod-coupling-grey-orange',
              'rod-to-rod-coupling',
              'rod-to-rod-coupling-orange-orange',
              'Small_Depth_Gauge',
              'Small_Quick_Guide',
              'spanner',
              'Sponge_Forceps',     
              'straight-post',
              'trocar-no3',  
              'trocar-no4',
              'Twist_Drill_1.8mm',
              'Twist_Drill_2.30mm',
              'Volar_Guide',
              'Volar_Impactor',
              'Wedge_Screws',
              'wrench',
            ]

# define our image size (width, height, channels)
target_size = (224, 224)

# load unprocessed images
data_x, data_y = load_imgs(datapath='D:\\592\\resnet\\whole_data_cv\\10\\train\\',
                           classes=classes, target_size=target_size)
print(data_x.shape)
# preprocess images for the model
data_x = preprocess_input(data_x)


# preprocess labels with scikit-learn LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(data_y)
data_y_int = label_encoder.transform(data_y)

# calculate class weight
cls_weight = class_weight.compute_class_weight('balanced', np.unique(data_y_int), data_y_int)
# cls_weight = {i : cls_weight[i] for i in range(len(cls_weight))}
# class_weight = {l:c for l,c in np.unique(data_y_int), class_weight}
# split the data to tran & valid data
data_x_train, data_x_valid, data_y_train, data_y_valid = train_test_split(data_x, data_y_int,
                                                                          test_size=0.2)

# define image data generator for on-the-fly augmentation
# generator = image.ImageDataGenerator(zca_whitening=False, rotation_range=10,
#                                      width_shift_range=0.1, height_shift_range=0.1,
#                                      shear_range=0.02, zoom_range=0.1,
#                                      channel_shift_range=0.05, horizontal_flip=True)

generator = image.ImageDataGenerator()


# fit the generator (required if zca_whitening=True)
generator.fit(data_x)

# load the model without output layer for fine-tuning
# model_baseline = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model_baseline = kap.ResNet101(weights='imagenet', include_top=False, pooling='avg')
features = model_baseline.output
# add output layer
predictions = Dense(len(classes), activation='softmax')(features)


model_tools = Model(inputs=model_baseline.input, outputs=predictions)

# freeze layers
for layer in model_baseline.layers[:154]:
    layer.trainable = False

# compile
opt = Adam(lr=0.0001)
opt_sgd = SGD(lr=5*1e-4, momentum=0.9)
model_tools.compile(optimizer=opt, 
                    loss='sparse_categorical_crossentropy',
                    # loss='categorical_crossentropy',
                    metrics=['accuracy'])
model_tools.summary()

# train
batch_size = 20
early_stop = EarlyStopping(monitor='val_loss', patience=20)
ckpt = ModelCheckpoint(filepath='ckpt_adam_nocrop_norm-{epoch:02d}-{val_loss:.2f}.h5', verbose=1, save_best_only=True, mode='auto')
model_tools.fit(data_x_train, data_y_train, batch_size=20, 
                          epochs=100,
                          steps_per_epoch=int(data_x.shape[0])/batch_size,
                        #   class_weight=cls_weight,
                          validation_data=(data_x_valid, data_y_valid),
                          callbacks=[early_stop, ckpt])
# model.fit(X, X, 
#           steps_per_epoch=train_steps, 
#           validation_data=(X_val, X_val), 
#           validation_steps=val_steps, 
#           epochs=100)
# save the model
model_tools.save('model_freeze_2_adam_clsweight_nocrop_norm.h5')
print('model saved')