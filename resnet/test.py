# from tensorflow import keras
# from keras.models import load_model
# from utils import load_imgs
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score
# # from keras.applications.resnet50 import preprocess_input
# from keras.applications.resnet import preprocess_input
# # from keras.applications.resnet import ResNet101
# # from ResNet101 import preprocess_input
# import numpy as np
# import tensorflow as tf
# import os
# # import keras.applications as kap

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# # vram limit for eficiency
# # config = tf.ConfigProto()
# # config.gpu_options.allow_growth = True
# # sess = tf.Session(config=config)
# config=tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# # sess = tf.Session(config=config)
# sess=tf.compat.v1.Session(config=config)
# print("strat validation")
# ############# same as training phase ##############
# # define class names
# # classes = ['6_Babcock_Tissue_Forceps',
# #            '6_Mayo_Needle_Holder',
# #             '7_Metzenbaum_Scissors',
# #              '7_Microvascular_Needle_Holder',
# #               '8_Babcock_Tissue_Forceps' ,
# #               '8_Mayo_Needle_Holder' ,
# #               '8_Microvascular_Needle_Holder' ,
# #               '9_DeBakey_Dissector' ,
# #               '9_DeBakey_Needle_Holder' ,
# #               '9_Metzenbaum_Scissors' ,
# #               'Allis_Tissue_Forceps' ,
# #               'Ball___Socket_Towel_Clips' ,
# #               'Bonneys_Non_Toothed_Dissector' ,
# #               'Bonneys_Toothed_Dissector',
# #               'Crile_Artery_Forceps',
# #               'Curved_Mayo_Scissors',
# #               'Dressing_Scissors',
# #               'Gillies_Toothed_Dissector',
# #               'Lahey_Forceps',
# #               'Littlewood_Tissue_Forceps',
# #               'Mayo_Artery_Forceps',
# #               'No3_BP_Handles',
# #               'No4_BP_Handles',
# #               'No7_BP_Handles',
# #               'Sponge_Forceps',]
# classes = ['1.1mm_K-Wire',
#           '6_Babcock_Tissue_Forceps',
#            '6_Mayo_Needle_Holder',
#             '7_Metzenbaum_Scissors',
#              '7_Microvascular_Needle_Holder',
#               '8_Babcock_Tissue_Forceps' ,
#               '8_Mayo_Needle_Holder' ,
#               '8_Microvascular_Needle_Holder' ,
#               '9_DeBakey_Dissector' ,
#               '9_DeBakey_Needle_Holder',
#               '9_Metzenbaum_Scissors' ,
#               'Allis_Tissue_Forceps' ,
#               'angled-post',
#               'Ball___Socket_Towel_Clips' ,
#               'Bearing_Plates_3_Hole_7_Peg_Left',
#               'Bearing_Plates_3_Hole_7_Peg_Left_Narrow',
#               'Bearing_Plates_3_Hole_7_Peg_Right',
#               'Bearing_Plates_3_Hole_7_Peg_Right_Narrow',
#               'Bearing_Plates_5_Hole_5_Peg_Left',
#               'Bearing_Plates_5_Hole_5_Peg_Right',
#               'Bearing_Plates_5_Hole_7_Peg_Left',  
#               'Bearing_Plates_5_Hole_7_Peg_Right',
#               'Bearing_Plates_7_Hole_7_Peg_Left',
#               'Bearing_Plates_7_Hole_7_Peg_Right',
#               'Bonneys_Non_Toothed_Dissector' ,
#               'Bonneys_Toothed_Dissector',
#               'chuck-no3',
#               'chuck-no4',
#               'Clavicle_Plate_3.5_6_Hole_Left',
#               'Clavicle_Plate_3.5_6_Hole_Right',  
#               'Clavicle_Plate_3.5_7_Hole_Left',
#               'Clavicle_Plate_3.5_7_Hole_Right',
#               'Clavicle_Plate_3.5_8_Hole_Left',
#               'Clavicle_Plate_3.5_8_Hole_Right',
#               'Clavicle_Plate_3.5-2.7_3_Hole_Left',
#               'Clavicle_Plate_3.5-2.7_3_Hole_Right',
#               'Clavicle_Plate_3.5-2.7_4_Hole_Left',  
#               'Clavicle_Plate_3.5-2.7_4_Hole_Right',
#               'Clavicle_Plate_3.5-2.7_5_Hole_Left',
#               'Clavicle_Plate_3.5-2.7_5_Hole_Right',
#               'Clavicle_Plate_3.5-2.7_6_Hole_Left',
#               'Clavicle_Plate_3.5-2.7_6_Hole_Right',
#               'Clavicle_Plate_3.5-2.7_7_Hole_Left',
#               'Clavicle_Plate_3.5-2.7_7_Hole_Right',  
#               'Clavicle_Plate_3.5-2.7_8_Hole_Left',
#               'Clavicle_Plate_3.5-2.7_8_Hole_Right',
#               'Clavicle_Plate_Lateral_7_Hole',
#               'Clavicle_Plate_Lateral_9_Hole',
#               'Clavicle_Plate_Lateral_10_Hole',
#               'Clavicle_Plate_Lateral_11_Hole',
#               'Clavicle_Plate_Lateral_12_Hole',  
#               'Clavicle_Plate_Medial_6_Hole',
#               'Clavicle_Plate_Medial_7_Hole',
#               'Clavicle_Plate_Medial_8_Hole',
#               'compression-distraction-rod',
#               'coupling-orange-green',
#               'Crile_Artery_Forceps',
#               'Curved_Mayo_Scissors',
#               'Depth_Measure',
#               'Dorsal_Guide',  
#               'Dorsal_Hook_Plate_4_Hole',
#               'Dorsal_Hook_Plate_7_Hole',
#               'Dressing_Scissors',
#               'Drill_Sleeve_2.7_Coaxial',
#               'Drill_Sleeve_2.7_Conical',
#               'drill-bit-2mm',
#               'drill-bit-3.2mm',
#               'Fixed_Angle_Plates_3_Hole_7_Peg_Left',  
#               'Fixed_Angle_Plates_3_Hole_7_Peg_Right',
#               'Fixed_Angle_Plates_5_Hole_6_Peg_Right',
#               'Fixed_Angle_Plates_5_Hole_7_Peg_Left',    
#               'Gillies_Toothed_Dissector',
#               'handle-for-guide-block',
#               'hoffman-4-hole-guide-block',
#               'Hook_Impactor',
#               'Hook_Plate_Volar_4_Hole',  
#               'Hook_Plate_Volar_7_Hole',
#               'Lahey_Forceps',
#               'Large_Depth_Gauge',
#               'Large_Quick_Guide',
#               'Littlewood_Tissue_Forceps',
#               'Mayo_Artery_Forceps',
#               'multi-pin-clamp-grey-orange',
#               'No3_BP_Handles',
#               'No4_BP_Handles',
#               'No7_BP_Handles',
#               'Olive_K-Wire_1.6mm',
#               'Peg_Extender_Quick_Release',
#               'Peg_Guides',  
#               'periarticular-clamp',
#               'pin-to-rod-coupling-grey-orange',
#               'rod-to-rod-coupling',
#               'rod-to-rod-coupling-orange-orange',
#               'Small_Depth_Gauge',
#               'Small_Quick_Guide',
#               'spanner',
#               'Sponge_Forceps',     
#               'straight-post',
#               'trocar-no3',  
#               'trocar-no4',
#               'Twist_Drill_1.8mm',
#               'Twist_Drill_2.30mm',
#               'Volar_Guide',
#               'Volar_Impactor',
#               'Wedge_Screws',
#               'wrench',
#             ]
# target_size = (224, 224)
# ###################################################

# # load the test data
# data_x, data_y = load_imgs(datapath='D:\\592\\resnet\\whole_data_cv\\10\\test\\', classes=classes, target_size=target_size)

# # preprocess images for the model
# # data_x = preprocess_input(data_x)
# data_x = preprocess_input(data_x)
# label_encoder = LabelEncoder()
# label_encoder.fit(data_y)
# data_y_int = label_encoder.transform(data_y)


# # load the model
# print("load model")
# model_tools = load_model('ckpt_adam_nocrop_norm-36-0.06.h5')

# # predict with the model
# preds = model_tools.predict(data_x, batch_size=12, verbose=1)
# preds = np.argmax(preds, axis=1)
# print(data_y_int)
# print(preds)
# print('')
# print(accuracy_score(y_true=data_y_int, y_pred=preds))
# # print(preds)








