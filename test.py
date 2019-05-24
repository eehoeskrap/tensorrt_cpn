import os
import os.path as osp
import numpy as np
import argparse
from config import cfg
import cv2
import sys
import time

# for tensorflow_cpn
from tfflat.base import Tester
from network import Network

# for keras_retinanet
#from keras_retinanet import models

# for yolov3_keras
from object_detection.yolo_model import YOLO
from object_detection.yolo_demo import *

# for GRN process
from Processing import *

import tensorflow.contrib.tensorrt as trt

from YOLOv3 import utils
from PIL import Image

from yolov3 import *

def read_pb_return_tensors(graph, pb_file, return_elements):

	with tf.gfile.FastGFile(pb_file, 'rb') as f:
		frozen_graph_def = tf.GraphDef()
		frozen_graph_def.ParseFromString(f.read())

	with graph.as_default():
		return_elements = tf.import_graph_def(frozen_graph_def,
						return_elements=return_elements)
	return return_elements


def getINT8InferenceGraph():

	with tf.gfile.FastGFile('/host_temp/test/test_INT8_batch1_trt_graph.pb', 'rb') as f:
		calibGraph = tf.GraphDef()
		calibGraph.ParseFromString(f.read())

	trt_graph=trt.calib_graph_to_infer_graph(calibGraph)

	with gfile.FastGFile("CPN_TRTINT8.pb",'wb') as f:
		f.write(trt_graph.SerializeToString())


	return trt_graph

'''
def get_frozen_graph(graph_file):
  """Read Frozen Graph file from disk."""
  with tf.gfile.FastGFile(graph_file, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
  return graph_def
'''
if __name__ == '__main__':


	# for TensorRT

	'''
	numb. of all_nodes in frozen graph: 1133
	numb. of trt_engine_nodes in TensorRT graph: 4
	numb. of all_nodes in TensorRT graph: 80

	TensorRT   : average inference time:  0.004047312162966975
	TensorFlow : average inference time:  0.00886301307745206
	'''


	# tf
	TENSORFLOW_MODEL = '/host_temp/test/Batch_1_FusedBatch_frozenModel.pb'

	# trt FP32
	TENSORRT_FP32_MODEL = '/host_temp/test/test_FP32_batch1_trt_graph.pb'

	# trt FP16
	TENSORRT_FP16_MODEL = '/host_temp/test/test_FP16_batch1_trt_graph.pb'

	# trt INT8
	TENSORRT_INT8_MODEL = '/host_temp/test/test_INT8_batch1_trt_graph.pb'

	input_tensor, output_tensors = \
	read_pb_return_tensors(tf.get_default_graph(),
			TENSORRT_INT8_MODEL,
			["tower_0/Placeholder:0", "tower_0/refine_out/BatchNorm/FusedBatchNorm:0"])


	getINT8InferenceGraph()

	print('done')


	fps_time = 0 
	total_time = 0
	crop_total_time = 0
	kd_total_time = 0
	total_total_time = 0
	detect_total_time = 0

	# HM_HandWave_025.mp4
	# HM_ArmFolding_003
	# HM_HandWave_018.mp4
	# HM_ArmFolding_018.mp4
	video_path = '/host_temp/tf-cpn-test/HM_HandWave_025.mp4'
	cap = cv2.VideoCapture(video_path)
	vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))




	GIVEN_ORIGINAL_YOLOv3_MODEL = "/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/TensorRT_YOLOv3_2.pb" # to use given original YOLOv3
	TENSORRT_YOLOv3_MODEL = "/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/TensorRT_YOLOv3_2.pb" # to use the TensorRT optimized model

	'''
	# get input-output tensor
	input_tensor, output_tensors = \
	utils.read_pb_return_tensors(tf.get_default_graph(),
		                     TENSORRT_YOLOv3_MODEL,
		                     ["Placeholder:0", "concat_9:0", "mul_9:0"])
	'''

	#tfconfig = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
	#with tf.Session(config=tfconfig) as sess:
	with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4))) as sess:





		# person detection model 
		yolo = YOLO(0.6, 0.5)
		file = '/host_temp/tf-cpn-test/data/coco_classes.txt'
		all_classes = get_classes(file)





		start = time.time()

		for vid_file in range(vid_length): 

			total_t1 = time.time()

			ret, frame = cap.read()

			# frame, video length difference
			if frame is None:
				print('Frame Done.')
				break



			# get box detection (yolo v3)
			#detection_t1 = time.time()
			person_dets = detect_person(frame, yolo, all_classes)
			#detection_t2 = time.time()
			#detection_delta_time = detection_t2 - detection_t1
			#detect_total_time += detection_delta_time
			#print("### object detection inference : ", detection_delta_time)


			#person_dets, image = person_detection(sess, frame, input_tensor, output_tensors)





			# for person empty
			if (len(person_dets) != 0):
			
				kd_t1 = time.time()
				# get pose estimation
				data, details = crop(frame, person_dets)

				
				t1 = time.time()
				res = sess.run(output_tensors, feed_dict={input_tensor: data[0]})
				t2 = time.time()
				delta_time = t2 - t1
				total_time += delta_time
				print("pose inference : ", delta_time)


				flat = keypoint_detection(res, details)

				# lower keypoint remove
				flat, upper = upper_detection(frame, flat, person_dets)

				# draw result
				draw_skeleton(frame, flat, upper)
				
				draw_bounding_box(frame, person_dets)
				cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - fps_time)), (20, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

				kd_t2 = time.time()
				kd_delta_time = kd_t2 - kd_t1 
				kd_total_time += kd_delta_time
				print("Keypoint Detection needed time in inference : ", kd_delta_time)

			fps_time = time.time()		

			total_t2 = time.time()
			total_delta_time = total_t2 - total_t1
			total_total_time += total_delta_time
			print("total inference time : ", total_delta_time)
			print("================================")
			print("")

			cv2.imshow('pose estimation', frame)



			#cv2.waitKey(0)
			#if cv2.waitKey(1) == 27:
			#	break

			if cv2.waitKey(1) & 0xFF == ord('q'): break


		avg_time_original_model = total_total_time / vid_length
		print("------------------ total average inference time: ", avg_time_original_model)

		avg_time_original_model2 = total_time / vid_length
		print("------------------ pose average inference time: ", avg_time_original_model2)

		avg_time_original_model3 = detect_total_time / vid_length
		print("------------------ detect average inference time: ", avg_time_original_model3)










'''

# TENSORFLOW_MODEL

### object detection inference :  0.03390383720397949
pose inference :  0.008146524429321289
Keypoint Detection needed time in inference :  0.013665199279785156
total inference time :  0.04883313179016113
================================

### object detection inference :  0.034833431243896484
pose inference :  0.008049488067626953
Keypoint Detection needed time in inference :  0.01354074478149414
total inference time :  0.049635887145996094
================================

### object detection inference :  0.03442811965942383
pose inference :  0.008107423782348633
Keypoint Detection needed time in inference :  0.013631582260131836
total inference time :  0.04931187629699707
================================

Frame Done.
------------------ total average inference time:  0.05112406475502148
------------------ pose average inference time:  0.0087516562980518
------------------ detect average inference time:  0.03565225538454558


# TENSORRT_FP16_MODEL

### object detection inference :  0.03417396545410156
pose inference :  0.003663778305053711
Keypoint Detection needed time in inference :  0.00896906852722168
total inference time :  0.044625282287597656
================================

### object detection inference :  0.03377985954284668
pose inference :  0.003682851791381836
Keypoint Detection needed time in inference :  0.00890040397644043
total inference time :  0.043932437896728516
================================

### object detection inference :  0.03356480598449707
pose inference :  0.003613710403442383
Keypoint Detection needed time in inference :  0.0088958740234375
total inference time :  0.04368472099304199
================================

Frame Done.
------------------ total average inference time:  0.04604456236487941
------------------ pose average inference time:  0.003897848045616819
------------------ detect average inference time:  0.03574482746291579



# TENSORRT_FP32_MODEL

### object detection inference :  0.03487515449523926
pose inference :  0.005206584930419922
Keypoint Detection needed time in inference :  0.010619163513183594
total inference time :  0.046727895736694336
================================

### object detection inference :  0.03437948226928711
pose inference :  0.005312442779541016
Keypoint Detection needed time in inference :  0.01089334487915039
total inference time :  0.04651904106140137
================================

### object detection inference :  0.03439164161682129
pose inference :  0.005234718322753906
Keypoint Detection needed time in inference :  0.010639667510986328
total inference time :  0.04624485969543457
================================

Frame Done.
------------------ total average inference time:  0.048068236886409293
------------------ pose average inference time:  0.00550817188463713
------------------ detect average inference time:  0.03587247865241871



'''


