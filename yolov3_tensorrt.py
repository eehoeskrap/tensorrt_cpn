# Import the needed libraries
import cv2
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.platform import gfile
from PIL import Image
from YOLOv3 import utils
'''
# function to read a ".pb" model 
# (can be used to read frozen model or TensorRT model)
def read_pb_graph(model):
	with gfile.FastGFile(model,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	return graph_def

frozen_graph = read_pb_graph("/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/yolov3_gpu_nms.pb")

your_outputs = ["Placeholder:0", "concat_9:0", "mul_9:0"]
# convert (optimize) frozen model to TensorRT model
trt_graph = trt.create_inference_graph(
	input_graph_def=frozen_graph,# frozen model
	outputs=your_outputs,
	max_batch_size=1,# specify your max batch size
	max_workspace_size_bytes=2*(10**9),# specify the max workspace
	precision_mode="FP16") # precision, can be "FP32" (32 floating point precision) or "FP16"

#write the TensorRT model to be used later for inference
with gfile.FastGFile("/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/TensorRT_YOLOv3_2.pb", 'wb') as f:
	f.write(trt_graph.SerializeToString())
print("TensorRT model is successfully stored!")

# check how many ops of the original frozen model
all_nodes = len([1 for n in frozen_graph.node])
print("numb. of all_nodes in frozen graph:", all_nodes)

# check how many ops that is converted to TensorRT engine
trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
all_nodes = len([1 for n in trt_graph.node])
print("numb. of all_nodes in TensorRT graph:", all_nodes)

'''


'''

2019-05-21 08:28:34.407568: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:581] Optimization results for grappler item: tf_graph
2019-05-21 08:28:34.407609: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:583]   constant folding: Graph size after: 880 nodes (-7136), 993 edges (-7985), time = 3814.25806ms.
2019-05-21 08:28:34.407616: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:583]   layout: Graph size after: 897 nodes (17), 1009 edges (16), time = 93.946ms.
2019-05-21 08:28:34.407623: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:583]   constant folding: Graph size after: 890 nodes (-7), 1009 edges (0), time = 460.098ms.
2019-05-21 08:28:34.407628: I tensorflow/core/grappler/optimizers/meta_optimizer.cc:583]   TensorRTOptimizer: Graph size after: 73 nodes (-817), 100 edges (-909), time = 54916.0703ms.
TensorRT model is successfully stored!
numb. of all_nodes in frozen graph: 8016
numb. of trt_engine_nodes in TensorRT graph: 18
numb. of all_nodes in TensorRT graph: 73



'''



# config
SIZE = [416, 416] #input image dimension
# video_path = 0 # if you use camera as input
video_path = "/host_temp/tf-cpn-test/HM_HandWave_025.mp4" # path for video input
classes = utils.read_coco_names('/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/YOLOv3/coco.names')
num_classes = len(classes)
GIVEN_ORIGINAL_YOLOv3_MODEL = "/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/TensorRT_YOLOv3_2.pb" # to use given original YOLOv3
TENSORRT_YOLOv3_MODEL = "/host_temp/tf-cpn-test/models/COCO.res50.256x192.CPN/TensorRT_YOLOv3_2.pb" # to use the TensorRT optimized model


# get input-output tensor
input_tensor, output_tensors = \
utils.read_pb_return_tensors(tf.get_default_graph(),
                             TENSORRT_YOLOv3_MODEL,
                             ["Placeholder:0", "concat_9:0", "mul_9:0"])


# perform inference


with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4))) as sess:

	vid = cv2.VideoCapture(video_path) # must use opencv >= 3.3.1 (install it by 'pip install opencv-python')
	while True:
		return_value, frame = vid.read()
		if return_value == False:
			print('ret:', return_value)
			vid = cv2.VideoCapture(video_path)
			return_value, frame = vid.read()
		if return_value:
			image = Image.fromarray(frame)
		else:
			raise ValueError("No image!")


            
		#img_resized = np.array(image, dtype = np.float32)
		img_resized = np.array(image.resize(size=tuple(SIZE)), 
                               dtype=np.float32)
		img_resized = img_resized / 255.
		prev_time = time.time()


		boxes, scores = sess.run(output_tensors, 
					feed_dict={input_tensor: 
					np.expand_dims(
					img_resized, axis=0)})


		boxes, scores, labels = utils.cpu_nms(boxes, 
					scores, 
					num_classes, 
					score_thresh=0.4, 
					iou_thresh=0.5)
		image = utils.draw_boxes(image, boxes, scores, labels, 
					classes, SIZE, show=False)

		curr_time = time.time()
		exec_time = curr_time - prev_time
		result = np.asarray(image)
		info = "time:" + str(round(1000*exec_time, 2)) + " ms, FPS: " + str(round((1000/(1000*exec_time)),1))


		cv2.putText(result, text=info, org=(50, 70), 
			fontFace=cv2.FONT_HERSHEY_SIMPLEX,
			fontScale=1, color=(255, 0, 0), thickness=2)
		#cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
		cv2.imshow("result", result)
		if cv2.waitKey(10) & 0xFF == ord('q'): break


