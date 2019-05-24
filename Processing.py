import os
import os.path as osp
import numpy as np
import sys
import matplotlib.pyplot as plt
import random
import cv2
import argparse
import math

import tensorflow as tf

# for tensorflow_cpn
from config import cfg
from dataset import Preprocessing

# for keras_retinanet
from keras_retinanet.utils.image import preprocess_image, resize_image

def draw_bounding_box(frame, person_dets):

	x, y, w, h = person_dets

	top = max(0, np.floor(x + 0.5).astype(int))
	left = max(0, np.floor(y + 0.5).astype(int))
	right = min(frame.shape[1], np.floor(x + w + 0.5).astype(int))
	bottom = min(frame.shape[0], np.floor(y + h + 0.5).astype(int))

	cv2.rectangle(frame, (top, left), (right, bottom), (255, 0, 0), 2)


def read_pb_return_tensors(graph, pb_file, return_elements):

	with tf.gfile.FastGFile(pb_file, 'rb') as f:
		frozen_graph_def = tf.GraphDef()
		frozen_graph_def.ParseFromString(f.read())

	with graph.as_default():
		return_elements = tf.import_graph_def(frozen_graph_def,
						return_elements=return_elements)
	return return_elements


def crop(pose_img, person_dets):

	#  cls_dets : x1, y1, x2, y2, score
	cls_dets = np.zeros((1, 4), dtype=np.float32)
	# test_data : x, y, w, h, score
	test_data = np.zeros((1, 4), dtype=np.float32)

	test_data[:] = person_dets[:]

	bbox = np.asarray(test_data[0])
	cls_dets[0, :4] = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])

	test_imgs = []
	details = []

	# cropping
	test_img, detail = Preprocessing(pose_img, test_data[0], stage='test')

	details.append(detail)

	details = np.asarray(details).astype(np.float32)

	feed = test_img

	data = [feed.transpose(0, 2, 3, 1).astype(np.float32)]

	return data, details

def keypoint_detection(res, details):

	flat = [0.0 for i in range(cfg.nr_skeleton * 2)]
	cls_skeleton = np.zeros((1, cfg.nr_skeleton, 3)).astype(np.float32)
	crops = np.zeros((1, 4)).astype(np.float32)

	res = res.transpose(0, 3, 1, 2)

	# single map
	r0 = res[0].copy()
	r0 /= 255.
	r0 += 0.5

	for w in range(cfg.nr_skeleton):
		res[0, w] /= np.amax(res[0, w])
	border = 10
	dr = np.zeros((cfg.nr_skeleton, cfg.output_shape[0] + 2 * border, cfg.output_shape[1] + 2 * border))
	dr[:, border:-border, border:-border] = res[:cfg.nr_skeleton].copy()

	for w in range(cfg.nr_skeleton):
		dr[w] = cv2.GaussianBlur(dr[w], (21, 21), 0)
	for w in range(cfg.nr_skeleton):
		lb = dr[w].argmax()
		y, x = np.unravel_index(lb, dr[w].shape)
		dr[w, y, x] = 0
		lb = dr[w].argmax()
		py, px = np.unravel_index(lb, dr[w].shape)
		y -= border
		x -= border
		py -= border + y
		px -= border + x
		ln = (px ** 2 + py ** 2) ** 0.5
		delta = 0.25
		if ln > 1e-3:
			x += delta * px / ln
			y += delta * py / ln
		x = max(0, min(x, cfg.output_shape[1] - 1))
		y = max(0, min(y, cfg.output_shape[0] - 1))
		cls_skeleton[0, w, :2] = (x * 4 + 2, y * 4 + 2)
		cls_skeleton[0, w, 2] = r0[w, int(round(y) + 1e-10), int(round(x) + 1e-10)]

	# map back to original images
	crops[0, :] = details[0, :]
	for w in range(cfg.nr_skeleton):
		cls_skeleton[0, w, 0] = cls_skeleton[0, w, 0] / cfg.data_shape[1] * (
					crops[0][2] - crops[0][0]) + crops[0][0]
		cls_skeleton[0, w, 1] = cls_skeleton[0, w, 1] / cfg.data_shape[0] * (
					crops[0][3] - crops[0][1]) + crops[0][1]

	# flat is keypoints(17)
	for w in range(cfg.nr_skeleton):
		flat[w*2] = cls_skeleton[0, w, 0]
		flat[w*2+1] = cls_skeleton[0, w, 1]

	return flat

def upper_detection(frame, flat, person_dets):

	# upper detection & lower keypoint remove
	upper = False

	"""
	lower keypoint remove using keypoint of hip, knee

	l_hip_y , r_hip_y  : flat[23], flat[25] or cls_skeleton[0, 11, 1], cls_skeleton[0, 12, 1]
	l_knee_y, r_knee_y : flat[27], flat[29] or cls_skeleton[0, 13, 1], cls_skeleton[0, 14, 1]

	"""

	l_hip_y = flat[23]
	r_hip_y = flat[25]
	
	l_knee_y = flat[27]
	r_knee_y = flat[29]

	bbox_y = person_dets[1] + person_dets[3]


	# remove based on hip keypoint
	hip_distance_r = r_hip_y - bbox_y
	hip_distance_l = l_hip_y - bbox_y

	# remove based on knee keypoint 
	knee_distance_r = r_knee_y - bbox_y
	knee_distance_l = l_knee_y - bbox_y

	# remove based on bounding box (frame.shape[0] = 720)
	box_distance = bbox_y - frame.shape[0]

	
	hip_distance_r = abs(hip_distance_r)
	hip_distance_l = abs(hip_distance_l)
	knee_distance_r = abs(knee_distance_r)
	knee_distance_l = abs(knee_distance_l)
	box_distance = abs(box_distance)


	if ((hip_distance_r < 110 and hip_distance_l < 110 and box_distance < 30) or (knee_distance_r < 50 and knee_distance_l < 50 and box_distance < 30)):
		upper = True
		
		# remove lower (knee, ankle) keypoint
		for i in range(26, 34):
			flat[i] = 0.0

	return flat, upper


def draw_skeleton(aa, kp, upper=False):

	#upper = False

	show_skeleton_labels = False

	kp = np.array(kp).astype(int)
	kp = kp.reshape(17, 2)

	kp_names = ['nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear', 'l_shoulder', 
			'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 
			'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

	skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

	# remove lower (knee, ankle) 
	if upper :
		skeleton = skeleton[4:]	
		kp_names = kp_names[:13]
		kp = kp[:13]

	for i, j in skeleton:
		if kp[i-1][0] >= 0 and kp[i-1][1] >= 0 and kp[j-1][0] >= 0 and kp[j-1][1] >= 0 and \
			(len(kp[i-1]) <= 2 or (len(kp[i-1]) > 2 and  kp[i-1][2] > 0.1 and kp[j-1][2] > 0.1)):
			cv2.line(aa, tuple(kp[i-1][:2]), tuple(kp[j-1][:2]), (0,255,255), 2)
	for j in range(len(kp)):
		if kp[j][0] >= 0 and kp[j][1] >= 0:

			if len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 1.1):
				cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((0,0,255)), 2)
			elif len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1):
				cv2.circle(aa, tuple(kp[j][:2]), 2, tuple((255,0,0)), 2)

			if show_skeleton_labels and (len(kp[j]) <= 2 or (len(kp[j]) > 2 and kp[j][2] > 0.1)):
				cv2.putText(aa, kp_names[j], tuple(kp[j][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0))



