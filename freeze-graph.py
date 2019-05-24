import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import tensorflow.contrib.tensorrt as trt
import os

def get_custom_frozen_graph(graph_file):

	#Read Frozen Graph file from disk
	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	for node in graph_def.node:
		if node.op == 'RefSwitch':
			node.op = 'Switch'
			for index in xrange(len(node.input)):
				if 'moving_' in node.input[index]:
					node.input[index] = node.input[index] + '/read'
		elif node.op == 'AssignSub':
			node.op = 'Sub'
			if 'use_locking' in node.attr: del node.attr['use_locking']
		elif node.op == 'AssignAdd':
			node.op = 'Add'
			if 'use_locking' in node.attr: del node.attr['use_locking']
		elif node.op == 'Assign':
			node.op = 'Identity'
			if 'use_locking' in node.attr: del node.attr['use_locking']
			if 'validate_shape' in node.attr: del node.attr['validate_shape']
			if len(node.input) == 2:
				# input0: ref: Should be from a Variable node. May be uninitialized.
				# input1: value: The value to be assigned to the variable.
				node.input[0] = node.input[1]
				del node.input[1]

	return graph_def

def write_graph_to_file(graph_name, graph_def, output_dir):
	"""Write Frozen Graph file to disk."""
	output_path = os.path.join(output_dir, graph_name)
	with tf.gfile.GFile(output_path, "wb") as f:
		f.write(graph_def.SerializeToString())



def get_frozen_graph(graph_file):

	#Read Frozen Graph file from disk
	with tf.gfile.FastGFile(graph_file, "rb") as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())

	return graph_def




def main():

	# make frozenModel.pb
	'''
	# save/restore_all
	freeze_graph.freeze_graph('/host_temp/tf-cpn-test/cpn_graph_batch_1.pbtxt', "", False, 
			'/host_temp/cpn_tensorrt/model_dir/snapshot_350.ckpt', 'tower_0/refine_out/BatchNorm/FusedBatchNorm',
			"save/restore_all", "save/Const",
			'Batch_1_FusedBatch_frozenModel.pb', True, ""  
			)
	'''
	
	'''
	freeze_graph.freeze_graph('/host_temp/test/cpn.pbtxt', "", False, 
			'/host_temp/cpn_tensorrt/model_dir/snapshot_350.ckpt', 'tower_0/refine_out/BatchNorm/AssignMovingAvg,tower_0/refine_out/BatchNorm/AssignMovingAvg_1',
			"save_1/restore_all", "save/Const",
			'refineout_frozenModel.pb', True, ""  
			)
	'''
	
	# tower_0/final_bottleneck/Relu
	# output node : group_deps (import ApplyAdam, AddN, Conv2DBackpropFilter, FusedBatchNormGrad, ReluGrad, ResizeBilinearGrad, Tile


	'''
	saver = tf.train.import_meta_graph('/host_temp/test/snapshot_350.ckpt.meta', clear_devices=True)
	graph = tf.get_default_graph()
	input_graph_def = graph.as_graph_def()
	sess = tf.Session()
	saver.restore(sess, '/host_temp/test/snapshot_350.ckpt')

	#[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

	output_node_names = 'group_deps'
	output_graph_def = tf.graph_util.convert_variables_to_constants(
		sess,
		input_graph_def,
		output_node_names.split(",")
	)

	

	output_graph = '/host_temp/test/frozen_cpn.pb'
	with tf.gfile.GFile(output_graph, "wb") as f :
		f.write(output_graph_def.SerializeToString())
	'''

	'''
	frozen_graph_def = get_frozen_graph('/host_temp/test/FusedBatch_frozenModel.pb')

	value_output_graph = '/host_temp/test/FusedBatch_custom_frozenModel.pb'
	with tf.gfile.GFile(value_output_graph, "wb") as f :
		f.write(frozen_graph_def.SerializeToString())
	'''

	#sess.close()

	#with tf.gfile.FastGFile('/host_temp/test/frozenModel.pb', "rb") as f:
	#	frozen_graph_def = tf.GraphDef()
	#	frozen_graph_def.ParseFromString(f.read())


	

	#custom_frozen_graph_def = get_custom_frozen_graph('/host_temp/test/Batch_1_FusedBatch_frozenModel.pb')

	frozen_graph_def = get_frozen_graph('/host_temp/test/Batch_1_FusedBatch_frozenModel.pb')




	output_nodes = ['tower_0/refine_out/BatchNorm/FusedBatchNorm']
	#output_nodes = ['tower_0/refine_out/Conv2D']
	trt_graph = trt.create_inference_graph(
		frozen_graph_def,
		output_nodes,
		max_batch_size=1,
		max_workspace_size_bytes=(2 << 10) << 20,
		precision_mode='INT8',
		use_calibration=True)

	print('!!!!!! trt graph create !!!!!!')

	
	write_graph_to_file('test_INT8_batch1_trt_graph.pb', trt_graph ,'./')


	
	'''
	# Import the TensorRT graph into a new graph and run:
	output_node = tf.import_graph_def(
		trt_graph,
		return_elements=output_nodes)

	#sess = tf.Session()
	#sess.run(output_node)

	print('!!!!!! new graph run  !!!!!!')
	


	#int8Graph = getINT8InferenceGraph(trt_graph)
	'''


	
	# check how many ops of the original frozen model
	all_nodes = len([1 for n in frozen_graph_def.node])
	print("numb. of all_nodes in frozen graph:", all_nodes)

	# check how many ops that is converted to TensorRT engine
	trt_engine_nodes = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
	print("numb. of trt_engine_nodes in TensorRT graph:", trt_engine_nodes)
	all_nodes = len([1 for n in trt_graph.node])
	print("numb. of all_nodes in TensorRT graph:", all_nodes)
	








if __name__ == '__main__':
    main()
