import tensorflow as tf
import numpy as np
import os
from scipy import misc
import glob

def rgba2rgb(img):
	return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def main():	
	output_folder = "./saliency-output"
	input_folder = "test/*"
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)	
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
	with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
		saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
		saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))
		image_batch = tf.get_collection('image_batch')[0]
		pred_mattes = tf.get_collection('mask')[0]
		g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])

		for filename in glob.glob(input_folder):
			slika = misc.imread(filename)
			if slika.shape[2]==4:
				slika = rgba2rgb(slika)
			origin_shape = slika.shape
			slika = np.expand_dims(misc.imresize(slika.astype(np.uint8),[320,320,3],interp="nearest").astype(np.float32)-g_mean,0)

			feed_dict = {image_batch:slika}
			pred_alpha = sess.run(pred_mattes,feed_dict = feed_dict)
			final_alpha = misc.imresize(np.squeeze(pred_alpha),origin_shape)
			misc.imsave(os.path.join(output_folder,filename[5:]),final_alpha)


main()
