import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image
from io import BytesIO
from keras.utils import image_dataset_from_directory
import os

def pattern(x = None):
	pattern ={0: 'I', 1: 'Ё', 2: 'А', 3: 'Б', 4: 'В', 5: 'Г', 6: 'Д', 7: 'Е', 8: 'Ж', 9: 'З', 10: 'И', 11: 'Й', 12: 'К', 13: 'Л', 14: 'М', 15: 'Н', 16: 'О', 17: 'П', 18: 'Р', 19: 'С', 20: 'Т', 21: 'У', 22: 'Ф', 23: 'Х', 24: 'Ц', 25: 'Ч', 26: 'Ш', 27: 'Щ', 28: 'Ъ', 29: 'Ы', 30: 'Ь', 31: 'Э', 32: 'Ю', 33: 'Я'}
	return pattern[x]
def file_open(image):
	with open(r'C:\Users\Антонио\PycharmProjects\Test_work_candidate\TRAIN_DATA' + f'\\{image}', 'rb') as file:
		imm = file.read()
		immB = Image.open(BytesIO(imm))
		img = np.array(immB)/255
		return img
def train_data_gen():
	train_gen = image_dataset_from_directory(
		directory=r"C:\Users\Антонио\PycharmProjects\Test_work_candidate\Cyrillic",
		labels='inferred',
		label_mode='int',
		validation_split=0.2,
		subset='training',
		seed = 123,
		image_size= (278, 278),
		shuffle=True,
		batch_size= 32,
	)
	val_data = image_dataset_from_directory(
		directory=r"C:\Users\Антонио\PycharmProjects\Test_work_candidate\Cyrillic",
		labels='inferred',
		label_mode='int',
		validation_split=0.2,
		subset = 'validation',
		seed = 123,
		image_size= (278, 278),
		batch_size= 32
	)
	return train_gen
def Chain_func():
	import tensorflow as tf
	pp = tf.io.gfile.listdir(path = r'C:\Users\Антонио\PycharmProjects\Test_concurs\Cyrillic')
	print(pp)
	# data = train_data_gen()
def main():
	Chain_func()


if __name__ == '__main__':
	main()