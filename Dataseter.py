"""
//////////////////////////

Подзагрузка Датасета

//////////////////////////

"""
import matplotlib.pyplot as plt
import numpy as np

import os, logging
import tensorflow as tf
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import image_dataset_from_directory, to_categorical

class Create_dataset():
	def __init__(self):
		self.Train_set = self.Create_data()
	def token_key(self):
		tokens = to_categorical(range(33), num_classes=34)
		print(tokens)
	def Image_convert(self, image):
		return image

	def train_data_gen(self, key = None):
		img_size =(28, 28)
		class_key = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
					 '10', '11', '12', '13', '14', '15', '16', '17',
					 '18', '19', '20', '21', '22', '23', '24', '25',
					 '26', '27', '28', '29', '30', '31', '32', '33']


		train_gen = image_dataset_from_directory(
			label_mode= 'categorical',
			directory=key,
			validation_split=0.2,
			class_names= class_key,
			subset='training',
			color_mode="grayscale",
			seed = 123,
			image_size= img_size,
			shuffle=True,
			batch_size= 32
		)

		val_data = image_dataset_from_directory(
			directory=key,
			validation_split=0.2,
			class_names=class_key,
			subset = 'validation',
			color_mode='grayscale',
			seed = 123,
			image_size= img_size,
			shuffle = True,
			batch_size= 32,
			label_mode = 'categorical',

		)
		# tf.data.Dataset.save((train_gen, val_data),r'C:\Users\Антонио\PycharmProjects\Test_concurs\DatasetComnist\ComnistD')


		return train_gen, val_data

	def Create_data(self):
		import os
		key = r"C:\Datasets\New_cyrrilic"
		Train, validat = self.train_data_gen(key)
		return Train, validat
	def Get_data(self):
		return self.Train_set

def main():
	Create_dataset().Create_data()


if __name__ == '__main__':
	main()