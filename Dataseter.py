"""
//////////////////////////

Подзагрузка Датасета

//////////////////////////

"""

import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.utils import image_dataset_from_directory

class Create_dataset():
	def __init__(self):
		self.Train_set = self.Create_data()

	def train_data_gen(self, key = None):
		img_size =(28, 28)
		class_key = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
					 '10', '11', '12', '13', '14', '15', '16', '17',
					 '18', '19', '20', '21', '22', '23', '24', '25',
					 '26', '27', '28', '29', '30', '31', '32', '33']


		train_gen = image_dataset_from_directory(
			directory=key,
			label_mode= 'categorical',
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
			label_mode='categorical',
			validation_split=0.2,
			class_names=class_key,
			subset = 'validation',
			color_mode='grayscale',
			seed = 123,
			image_size= img_size,
			shuffle = True,
			batch_size= 32,

		)
		return train_gen, val_data

	def Create_data(self):
		key = r"C:\Datasets\New_cyrrilic"
		Train, validat = self.train_data_gen(key)
		return Train, validat
	def Get_data(self):
		return self.Train_set

def main():
	Data = Create_dataset().Create_data()


if __name__ == '__main__':
	main()