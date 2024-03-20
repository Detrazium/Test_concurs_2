"""
//////////////////////////

Подзагрузка Датасета

//////////////////////////

"""


import os, logging
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
		train_gen = image_dataset_from_directory(
			directory=key,
			validation_split=0.2,
			subset='training',
			seed = 123,
			image_size= (278, 278),
			shuffle=True,
			batch_size= 32,
		)
		val_data = image_dataset_from_directory(
			directory=key,
			validation_split=0.2,
			subset = 'validation',
			seed = 123,
			image_size= (278, 278),
			shuffle = True,
			batch_size= 32
		)

		return train_gen, val_data

	def Create_data(self):
		key = r"C:\Datasets\Cyrillic"
		Train, validat = self.train_data_gen(key)
		for i in Train:
			print(i[0].shape)
		return Train, validat
	def Get_data(self):
		return self.Train_set

def main():
	Create_dataset().Create_data()


if __name__ == '__main__':
	main()