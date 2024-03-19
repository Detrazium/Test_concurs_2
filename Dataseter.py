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

	def pattern(self, x = None):
		pattern = {0: 'А', 1: 'Б', 2: 'В',
				   3: 'Г', 4: 'Д', 5: 'Е',
				   6: 'Ё', 7: 'Ж', 8: 'З',
				   9: 'И', 10: 'Й', 11: 'К',
				   12: 'Л', 13: 'М', 14: 'Н',
				   15: 'О', 16: 'П', 17: 'Р',
				   18: 'С', 19: 'Т', 20: 'У',
				   21: 'Ф', 22: 'Х', 23: 'Ц',
				   24: 'Ч', 25: 'Ш', 26: 'Щ',
				   27: 'Ъ', 28: 'Ы', 29: 'Ь',
				   30: 'Э', 31: 'Ю', 32: 'Я' }
		return pattern[x]
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
		return Train, validat
	def Get_data(self):
		return self.Train_set

def main():
	Create_dataset().token_key()


if __name__ == '__main__':
	main()