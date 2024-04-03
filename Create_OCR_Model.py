"""
//////////////////////////

Создание модели

//////////////////////////
"""
import os, logging
import matplotlib.pyplot as plt
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import (
Conv2D,
MaxPooling2D,
Dense,
Flatten,

)

class OCR_Model():
	def __init__(self, train = None, val=None):
		self.model = self.Create_model()
		self.Train = train
		self.Val = val

	def Train_clip(self):
		return

	def Create_model(self):
		Model = Sequential([
			Conv2D(32, (3, 3), padding = 'same', input_shape=(28, 28, 1), activation='relu'),
			MaxPooling2D((2, 2), strides=2),
			Conv2D(64, (3, 3), padding='same', activation='relu'),
			Flatten(),
			Dense(128, activation='relu'),
			Dense(33, activation='softmax')
		])
		Model.summary()
		Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return Model
	def Train_model(self):
		history = self.model.fit(self.Train, validation_data=self.Val, epochs=8, batch_size=32, shuffle = True)
		print(history.history.keys())
		plt.plot(history.history['accuracy'], label = 'acc train')
		plt.plot(history.history['val_accuracy'], label = 'acc test')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		plt.legend()
		plt.show()


		self.model.save('test_model_ocr_recovV1.h5')
		return history

def test_model():
	model = keras.models.load_model('test_model_ocr.h5')

def main():
	test_model()
	# OCR_Model()
if __name__ == '__main__':
	main()