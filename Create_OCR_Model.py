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
			Conv2D(64, (3, 3), padding = 'same', activation='relu'),
			MaxPooling2D((2, 2), strides=2),
			Conv2D(64, (3, 3), padding='same', activation='relu'),
			Conv2D(64, (3, 3), padding='same', activation='relu'),
			Flatten(),
			Dense(132, activation='relu'),
			Dense(33, activation='softmax')
		])
		Model.summary()
		Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return Model
	def Train_model(self):
		history = self.model.fit(self.Train, validation_data=self.Val, epochs=10, batch_size=32, shuffle = True)

		print(history.history.keys())
		plt.plot(history.history['accuracy'], label = 'acc train')
		plt.plot(history.history['loss'], label = 'loss train')
		plt.plot(history.history['val_accuracy'], label = 'acc test')
		plt.plot(history.history['val_loss'], label='loss test')
		plt.xlabel('epochs')
		plt.ylabel('accuracy')
		plt.legend()
		plt.show()


		self.model.save(r'OCR_models\test_model_ocr_recovVV4.h5')
		return history

def main():
	OCR_Model()
if __name__ == '__main__':
	main()