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
from keras import Sequential
from keras.layers import (
Conv2D,
MaxPooling2D,
Dense,
Flatten,

)

class OCR_Model():
	def __init__(self, train, val):
		self.model = self.Create_model()
		self.Train = train
		self.Val = val

	def Train_clip(self):
		return

	def Create_model(self):
		Model = Sequential([
			Conv2D(128, (3, 3), padding='same', input_shape=(150, 220, 1), activation='relu'),
			Conv2D(64, (3, 3), padding='same', activation='relu'),
			MaxPooling2D((2, 2), strides=2),
			Conv2D(100, (3, 3), padding='same', activation='relu'),
			MaxPooling2D((2, 2), strides=2),
			Conv2D(134, (3, 3), padding='same', activation='relu'),
			Flatten(),
			Dense(134, activation='relu'),
			Dense(300, activation='sigmoid'),
		])
		Model.summary()
		Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		return Model
	def Train_model(self):
		history = self.model.fit(x = self.Train[0], y= self.Train[1], validation_data = (self.Val[0], self.Val[1]), epochs=1, batch_size=300, shuffle = True)
		plt.imshow([history])
		plt.show()
		return history



def main():
	OCR_Model()

if __name__ == '__main__':
	main()