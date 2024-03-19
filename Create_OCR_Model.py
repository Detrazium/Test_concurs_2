"""
//////////////////////////

Создание модели

//////////////////////////
"""
import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import keras
from keras import Sequential
from keras.layers import (
Conv2D,
MaxPooling2D,
Dense,
Flatten,
Reshape,
LSTM,
GRU,
Input
)

class OCR_Model():
	def __init__(self, Train_, Val_):
		self.model = self.Create_model()
		self.Train = Train_
		self.Val = Val_

	def Create_model(self):
		Model = Sequential([
			Dense(10)
		])
		Model.summary()
		Model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
		return Model
	def Train_model(self):
		history = self.model.fit(self.Train, validation_data=self.Val, epochs=5, batch_size=32)
		return history



def main():
	OCR_Model()

if __name__ == '__main__':
	main()