"""
|||||||||||||||||||||
Делит файлы на валидацию и тестовую выборки
|||||||||||||||||||||
"""
import keras.utils


class Validat_split():
	def __init__(self, Dataset):
		self.Data = Dataset
	def splitD(self):
		X, Y = self.Data
		X = X / 255
		Y = Y / 133
		# Y = keras.utils.to_categorical(Y, num_classes=134)
		Trip = (len(X) * 20) // 100
		X_train = X[Trip:]
		Y_train = Y[Trip:]

		X_test = X[:Trip]
		Y_test = Y[:Trip]

		print("X_train: ",X_train.shape)
		print("Y_train: ",Y_train.shape)
		print('val X: ', X_test.shape)
		print("val Y: ",Y_test.shape)


		return (X_train, Y_train), (X_test, Y_test)
