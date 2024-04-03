collection = r"C:\Datasets\Cyrillic"

import os
import cv2
from PIL import Image
from skimage import transform
from skimage.transform import AffineTransform, warp, rotate
import numpy as np
class Generate_dataset():
	def __init__(self):
		self.Cyr = r"C:\Datasets\Cyrillic"
		self.Ncyr = r"C:\Datasets\New_cyrrilic"

		"""Test fragment"""
		# self.delete_alpha()
		# self.rename_objects()
		# self.generate_next_data(self.Ncyr)
		# self.GG()

	def GG(self):
		data = self.Ncyr
		k = 0
		for colections in os.listdir(data):
			new_datA = data + f'\\{colections}'
			for img in os.listdir(new_datA):
				k += 1
				print(k)

	def balanced_image(self):
		data = self.Ncyr
		for classes in os.listdir(data):
			new_D = data + f'\\{classes}'
			for image in new_D:
				image = new_D + f'\\{image}'
				print(image)


	def generate_next_data(self, data):
		for colections in os.listdir(data):
			new_data = data + f'\\{colections}'
			for img in os.listdir(new_data):
				image = new_data + f'\\{img}'
				fwe = image.split('.')[0]
				image = Image.open(image)
				annef = np.ndarray((2,), buffer= np.array([-13, 13]), dtype=int)

				for iter in annef:
					trans = transform.rotate(np.array(image), iter, cval=255, preserve_range=True).astype(np.uint8)
					cv2.imwrite(fwe + 'rotate.jpeg', trans)


	def delete_alpha(self):
		data = self.Cyr
		for colections in os.listdir(data):
			new_datA = data + f'\\{colections}'
			for img in os.listdir(new_datA):
				new_data = new_datA + f'\\{img}'
				image = cv2.imread(new_data, cv2.IMREAD_UNCHANGED)
				trans_Mask = image[:, :, 3] == 0
				image[trans_Mask] = [255, 255, 255, 255]
				new_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				cv2.imwrite(self.Ncyr + f'\\{colections}' + f'\\new{img}', new_image)
	def rename_objects(self):
		dir = r'C:\Users\Антонио\PycharmProjects\Test_concurs\Cyrillic'
		for colections in os.listdir(dir):
			nfn = f'{colections}'
			new_dir = dir + f"\\{colections}"

			n = 0
			for image in os.listdir(new_dir):
				n += 1
				os.rename(new_dir + '\\' + image, new_dir + '\\' + nfn + f'({n}).jpeg')




def main():
	Generate_dataset()
if __name__ == '__main__':
	main()