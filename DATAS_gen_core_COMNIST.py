collection = r"C:\Datasets\Cyrillic"
cyrril = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ'

import os
import cv2
from PIL import Image
from skimage import transform
from skimage.transform import AffineTransform, warp, rotate
import numpy as np
import shutil
from Token_word import OCR_TOKEN
class Generate_dataset():
	"""
	Gen to need min(115 000):
	49203 |Catalog to| А!
	112232 |Catalog to| И!
	63984 |Catalog to| Й!
	118017 |Catalog to| К
	68864 |Catalog to| Л!
	93968 |Catalog to| М!
	115358 |Catalog to| Н
	120478 |Catalog to| О
	22424 |Catalog to| П!
	35718 |Catalog to| Р!
	34604 |Catalog to| С!
	20654 |Catalog to| Б!
	24692 |Catalog to| Т!
	21468 |Catalog to| У!
	25740 |Catalog to| Ф!
	37356 |Catalog to| Х!
	25410 |Catalog to| Ц!
	36872 |Catalog to| Ч!
	22814 |Catalog to| Ш!
	23144 |Catalog to| Щ!
	43428 |Catalog to| Ъ!
	24684 |Catalog to| Ы!
	101816 |Catalog to| В!
	18282 |Catalog to| Ь!
	18964 |Catalog to| Э!
	25696 |Catalog to| Ю!
	36300 |Catalog to| Я!
	95986 |Catalog to| Г!
	79112 |Catalog to| Д!
	22784 |Catalog to| Е!
	16456 |Catalog to| Ё!
	36828 |Catalog to| Ж!
	33924 |Catalog to| З!
	"""

	def __init__(self):
		# self.Cyr = r"C:\Datasets\Cyrillic"
		self.Ncyr = r"C:\Datasets\New_cyrrilic"
		self.NC_pech = r"C:\Datasets\NC_Ppech"

		"""Test fragment"""
		# self.Create_catalofert(self.NC_pech)
		# self.Movered_pech()
		# self.Regrite_XMS_Trash(self.NC_pech)
		# self.shift_image(self.NC_pech)
		# self.create_cataloger()
		# self.delete_alpha()
		# self.rename_objects()
		# self.generate_next_data(self.NC_pech)
		# self.GG(self.NC_pech)
		self.Coll_catag(self.Ncyr)

	def Coll_catag(self, data):
		for catalog in os.listdir(data):
			new_data = data+ '\\' + catalog
			k=0
			for img in os.listdir(new_data):
				k+=1
			print(f'{k} |Catalog to| {OCR_TOKEN(int(catalog)-1).get_lit()}')


	def Create_catalofert(self, data):
		for i in range(1, 34):
			os.mkdir(data + f'\\{i}')

	def Movered_pech(self):
		ccclip = ['10', '12', '13', '14', '15', '16']

		DInto = self.Ncyr
		Dout = self.NC_pech
		for col1, col2 in zip(os.listdir(Dout), os.listdir(DInto)):
			out = Dout + f'\\{col1}'
			into = DInto + f'\\{col1}'
			l = 0
			k = 0
			if col1 in ccclip:
				for img in os.listdir(out):
					if k%12 == 0:
						l+=1
						# shutil.move(os.path.join(out, img), into)
					k += 1
				print(f'  {col1}:|CLASS|: moved to: {l}')
			else:
				for img in os.listdir(out):
					k+=1
					# shutil.move(os.path.join(out, img), into)
				print(f'  {col1}:|CLASS|: moved to: {k} END_OK')
	def Regrite_XMS_Trash(self, data):
		for colections in os.listdir(data):
			kk = ['3', '4', '8', '5', '9']
			if colections in kk:
				new_data = data + f'\\{colections}'
				for img in os.listdir(new_data):
					image = new_data + f'\\{img}'
					if 'копия' not in image:
						fwe = image.split('.')[0]
						image = cv2.imread(image)
						image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
						for i in range(100):
							image = cv2.resize(image, (150, 150))
							image = cv2.resize(image, (278, 278))
							if i%10 == 0:
								image = cv2.erode(image, None, iterations=1)
							# cv2.imshow('file', image)
							# cv2.waitKey()

							cv2.imwrite(fwe + f'ErodietPech_({i})_.jpeg', image)

	def shift_image(self, data):
		arr_trans = [[15, -15], [-15, 15], [-15, -15],
					 [15, 15], [16, 14], [-14, 14],
					 [17, 14], [-16, 16], [16, -16],
					 [16, 16]]
		for colections in os.listdir(data):
			new_data = data + f'\\{colections}'
			for img in os.listdir(new_data):
				image = new_data + f'\\{img}'
				if 'копия' not in image:
					flt_tgr = image.split('.')[0]
					image = cv2.imread(image)
					for i in range(10):
						trans = AffineTransform(translation=tuple(arr_trans[i]))
						warper = warp(image, trans, mode='wrap')
						img_conv = cv2.convertScaleAbs(warper, alpha=(255.0))
						cv2.imwrite(flt_tgr + f'{i}_transform_.jpeg', img_conv)

	def create_cataloger(self):
		from PIL import ImageDraw, Image, ImageFont
		for i, l in zip(range(1, 34), cyrril):
			font = ImageFont.truetype("C:\Windows\Fonts\Georgia.ttf", size=180)
			img = Image.new("RGB", (278, 278), color=(255, 255, 255))
			draw = ImageDraw.Draw(img)
			draw.text((60, 40), l, fill=(0, 0, 0), font=font)
			img.save(self.NC_pech + f'\\{i}' +f'\\{i}.jpeg')
			pass

	def GG(self, data):
		k = 0
		for colections in os.listdir(data):
			if colections == '28':
				new_datA = data + f'\\{colections}'
				for img in os.listdir(new_datA):
					if 'копия' not in img:
						k += 1
						print(k)

	def balanced_image(self, data):
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
				if 'копия' not in image:
					fwe = image.split('.')[0]
					image = Image.open(image)
					annef = np.ndarray((2,), buffer= np.array([-13, 13]), dtype=int)

					for iter in annef:
						trans = transform.rotate(np.array(image), iter, cval=255, preserve_range=True).astype(np.uint8)
						cv2.imwrite(fwe + 'rotateDTD.jpeg', trans)


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