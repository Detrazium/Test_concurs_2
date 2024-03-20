from PIL import ImageDraw, ImageFont, Image
from io import BytesIO
from Token_word import OCR_TOKEN
import numpy as np
import time
class DatagenImage():
	def __init__(self):
		self.Text = self.ReadText()
		self.Token = self.gen_token_klasser()
	def ReadText(self):
		with open(r'C:\Users\Антонио\PycharmProjects\Test_concurs\DATASET\New_data.txt', 'r', encoding='utf=8') as File:
			Text = File.readlines()
		return Text
	def gen_token_klasser(self):
		tokens = OCR_TOKEN().readTK()
		return tokens
	def GEN_keys(self, text):
		x = []
		for i in text:
			for key, item in self.Token.items():
				if i == item: x.append(key)
		if len(x) < 3500:
			for i in range(3500 - len(x)):
				x.append(15)
		return np.array(x)

	def GenImage(self, text):
		lost = Image.new("RGB", (600, 850), (192, 181, 185))
		font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 15, encoding='utf=8')
		drawi = ImageDraw.Draw(lost)

		drawi.multiline_text((10, 10), text, font=font, fill=(0, 0, 0))
		matrics = np.array(lost)
		return matrics
	def StripPrint_texter(self, itemOne):
		maxs = len(itemOne)
		k = 45
		X, Y = [], []
		for i in range(0, maxs, 45):
			f = ''.join(itemOne[i:k])
			k += 45
			tok = self.GEN_keys(f)
			Img = self.GenImage(f)
			X.append(Img)
			Y.append(tok)

		return np.array(X), np.array(Y)



def Start_gen():
	key__ = DatagenImage()
	itemOne = key__.ReadText()
	Dataset = key__.StripPrint_texter(itemOne)
	return Dataset


def main():
	Start_gen()
if __name__ == '__main__':
	main()

