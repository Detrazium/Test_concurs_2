from PIL import ImageDraw, ImageFont, Image
import textwrap
from Token_word import OCR_TOKEN
import numpy as np
import time
import cv2
class DatagenImage():
	def __init__(self):
		self.Text = self.ReadText()
		self.Token = self.gen_token_klasser().readTK()
		self.cat_token = self.gen_token_klasser().cat_token()
		self.Cliped = 130
		self.len_dat = 6000
	def ReadText(self):
		with open(r'C:\Users\Антонио\PycharmProjects\Test_concurs\DATASET\New_data.txt', 'r', encoding='utf=8') as File:
			Text = File.read()[:22742700]
			print('Len All Data to: ', len(Text))
		return Text
	def gen_token_klasser(self):
		tokens = OCR_TOKEN()
		return tokens
	def GEN_keys(self, text):
		x = []
		for i in text:
			for key, item in self.Token.items():
				if i == item: x.append(key)
		if len(x) < 300:
			for ii in range(300 - len(x)):
				x.insert(0, 15)
		return x

	def GenImage(self, text):
		lost = Image.new("RGB", (150, 220), (192, 181, 185))
		font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 15, encoding='utf=8')
		drawi = ImageDraw.Draw(lost)

		drawi.multiline_text((20, 10), textwrap.fill(text, 15), font=font, fill=(0, 0, 0))
		lost = lost.convert('L')
		matrics = np.array(lost)
		return matrics
	def StripPrint_texter(self):
		maxs = len(self.Text)
		k = self.Cliped
		X, Y = [], []
		for i in range(0, maxs, self.Cliped):
			f = ''.join(self.Text[i:k])
			k += self.Cliped
			tok = self.GEN_keys(f)
			Img = self.GenImage(f)
			X.append(Img)
			Y.append(tok)
			if len(Y) == self.len_dat:
				return np.array(X), np.array(Y)



def Start_gen():
	key__ = DatagenImage()
	Dataset = key__.StripPrint_texter()
	return Dataset


def main():
	Start_gen()
if __name__ == '__main__':
	main()

