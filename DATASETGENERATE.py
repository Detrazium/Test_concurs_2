from PIL import ImageDraw, ImageFont, Image


class DatagenImage():
	def __init__(self):
		self.Text = self.ReadText()

	def ReadText(self):
		with open(r'C:\Users\Антонио\PycharmProjects\
		Test_concurs\DATASET\ALL_data_in_one.txt',
		'r', encoding='utf=8') as file:
			Text = file.read()
		return Text

	def test_read(self):
		k = 0
		for i in self.Text:
			k += 1
			print(k, '  ', i)

def Start_gen():
	DatagenImage().test_read()

def main():
	Start_gen()
if __name__ == '__main__':
	main()

