"""
Тестовое задание кандидата

Основная задача:
	Мне необходимо построить базовый алгоритм распознавания текста из документов.
	Основным документом будет являться паспорт РФ ||| path = r"C:\Datasets\Pasports" |||

1. 	подобрать открытые библиотеки по распознаванию документов
	и провести распознавания паспортов с указанной ссылки

2. 	подобрать API c помощью которых можно провести распознавание,
	провести распознавание

3. 	САМЫЙ ВАЖНЫЙ ПУНКТ - обучить собственный алгоритм (pytorch, tensorflow),
	которые распознает ФИО с паспорта (не пользуясь готовыми библиотеками OCR,
	но пользуясь открытыми данными по распознаванию текста)

||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
"""
import numpy as np

from Dataseter import Create_dataset
from Create_OCR_Model import OCR_Model
from Val_gen_split import Validat_split
import keras

def Chain_functions():


	Dataset = Create_dataset().Get_data()
	train, val = Dataset
	history = OCR_Model(train, val).Train_model()



def main():
	Chain_functions()
if __name__ == '__main__':
	main()