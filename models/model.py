import easyocr
import numpy as np

# class ocr_own():
#     def __init__(self, img_path):
#         # img_path = './real-data'
#         reader = easyocr.Reader(['en'])
#         result = reader.readtext(img_path)
#         print(result)


img_path = 'https://m.media-amazon.com/images/I/61I9XdN6OFL.jpg'
reader = easyocr.Reader(['en'])
result = reader.readtext(img_path)
print(result)

# https://m.media-amazon.com/images/I/61I9XdN6OFL.jpg,748919,item_weight,500.0 gram
# https://m.media-amazon.com/images/I/71gSRbyXmoL.jpg,916768,item_volume,1.0 cup
# https://m.media-amazon.com/images/I/61BZ4zrjZXL.jpg,459516,item_weight,0.709 gram
# https://m.media-amazon.com/images/I/612mrlqiI4L.jpg,459516,item_weight,0.709 gram
# https://m.media-amazon.com/images/I/617Tl40LOXL.jpg,731432,item_weight,1400 milligram
# https://m.media-amazon.com/images/I/61QsBSE7jgL.jpg,731432,item_weight,1400 milligram
# https://m.media-amazon.com/images/I/81xsq6vf2qL.jpg,731432,item_weight,1400 milligram