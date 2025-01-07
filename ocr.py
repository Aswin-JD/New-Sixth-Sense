import cv2
import easyocr

class OCR:
    def __init__(self, languages=['en'], use_gpu=False):
        self.reader = easyocr.Reader(languages, gpu=use_gpu)

    def extract_text_from_image(self, image_path):
        image = cv2.imread(image_path)
        new_width = 800
        new_height = 600
        resized_image = cv2.resize(image, (new_width, new_height))
        result = self.reader.readtext(resized_image)
        text_concatenated = ''.join([item[1] for item in result])
        if len(text_concatenated) > 0:
            return text_concatenated
        else:
            return "Sorry, I can't answer that question."
