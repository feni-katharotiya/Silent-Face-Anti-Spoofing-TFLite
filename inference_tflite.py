# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import shutil
import numpy as np
import argparse
import warnings
import math
import time
import tensorflow as tf
from src.generate_patches import CropImage
from src.utility import parse_model_name
from src.anti_spoof_predict import AntiSpoofPredict
warnings.filterwarnings('ignore')


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


class Detection:
    def __init__(self, image):
        caffemodel = "./resources/detection_model/Widerface-RetinaFace.caffemodel"
        deploy = "./resources/detection_model/deploy.prototxt"
        self.detector = cv2.dnn.readNetFromCaffe(deploy, caffemodel)
        self.detector_confidence = 0.6
        self.image = image

    def get_bbox(self):
        img = self.image
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect_ratio)),
                              int(192 / math.sqrt(aspect_ratio))), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = out[max_conf_index, 3]*width, out[max_conf_index, 4]*height, \
                                   out[max_conf_index, 5]*width, out[max_conf_index, 6]*height
        bbox = [int(left), int(top), int(right-left+1), int(bottom-top+1)]
        return bbox

class AntiSpoofPredict:
    def __init__(self, device_id, tflite_model_path):
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()

    def predict(self, img):
        # Prepare the image to match the input size for TFLite model
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_data = np.expand_dims(img, axis=0).astype(np.float32)  # Assuming model takes float32 input
        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get the prediction output
        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        return output_data


def test(images_folder, model_dir, device_id):
    image_cropper = CropImage()
    
    real_dir =  f"./Results_New_18_02_2025_fp16/{os.path.basename(images_folder)}/Real"
    fake_dir =  f"./Results_New_18_02_2025_fp16/{os.path.basename(images_folder)}/Fake"

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Initialize the model_test outside the loop
    for image_name in os.listdir(images_folder):
        full_image = os.path.join(images_folder, image_name)
        image = cv2.imread(full_image)
        result = check_image(image)
        if result is False:
            return
        image_bbox = Detection(image).get_bbox()  # This will work now
        # print(image_bbox.get_bbox())

        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": 80,
                "out_h": 80,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            # cv2.imwrite(f"./Dump_Mc_Real_NoCondition/{scale}_{image_name}", img)
            start = time.time()
            
            # Create the model_test instance inside the loop for each model
            tflite_model_path = os.path.join(model_dir, model_name)
            model_test = AntiSpoofPredict(device_id, tflite_model_path)
            prediction += model_test.predict(img)
            
            test_speed += time.time() - start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        
        if label == 1:
            print(f"Image '{os.path.basename(image_name)}' is Real Face. Score: {value:.2f}.")
            result_text = f"RealFace Score: {value:.2f}"
            color = (255, 0, 0)
        else:
            print(f"Image '{os.path.basename(image_name)}' is Fake Face. Score: {value:.2f}.")
            result_text = f"FakeFace Score: {value:.2f}"
            color = (0, 0, 255)
        
        print(f"Prediction cost {test_speed:.2f} s")
        # print(full_image, f"{real_dir}/{value:.2f}_{os.path.basename(image_name)}")
        # exit()
        if label == 1:  # and value >= 0.99:
            shutil.copy(full_image, f"{real_dir}/{value:.2f}_{os.path.basename(image_name)}")
        else:
            shutil.copy(full_image, f"{fake_dir}/{value:.2f}_{os.path.basename(image_name)}")



if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(args.image_name, args.model_dir, args.device_id)
