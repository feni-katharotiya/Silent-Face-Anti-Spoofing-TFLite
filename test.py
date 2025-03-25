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
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


# SAMPLE_IMAGE_PATH = "/home/deepkaneria/HDD/Vrushank/LV_pytorch_training/dataset/benchmark_set_original/222-benchmark_fd_crop/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(images_folder, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    real_dir =  f"./Results_New_18_10_2024/{os.path.basename(images_folder)}/Real"
    fake_dir =  f"./Results_New_18_10_2024/{os.path.basename(images_folder)}/Fake"

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    for image_name in os.listdir(images_folder):
        full_image = os.path.join(images_folder, image_name)
        image = cv2.imread(full_image)
        result = check_image(image)
        if result is False:
            return
        image_bbox = model_test.get_bbox(image)

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
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.".format(os.path.basename(image_name), value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format(os.path.basename(image_name), value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        
        print("Prediction cost {:.2f} s".format(test_speed))
        
        # print(f"{real_dir}/{value:.2f}_{os.path.basename(image_name)}")
        # exit()

        if label == 1: #and value >= 0.99:
            shutil.copy(full_image, f"{real_dir}/{value:.2f}_{os.path.basename(image_name)}")
        else:
            shutil.copy(full_image, f"{fake_dir}/{value:.2f}_{os.path.basename(image_name)}")

    
    # cv2.rectangle(
    #     image,
    #     (image_bbox[0], image_bbox[1]),
    #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
    #     color, 2)
    # cv2.putText(
    #     image,
    #     result_text,
    #     (image_bbox[0], image_bbox[1] - 5),
    #     cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


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
