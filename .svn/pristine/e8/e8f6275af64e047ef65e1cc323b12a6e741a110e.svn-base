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


def check_and_resize_image(image):
    # img = cv2.imread(image_path)
    # if img is None:
    #     print(f"Failed to load image: {image_path}")
    #     return
    aspect_ratio =(3, 4)
    img = image
    img_height, img_width = img.shape[:2]
    target_aspect = aspect_ratio[0] / aspect_ratio[1]
    
    # Determine current aspect ratio
    current_aspect = img_width / img_height

    if current_aspect > target_aspect:
        # Image is wider than the target aspect ratio, need to crop width
        new_width = int(target_aspect * img_height)
        left = (img_width - new_width) // 2
        right = left + new_width
        img_cropped = img[:, left:right]
    else:
        # Image is taller than the target aspect ratio, need to crop height
        new_height = int(img_width / target_aspect)
        top = (img_height - new_height) // 2
        bottom = top + new_height
        img_cropped = img[top:bottom, :]

    # Resize to specific size
    resized_img = cv2.resize(img_cropped, (aspect_ratio[0] * 100, aspect_ratio[1] * 100), interpolation=cv2.INTER_AREA)

    return resized_img


def process_frame(image, model_test, image_cropper, model_dir, real_dir, fake_dir):
    # Check and resize image to have the 4:3 aspect ratio
    image = check_and_resize_image(image)
    image_bbox = model_test.get_bbox(image)

    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from each model
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
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    # Draw the result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    
    if label == 1:
        # print("Real Face detected. Score: {:.2f}.".format(value))
        result_text = "RealFace Score: {:.2f}".format(value)
        color = (0, 255, 0)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # shutil.copy(image, f"{real_dir}/{value:.2f}_{time.time()}.jpg")
    else:
        # print("Fake Face detected. Score: {:.2f}.".format(value))
        result_text = "FakeFace Score: {:.2f}".format(value)
        color = (0, 0, 255)
        # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        # shutil.copy(image, f"{fake_dir}/{value:.2f}_{time.time()}.jpg")

    # print("Prediction cost {:.2f} s".format(test_speed))
    return image, image_bbox, result_text, color


def process_video(video_source, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    real_dir = "./Results/Real"
    fake_dir = "./Results/Fake"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if the video is horizontal
        if width > height:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        result = process_frame(frame, model_test, image_cropper, model_dir, real_dir, fake_dir)
        if result:
            image, image_bbox, result_text, color = result
            
            # Display the result on the frame
            cv2.rectangle(image,(image_bbox[0], image_bbox[1]), (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)
            cv2.putText(image, result_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.imshow("Anti-Spoofing Detection", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    desc = "Anti-Spoofing detection on images, videos, or webcam"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU ID to use, [0/1/2/3]"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="Directory of the anti-spoofing models"
    )
    parser.add_argument(
        "--video_source",
        type=str,
        default="0",
        help="Path to video file or webcam index (default is '0' for webcam)"
    )
    args = parser.parse_args()

    video_source = int(args.video_source) if args.video_source.isdigit() else args.video_source
    process_video(video_source, args.model_dir, args.device_id)
