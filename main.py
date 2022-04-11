import sys
import os
import numpy as np
from tkinter import filedialog
from tkinter import messagebox
from facenet_pytorch import MTCNN
import torch
from imutils.video import FileVideoStream
import cv2
import time
import glob
from tqdm.notebook import tqdm

# 목적은 python 프로그램하고, simple하게 CNN 사용법을 알 수 있게 하는 것.
# 되도록 코드를 준비하고, 그 이후 구체적인 안은 상담하도록.

# 글로벌 변수 작성 (video, photo 에 필요할 경우 형식을 추가)
video = ["avi", "mp4"]
photo = ["png", "jpeg", "jpg"]
divider = 4
detector = MTCNN()


# 한글 경로 파일 읽기
def korean_directory_import(path):
    img_array = np.fromfile(path, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img


# 한글 경로 파일 출력
def korean_directory_export(path, img):
    ret, img_arr = cv2.imencode(os.path.splitext(path)[1], img, cv2.IMREAD_COLOR)
    if ret:
        with open(path, mode='w+b') as f:
            img_arr.tofile(f)


def face_detecting(frame):
    boxes = []
    faces = detector.detect_faces(frame)
    for face in faces:
        boxes.append(face['box'])
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Detected Face', (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return frame


def photo_reading(file):
    # 그림 불러 오기 (한글 경로로)
    image = korean_directory_import(file)

    # 얼굴 인식 부분
    image = face_detecting(image)
    cv2.imshow('Face', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_reading(file):
    # 비디오 캡처 및 조정창 생성
    cap = cv2.VideoCapture(file)
    frame_width = int(cap.get(3)/divider)
    frame_height = int(cap.get(4)/divider)
    size = (frame_width, frame_height)
    print(size)
    frame_num = 0
    while True:
        # frame 읽기 및 FPS 조정
        ret, frame = cap.read()
        frame_num += 1
        if ret is True:
            # 얼굴 인식 부분
            start_time = time.time()
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = face_detecting(frame)
            last_time = time.time() - start_time
            cv2.imshow('Face', frame)
            print("fps : ", int(1. / last_time), "  frame : ", frame_num)
            if cv2.waitKey(1) == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 파일 타입 설정
    video_types = ""
    photo_types = ""
    for t in video:
        video_types = video_types + "*." + t + " "
    for t in photo:
        photo_types = photo_types + "*." + t + " "
    video_types = video_types[:-1]
    photo_types = photo_types[:-1]

    # 파일 불러 오기, 파일이 없을 때 앱을 강제 종료
    files = filedialog.askopenfilenames(initialdir=os.path.join(os.path.expanduser('~'), 'Desktop'),
                                        title="파일을 선택 해 주세요",
                                        filetypes=[("video (" + video_types.replace(" ", ", ") + ")", video_types),
                                                   ("photo (" + photo_types.replace(" ", ", ") + ")", photo_types)])
    if files == '':
        messagebox.showwarning("경고", "파일을 추가 하세요")
        sys.exit()

    # 파일 경로 다듬기
    files = files[0].replace("/", "\\\\")
    form = files.split(".")[-1]

    # form 은 마지막 단어(파일의 형식)을 의미함.
    if form in video:
        video_reading(files)
    elif form in photo:
        photo_reading(files)
