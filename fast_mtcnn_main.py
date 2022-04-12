from facenet_pytorch import MTCNN
import torch
import cv2
import time
import sys
"""
    device (string): FastMTCNN 을 돌리는 device 의 주체를 자동 으로 선택 해주는 부분. 
                     기본적으로 cuda(GPU)이며, cuda 를 찾지 못할 경우 cpu 로 변경됨.
    file_dir (string): 동영상 파일이 존재 하는 주소. 기본 주소는 main.py 파일이 존재 하는 주소 이며, 
                       다른 주소의 파일을 가져 오고 싶을 경우, 해당 파일의 전체 주소를 기록 하면 된다.
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
file_dir = "test.mp4"


class FastMTCNN(object):
    """
    FastMTCNN class 의 구성

    :argument stride: (int) 얼굴 검출의 폭. 매 stride 프레임 마다 인식을 진행 한다.
                      남은 frame 은 첫번째 frame 에서 인식한 결과를 공유 한다.
                      (예를 들어 stride 가 4라면 매 4프레임 마다 얼굴 인식을 진행.)
    :argument resize: (float) 영상의 크기 (비율)를 조정. (기본값 1.)
    :argument *args: MTCNN constructor 에 사용 되는 독립 변수(argument)들
    :argument **kwargs: MTCNN constructor 에 사용 되는 키워드 독립 변수들
    """
    def __init__(self, stride, resize=1.0, *args, **kwargs):
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        # 영상의 크기를 조절
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]
        # face detect 를 실행
        boxes, probs = self.mtcnn.detect(frames[::self.stride])
        # 각각의 frame 별로 인식된 결과를 바탕 으로 영상에 사각형(인식 결과)을 그림
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(frame, 'Detected Face', (box[0] - 5, box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        return frames


fast_mtcnn = FastMTCNN(
    # FastMTCNN class 에서 정의된 설정 값들로 제작된 fast_mtcnn 객체
    # 각 설정값에 대한 설명은 FastMTCNN 의 argument 정의를 참조
    # stride 의 값이 작아질수록, resize 의 값이 커질수록 사용 하는 메모리 양은 늘어남
    stride=2,
    resize=0.5,
    margin=14,
    factor=0.5,
    keep_all=True,
    device=device
)


def run_detection(f_mtcnn, filename):
    """
    얼굴 인식의 실행 함수. face_detected 에 box 가 그려진 frame 이 저장됨.
    동영상을 인식 하여 batch_size 만큼의 frame 을 frames 에 넣고 한꺼번에 인식.

    :param f_mtcnn: (class FastMTCNN)MTCNN constructor class
    :param filename: (string)인식할 동영상이 저장된 파일 위치
    :return: None
    """
    face_detected = []
    frames = []
    batch_size = 60
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for j in range(v_len):
        _, frame = v_cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) >= batch_size or j == v_len - 1:
            # FastMTCNN 의 __call__ 함수를 호출하여 얼굴 인식 실행
            faces = f_mtcnn(frames)
            frames = []
            for face in faces:
                face_detected.append(face)
    # 사용한 테스트 영상의 fps 를 가져 오는 부분
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    v_cap.release()
    # 가져온 fps 와 box 가 기록된 frame 을 사용 하여 영상을 저장한 뒤 실행
    save_video(face_detected, fps)
    play_video(face_detected, fps)


def save_video(frames, fps):
    """
    opencv 를 사용 하여 비디오를 저장 하는 함수

    :param frames: (list Mat)얼굴 인식 결과가 기록된 프레임들
    :param fps: (int)원본 영상의 fps(frame per second)
    :return: None
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
    if not out.isOpened():
        print('File open failed!')
        sys.exit()
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def play_video(frames, fps):
    """
    opencv 를 사용 하여 비디오를 재생 하는 함수

    :param frames: (list Mat)얼굴 인식 결과가 기록된 프레임들
    :param fps: (int)원본 영상의 fps(frame per second)
    :return:
    """
    frame_count = 0
    prev_time = time.time()
    while True:
        current_time = time.time() - prev_time
        if frame_count < len(frames) and (current_time > 1. / fps):
            prev_time = time.time()
            face = cv2.cvtColor(frames[frame_count], cv2.COLOR_RGB2BGR)
            cv2.imshow('face', face)
            frame_count += 1
        if cv2.waitKey(1) > 0 or frame_count >= len(frames):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_detection(fast_mtcnn, file_dir)
