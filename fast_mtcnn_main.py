from facenet_pytorch import MTCNN
import torch
import cv2
import time
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filename = "test1.mp4"


class FastMTCNN(object):

    def __init__(self, stride, resize=1.0, *args, **kwargs):
        """Constructor for FastMTCNN class.

            Arguments:
                stride (int): The detection stride. Faces will be detected every `stride` frames
                    and remembered for `stride-1` frames.

            Keyword arguments:
                resize (float): Fractional frame scaling. [default: {1}]
                *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
                **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
        """
        self.stride = stride
        self.resize = resize
        self.mtcnn = MTCNN(*args, **kwargs)

    def __call__(self, frames):
        if self.resize != 1:
            frames = [
                cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
                for f in frames
            ]

        boxes, probs = self.mtcnn.detect(frames[::self.stride])

        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(frame, 'Detected Face', (box[0] - 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        return frames


fast_mtcnn = FastMTCNN(
    stride=2,
    resize=0.5,
    margin=14,
    factor=0.5,
    keep_all=True,
    device=device
)


def run_detection(f_mtcnn):
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
            faces = f_mtcnn(frames) #f_mtcnn 위치 정보 찾기 + frame가 결함. , 설치법 관련 ppt/문서 제작. 코멘트 넣기.
            frames = []
            for face in faces:
                face_detected.append(face)
    fps = v_cap.get(cv2.CAP_PROP_FPS)
    v_cap.release()
    save_video(face_detected, fps)
    play_video(face_detected)


def save_video(frames, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))
    if not out.isOpened():
        print('File open failed!')
        sys.exit()
    for frame in frames:
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        out.write(frame)
    out.release()


def play_video(frames):
    frame_count = 0
    prev_time = time.time()
    FPS = 60
    while True:
        current_time = time.time() - prev_time
        if frame_count < len(frames) and (current_time > 1. / FPS):
            prev_time = time.time()
            face = cv2.cvtColor(frames[frame_count], cv2.COLOR_RGB2BGR)
            cv2.imshow('face', face)
            frame_count += 1
        if cv2.waitKey(1) > 0 or frame_count >= len(frames):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_detection(fast_mtcnn)
