from facenet_pytorch import MTCNN
import torch
from imutils.video import FileVideoStream
import cv2
import time
import numpy as np
from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
filenames = ["test1.mp4"]
output_dir = './output/'


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

        faces = []
        for i, frame in enumerate(frames):
            box_ind = int(i / self.stride)
            if boxes[box_ind] is None:
                continue
            for box in boxes[box_ind]:
                box = [int(b) for b in box]
                faces.append(frame[box[1]:box[3], box[0]:box[2]])

        return faces


# define our extractor


fast_mtcnn = FastMTCNN(
    stride=4,
    resize=0.5,
    margin=14,
    factor=0.5,
    keep_all=True,
    device=device
)


def run_detection(f_mtcnn, files):
    face_detected = []
    frames = []
    frames_processed = 0
    faces_detected = 0
    batch_size = 60
    start = time.time()
    save_count = 0

    for filename in tqdm(files):
        v_cap = FileVideoStream(filename).start()
        v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        for j in range(v_len):
            frame = v_cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if len(frames) >= batch_size or j == v_len - 1:
                faces = f_mtcnn(frames)
                frames_processed += len(frames)
                faces_detected += len(faces)
                frames = []
                for face in faces:
                    face_detected.append(face)
                print(f'Frames per second: {frames_processed / (time.time() - start):.3f},',
                      f'faces detected: {faces_detected}\r')
        v_cap.stop()
        print(f'total running time: {time.time()-start}\r')
    for face in face_detected:
        output_x, output_y, _ = face.shape
        if output_x and output_y:
            img = np.reshape(face, (output_x, output_y, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{output_dir+str(save_count)}_detected_face.jpg', img)
            save_count += 1


if __name__ == '__main__':
    run_detection(fast_mtcnn, filenames)
