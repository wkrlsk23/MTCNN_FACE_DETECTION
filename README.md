# MTCNN_FACE_DETECTION
[![](https://img.shields.io/badge/python-%3E%3D3.7-blue)](#) [![](https://img.shields.io/badge/pytorch-1.11.0-orange)](#)  

MTCNN을 이용하여 영상 속 얼굴을 인식하는 프로젝트 예제입니다.

## 시작하기
### 프로젝트 클론 혹은 다운로드
[![](https://img.shields.io/badge/download-click%20this-brightgreen)](https://github.com/wkrlsk23/MTCNN_FACE_DETECTION/archive/refs/heads/main.zip)  

위 버튼을 클릭하여 프로젝트를 다운로드하거나 아래 명령을 통해 저장소를 클론하세요.  

```sh
git clone https://github.com/wkrlsk23/MTCNN_FACE_DETECTION.git
```

### 의존성 패키지 설치
이 예제는 pytorch 계열 의존성 패키지를 사용합니다. 아래 두 명령 중 하나를 사용하여 의존성 패키지를 설치하세요.

```sh
# (권장) NVIDIA CUDA 11.3 환경에서 실행 가능
pip install -r requirements.cuda.txt

# 모든 실행 환경에서 실행 가능, CPU 사용 (맥 환경에서 사용하세요.)
pip install -r requirements.txt
```

혹은 [PyTorch Getting Started](https://pytorch.org/get-started/locally/) 페이지에서 자신의 환경에 맞게 의존성 패키지를 설치할 수 있습니다. 자세한 내용은 저장소를 다운로드받아 동봉된 `CUDA 설치 및 실행 예시.pptx` 파일을 참조하세요.

### 프로젝트 실행
아래 명령을 실행하면 프로젝트를 실행할 수 있습니다.  
```
python main.py
```

## 참조
[Reference.pdf](./Reference.pdf)  
* MTCNN 설명 자료 – https://yeomko.tistory.com/16
* 얼굴인식에서 사용한 학습 데이터와 MTCNN에서 사용된 bounding box 기법 설명 -
https://towardsdatascience.com/how-do-you-train-a-face-detection-model-a60330f15fd5
* opencv에서 동영상을 저장하는 방법 - https://deep-learning-study.tistory.com/108 
* Face Detection using facenet-pytorch - https://github.com/timesler/facenet-pytorch
* Cuda 설치 참고 사이트 - https://pytorch.org/get-started/locally/
      