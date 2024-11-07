# drag_for-submit
# 설치 방법
```bash
conda env create -f environment.yaml
conda activate drag_submit_test
```

### X-Pose 사용을 위한 전처리
```bash
cd X-Pose/models/UniPose/ops
python setup.py build install

## test를 위한 부분
python test.py
```
해당 명령어로 X-Pose 모델 사용을 위한,
CUDA operators 모듈을 설치해줘야함.

# 추론 방법

