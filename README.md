# drag_for-submit
# 설치 방법
```bash
conda env update -f environment.yaml
conda activate drag_submit_test
```
```bash
 pip install git+https://github.com/openai/CLIP.git
```
clip 설치 명령어

### X-Pose 사용을 위한 전처리
```bash
cd X-Pose/models/UniPose/ops
python setup.py build install

## test를 위한 부분
python test.py
```
해당 명령어로 X-Pose 모델 사용을 위한,
CUDA operators 모듈을 설치해줘야함.

## Checkpoints
<!-- insert a table -->
<table>
  <thead>
    <tr style="text-align: center;">
      <th></th>
      <th>name</th>
      <th>backbone</th>
      <th>Keypoint AP on COCO</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr style="text-align: center;">
      <th>1</th>
      <td>X-Pose</td>
      <td>Swin-T</td>
      <td>74.4</td>
      <td><a href="https://drive.google.com/file/d/13gANvGWyWApMFTAtC3ntrMgx0fOocjIa/view"> Google Drive</a> /<a href="https://openxlab.org.cn/models/detail/IDEA-Research/UniPose"> OpenXLab</a>
      <td><a href="https://github.com/IDEA-Research/UniPose/blob/master/config_model/UniPose_SwinT.py">GitHub Link</a></td>
    </tr>
  </tbody>
  <tbody>
    <tr style="text-align: center;">
      <th>2</th>
      <td>X-Pose</td>
      <td>Swin-L</td>
      <td>76.8</td>
      <td> Coming Soon</td>
      <td> Coming Soon</td>
    </tr>
  </tbody>
</table>

위의 체크포인트 파일(.pth)을 X-Pose/weights 폴더에 넣어줘야함

# 추론 방법
#GoodDrag 디렉토리로 들어가는 부분

```
cd ../../../..

cd GoodDrag

#추론을 위한 gradio_ui 파일 실행.

python gooddrag_ui.py
```

<hr>

### gradio내 사용법

1. Upload Image: 이미지를 업로드하고 원하는 부분의 마스크를 그려주세요.(Draw Mask)
2. Prepare for Training: Train LoRA버튼을 누르면, 선택한 이미지에 대한 파라미터가 .lora_data/test(경로 변경 가능)에 저장이 됩니다.
3. Select Keypoint: Prompt에 원하는 부분의 Keypoint를 텍스트로 입력하고, Keypoint Detection버튼을 누르면 됩니다. ex: hand right
    
    X-Pose/predefined_keypoints.py에 keypoiny의 대상과 keypoint에 대한 내용이 들어있습니다.
    
    또한, 방향은 up, down, right, left가 있어 조합하여 사용하시면 됩니다.
    
4. Save Current Data (Optional): 현재 상태(이미지, 마스크, 포인트, 마스크와 포인트가 포함된 합성 이미지 포함)를 저장하고 싶으면, Save Current Data버튼을 눌러주세요. 데이터는 ./dataset/test에 저장이 됩니다.
5. Run Drag Process: Run버튼을 눌러서 drag과정을 진행해주세요.
6. Save the Results (Optional): drag가 된 이미지, 새로운 포인트, 새로운 포인트가 그려진 이미지를 저장하고 싶으면, Save Result버튼을 눌러주세요. 데이터는 ./result/test에 저장이 됩니다.
7. Save Intermediate Images (Optional): drag과정의 중간 과정을 보고 싶으면 Save Intermediate Images칸을 체크해주세요. 그리고 중간 이미지들을 연결해서 만든 비디오를 얻고 싶으면 Get video버튼을 눌러주세요. 중간 과정 이미지들과 비디오가 저장이 됩니다.

<hr>

### 중요 파라미터 소개

1. Drag Parameters
    
    **Prompt**를 통해 원하는 부분의 keypoint를 선택할 수 있습니다.
    
    **End time step**은 총 drag과정을 몇번 하는지 정하는 파라미터입니다.
    
    Point tracking number per each step은 한번 drag과정내에서 point tracking을 몇번 진행하는지 정하는 파라미터입니다.
    
    → ex: End time step 35/ Point tracking number per each step 2
    
    로 진행했을 때, 중간 이미지의 개수는 35개가 됩니다.
    
2. Advanced Parameters
    
    Point tracking의 경우에는 Motion supervision을 진행한 다음 새로운 handle point를 찾는 과정입니다. Point tracking feature patch size는 탐색 범위를 설정하는 변수입니다. 한마디로 Motion supervision feature path size보다 크게 잡는 걸 권장하는 이유는 Motion supervision path size를 r_m라고 했을 때, 실제 정사각형 한 변의 길이는 2r_m + 1이므로 Motion supervision을 진행한 범위 안에서 새로운 handle point를 찾기 위해 Point tracking patch size를 크게 잡는 편이 결과물의 정확도를 높인다고 생각합니다.
    
    Motion supervision feature path size < Point tracking feature patch size (Optional)
```


@article{zhang2024gooddrag,
    title={GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models},
    author={Zhang, Zewei and Liu, Huan and Chen, Jun and Xu, Xiangyu},
    journal = {arXiv},
    year={2024}
}
```

```
@article{xpose,
  title={X-Pose: Detection Any Keypoints},
  author={Yang, Jie and Zeng, Ailing and Zhang, Ruimao and Zhang, Lei},
  journal={ECCV},
  year={2024}
}
```
