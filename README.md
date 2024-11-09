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



<hr>

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
