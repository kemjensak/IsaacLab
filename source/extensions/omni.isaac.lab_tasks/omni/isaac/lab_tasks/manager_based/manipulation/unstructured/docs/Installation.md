
# Unstructured Env 설치
[Installation](../Installation.md)과 이어지는 설명입니다.

## Object USD 다운로드
- [링크를](https://raw.githubusercontent.com/IROL-SSU/file_storage/main/objects.zip) 통해 env에서 사용되는 object들의 USD 파일을 다운로드
- 다운로드 받은 파일을 Nvidia omniverse nucleus를 실행시켜, 아래 경로로 복사
  ```
  omniverse://localhost/Library/usd/unstructured/objects/
  ```
- 정상적으로 복사되었으면, 아래 사진과 같은 상태임
  ![alt text](<img/nucleus.png>)



## Train 테스트
### Grasping
- RSL-RL을 통한 `Isaac-Grasp-Object-Franka-v0` 기반 환경에서의 agent training
  - Joint position 제어 

    ```bash
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Grasp-Object-Franka-v0 --headless
    # 학습 완료 후, 결과 확인
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Grasp-Object-Franka-v0 --num_envs 4 --cpu --disable_fabric
    ```

  - Absolute IK position 제어 
    
    ```bash
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Grasp-Object-Franka-IK-Abs-v0 --headless
    # 학습 완료 후, 결과 확인
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Grasp-Object-Franka-IK-Abs-v0 --num_envs 4 --cpu --disable_fabric
    ```

  - Reletive IK position 제어 
    
    ```bash
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Grasp-Object-Franka-IK-Rel-v0 --headless
    # 학습 완료 후, 결과 확인
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Grasp-Object-Franka-IK-Rel-v0 --num_envs 4 --cpu --disable_fabric
    ```
    
    

### Flipping
- RSL-RL을 통한 `Isaac-Flip-Object-Franka-v0`기반 환경에서의 agent training
  - Joint position 제어

    ```bash
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Flip-Object-Franka-v0 --headless
    # 학습 완료 후, 결과 확인
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Flip-Object-Franka-v0 --num_envs 4 --cpu --disable_fabric
    ```

  - Absolute IK position 제어 
    
    ```bash
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Flip-Object-Franka-IK-Abs-v0 --headless
    # 학습 완료 후, 결과 확인
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Flip-Object-Franka-IK-Abs-v0 --num_envs 4 --cpu --disable_fabric
    ```

  - Reletive IK position 제어 
    
    ```bash
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Flip-Object-Franka-IK-Rel-v0 --headless
    # 학습 완료 후, 결과 확인
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Flip-Object-Franka-IK-Rel-v0 --num_envs 4 --cpu --disable_fabric
    ```

