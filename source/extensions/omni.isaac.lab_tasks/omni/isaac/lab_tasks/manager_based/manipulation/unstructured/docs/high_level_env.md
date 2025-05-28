# High level environment
- 본 문서는 작성 시기(24/07/29) 특정 commit의 permalink로 작성되었으므로, 최신 코드와 다를 수 있음

---

- 본 문서에서 설명할 High level environment은, 타 environment에서 이미 학습된 model을 1개 이상 load하여, 계층적으로 사용할 수 있도록 해 준다. 

## pre_trained_policy_action.py
- High level policy를 위한 environment configuration class인 `HighLevelEnvCfg`를 [pre_trained_policy_action.py](https://github.com/IROL-SSU/IsaacLab/blob/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_high_level_env_cfg.py)에서 정의함
- low level env와 달리, high level env에서는 `ActionCfg` class에, 새롭게 정의된 `mdp.PreTrainedPolicyActionCfg` class를 사용한다.
- 관련 내용이 코딩된 [#L36](https://github.com/IROL-SSU/IsaacLab/blob/ebea2236b84ac31b920fee6a918c3362eac56a2e/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_high_level_env_cfg.py#L36)의 일부와 이에 대한 설명은 아래와 같다.   
    

  ```python
  ...
  # 이전에 정의한 각 low level env Cfg 불러옴
  LOW_LEVEL_FLIP_ENV_CFG = FrankaFlipObjectEnvCfg()
  LOW_LEVEL_GRASP_ENV_CFG = FrankaGraspObjectEnvCfg()


  ##
  # MDP settings
  ##

  @configclass
  class ActionsCfg:
      """Action specifications for the MDP."""
      # Robot의 직접 제어용 class 가 아닌 pre_trained_policy_action 사용
      pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
          asset_name="robot",
          # 각 policy model 파일 경로 지정,
          grasp_policy_path=f"/home/kjs-dt/RL/orbit/logs/rsl_rl/franka_grasp/2024-06-16_19-47-03/exported/policy.pt",
          flip_policy_path=f"/home/kjs-dt/RL/orbit/logs/rsl_rl/franka_flip/2024-06-15_23-18-36/exported/policy.pt",
          low_level_decimation=2,
          #low level 에서 공통으로 사용하는 ActionCfg class 지정,
          low_level_body_action=LOW_LEVEL_FLIP_ENV_CFG.actions.body_joint_pos,
          low_level_finger_action=LOW_LEVEL_FLIP_ENV_CFG.actions.finger_joint_pos,
          # 각 low level policy의 ObservationCfg class 지정
          low_level_flip_observations=LOW_LEVEL_FLIP_ENV_CFG.observations.policy,
          low_level_grasp_observations=LOW_LEVEL_GRASP_ENV_CFG.observations.policy,
      )
  ...

  ```   


## pre_trained_policy_action.py
- 위에서 설명한 `PreTrainedPolicyActionCfg`와 `PreTrainedPolicyAction`가 [pre_trained_policy_action.py](https://github.com/IROL-SSU/IsaacLab/blob/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/mdp/pre_trained_policy_action.py)에서 실제 구현되어 있다.