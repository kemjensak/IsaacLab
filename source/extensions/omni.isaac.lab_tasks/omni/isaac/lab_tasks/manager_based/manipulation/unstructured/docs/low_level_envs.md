# Environment configuration

- 본 문서는 작성 시기(24/07/12) 특정 commit의 permalink로 작성되었으므로, 최신 코드와 다를 수 있음

## Grasping
### unstructured_grasp_env_cfg.py
- Grasp을 위한 environment configuration class인 `UnstructuredGraspEnvCfg`를 [unstructured_grasp_env_cfg.py](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L395C7-L395C30)에서 정의함
- `UnstructuredGraspEnvCfg`의 super class인 `ManagerBasedRLEnvCfg` class에서 필요한 [scene](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L34), [command](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L92), [action](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L106), [observation](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L115), [event(reset)](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L137), [reward](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L266), [termination](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L319), [curriculum](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_grasp_env_cfg.py#L360) object들을 term instance들로 구성된 class로 각각 정의한다.
-  `UnstructuredGraspEnvCfg`는 이를 사용해 super class에서 정의하지 않았던 object들을 정의하고, 기타 sim 관련 parameter들을 설정한다.

### joint_pos_env_cfg.py
- 위에서 정의하였던 `UnstructuredGraspEnvCfg` class에서, sentinal objcet `MISSING` instance인 `robot`, `ee_frame`, `object`를 [joint_pos_env_cfg.py](https://github.com/kemjensak/IsaacLab/blob/f394abee3b249c583850afdfe3d2aee9833d1d7d/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/config/franka/grasp/joint_pos_env_cfg.py#L26)의 `FrankaGraspObjectEnvCfg` class에서 상속받아 재정의한다.
- 즉, `UnstructuredGraspEnvCfg`에서 Franka 로봇, target object 관련 설정 등을 추가하여 `FrankaGraspObjectEnvCfg`를 정의한다.

### ik_*_env_cfg.py
- 위에서 정의하였던 `FrankaGraspObjectEnvCfg`를 상속받아, `robot`과, `self.actions.body_joint_pos`를 absolute/relative IK controller에 적절한 값으로 변경하여 재정의한다.

### \__init\__.py
- Environment에 대한 ID(`Isaac-Grasp-Object-Franka-v0`,...)와 entrypoint(`ManagerBasedRLEnv`)를 지정하며, environment config entry point로 위에서 정의한 `FrankaGraspObjectEnvCfg` class를 지정한다.
- 그리고, 사전 정의된 train/learn 관련 configuration을 각 RL library의 entry point로 지정한다.
- 마지막으로, [\__init\__.py](https://github.com/kemjensak/IsaacLab/blob/f394abee3b249c583850afdfe3d2aee9833d1d7d/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/config/franka/grasp/__init__.py#L21)의 `gym.register`를 통해 해당 환경을 register 한다.

### rsl_rl_cfg.py(agents)
- 각 RL library, algorithm에 따라 사전 정의된 parameter설정 파일
- RSL_RL의 경우 [rsl_rl_cfg.py](https://github.com/kemjensak/IsaacLab/blob/f394abee3b249c583850afdfe3d2aee9833d1d7d/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/config/franka/grasp/agents/rsl_rl_cfg.py#L16)의 `GraspPPORunnerCfg` class에서 이를 정의함. 
---
## Flipping
### unstructured_flip_env_cfg.py
- Flip을 위한 environment configuration class인 `UnstructuredflipEnvCfg`를 [unstructured_flip_env_cfg.py](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L388)에서 정의함
- `UnstructuredflipEnvCfg`의 super class인 `ManagerBasedRLEnvCfg` class에서 필요한 [scene](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L34), [command](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L88), [action](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L102), [observation](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L111), [event(reset)](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L135), [reward](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L245), [termination](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L309), [curriculum](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/unstructured_flip_env_cfg.py#L354) object들을 term instance들로 구성된 class로 각각 정의한다.
- `UnstructuredflipEnvCfg`는 이를 사용해 super class에서 정의하지 않았던 object들을 정의하고, 기타 sim 관련 parameter들을 설정한다.
### joint_pos_env_cfg.py

### ik_*_env_cfg.py

---

### rewards.py
- `UnstructuredGraspEnvCfg`, `UnstructuredflipEnvCfg`에서 정의하였던 `rewards` object의 `RewardsCfg` class에서 사용되는 `RewTerm`의 argument인 `func`에 들어갈 method, class를 [rewards.py](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/mdp/rewards.py)에서 정의함
- 즉, reward function을 정의함 

### observations.py
- `UnstructuredGraspEnvCfg`, `UnstructuredflipEnvCfg`에서 정의하였던 `observations` object의 `ObservationsCfg ` class에서 사용되는 `ObsTerm`의 argument인 `func`에 들어갈 method, class를 [observations.py](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/mdp/observations.py)에서 정의함
- 즉, reward function을 정의함 

### events.py
- `UnstructuredGraspEnvCfg`, `UnstructuredflipEnvCfg`에서 정의하였던 `rewards` object의 `RewardsCfg` class에서 사용되는 `ObsTerm`의 argument인 `func`에 들어갈 method, class를 [observations.py](https://github.com/kemjensak/IsaacLab/blob/986fd528eb52876316dccde11243e2b92a403779/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/mdp/observations.py)에서 정의함
- 즉, observation값을 sim으로부터 가져오는 function을 정의함

### [pre_trained_policy_action.py](high_level_env.md)

