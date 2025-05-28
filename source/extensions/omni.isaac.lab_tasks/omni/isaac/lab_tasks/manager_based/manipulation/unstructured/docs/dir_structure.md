# Directory Structure
- 본 문서는 작성 시기(24/07/23) 특정 commit을 기준으로 작성되었으므로, 최신 코드와 다를 수 있음

## Tree from `unstructured` directory
- 아래에서 설명하는 `unstructured` directory는 repository 내`/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured` 에 위치해 있음

📦unstructured   
 ┣ 📂config - `Unstructured*EnvCfg에  대한 설정`   
 ┃ ┣ 📂franka - `env의 robot object로 사용할 franka 하위 설정`   
 ┃ ┃ ┣ 📂flip - `franka를 사용하는 flip env 관련 설정`   
 ┃ ┃ ┃ ┣ 📂agents - `flip env 에 사용될 agent 관련 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜__init__.py - `rsl-rl의 config python file import`   
 ┃ ┃ ┃ ┃ ┣ 📜rl_games_ppo_cfg.yaml - `rl_games 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜rsl_rl_cfg.py - `rsl-rl 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜sb3_ppo_cfg.yaml - `sb3 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┗ 📜skrl_ppo_cfg.yaml - `skrl 파라미터 설정`   
 ┃ ┃ ┃ ┣ 📜__init__.py - `flip env를 entry point별로 register`   
 ┃ ┃ ┃ ┣ 📜ik_abs_env_cfg.py - `IK-absolute 방식 제어 env class 정의`   
 ┃ ┃ ┃ ┣ 📜ik_rel_env_cfg.py - `IK-reletive 방식 제어 env class 정의`   
 ┃ ┃ ┃ ┗ 📜joint_pos_env_cfg.py - `joint position 방식 제어 env class 정의`   
 ┃ ┃ ┣ 📂grasp - `franka를 사용하는 grasp env 관련 설정`   
 ┃ ┃ ┃ ┣ 📂agents - `grasp env 에 사용될 agent 관련 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜__init__.py - `rsl-rl의 config python file import`   
 ┃ ┃ ┃ ┃ ┣ 📜rl_games_ppo_cfg.yaml - `rl_games 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜rsl_rl_cfg.py - `rsl-rl 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜sb3_ppo_cfg.yaml - `sb3 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜skrl_ppo_cfg.yaml - `skrl 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜skrl_ppo_cnn_cfg.yaml - `CNN-PPO용 skrl 파라미터 설정(미완성)`   
 ┃ ┃ ┃ ┃ ┗ 📜skrl_sac_cfg.yaml - `SAC용 skrl 파라미터 설정(미완성)`   
 ┃ ┃ ┃ ┣ 📜__init__.py - `grasp env를 entry point별로 register`   
 ┃ ┃ ┃ ┣ 📜ik_abs_env_cfg.py - `IK-absolute 방식 제어 env class 정의`   
 ┃ ┃ ┃ ┣ 📜ik_rel_env_cfg.py - `IK-reletive 방식 제어 env class 정의`   
 ┃ ┃ ┃ ┣ 📜ik_rel_env_cfg_sac.py - `SAC + joint position 방식 제어 env class 정의(미완성)`   
 ┃ ┃ ┃ ┗ 📜joint_pos_env_cfg.py - `joint position 방식 제어 env class 정의`   
 ┃ ┃ ┣ 📂high_level - `franka를 사용하는 high_level env 관련 설정`   
 ┃ ┃ ┃ ┣ 📂agents - `high_level env 에 사용될 agent 관련 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜__init__.py   
 ┃ ┃ ┃ ┃ ┣ 📜rl_games_ppo_cfg.yaml - `rl_games 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜rsl_rl_cfg.py - `rsl-rl의 config python file import`   
 ┃ ┃ ┃ ┃ ┣ 📜sb3_ppo_cfg.yaml - `sb3 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜skrl_ppo_cfg.yaml - `skrl 파라미터 설정`   
 ┃ ┃ ┃ ┃ ┣ 📜skrl_ppo_cnn_cfg.yaml - `CNN-PPO용 skrl 파라미터 설정(미완성)`   
 ┃ ┃ ┃ ┃ ┗ 📜skrl_sac_cfg.yaml - `SAC용 skrl 파라미터 설정(미완성)`   
 ┃ ┃ ┃ ┗ 📜__init__.py - `high_level env를 entry point별로 register`   
 ┃ ┃ ┗ 📜__init__.py   
 ┃ ┗ 📜__init__.py   
 ┣ 📂mdp - `unstructured env의 mdp 관련 정의, 설정`   
 ┃ ┣ 📂commands - `CommandsCfg에 사용되는 object 정의(현재 미사용)`   
 ┃ ┃ ┣ 📜__init__.py   
 ┃ ┃ ┣ 📜commands_cfg.py   
 ┃ ┃ ┗ 📜pose_command.py   
 ┃ ┣ 📜__init__.py - `현위치의 mdp관련 python file import`   
 ┃ ┣ 📜events.py - `env에서 object의 pose set를 저장/불러오는 함수 정의`   
 ┃ ┣ 📜observations.py - `env의 ObservationsCfg class에 사용될 함수 정의`   
 ┃ ┣ 📜pre_trained_policy_action.py - `high_level env의 ActionsCfg class에 사용될 objcet 정의`   
 ┃ ┣ 📜rewards.py - `env의 RewardsCfg class에 사용될 함수 정의`   
 ┃ ┗ 📜terminations.py - `env의 TerminationsCfg class에 사용될 함수 정의`   
 ┣ 📜__init__.py   
 ┣ 📜unstructured_env_tools.py - `env에서 USD 파일을 불러오기 위한 함수 정의`   
 ┣ 📜unstructured_flip_env_cfg.py - `flip env인 UnstructuredFlipEnvCfg 정의`   
 ┣ 📜unstructured_grasp_env_cfg.py - `grasp env인 UnstructuredGraspEnvCfg 정의`   
 ┗ 📜unstructured_high_level_env_cfg.py - `high_level env인 HighLevelEnvCfg 정의`