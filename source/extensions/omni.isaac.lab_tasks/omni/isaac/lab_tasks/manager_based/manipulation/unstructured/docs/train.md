
# Learning 과정

## train.py 실행

main함수에서,

gym.make를 통해 실행 시 arg로 넣어준 env(RLtask)로 gymnasium의 env클래스 인스턴스 만듦(make도 gymnasium 메서드)

gym.make 설명

- *Creates an environment previously registered with gymnasium.register or a EnvSpec.*
- gymnasium의 env클래스 설명
    
    The main Gymnasium class for implementing Reinforcement Learning Agents environments.
    
    The class encapsulates an environment with arbitrary behind-the-scenes dynamics through the `step` and `reset` functions. An environment can be partially or fully observed by single agents. For multi-agent environments, see PettingZoo.
    
    The main API methods that users of this class need to know are:
    
    - `step` - Updates an environment with actions returning the next agent observation, the reward for taking that actions, if the environment has terminated or truncated due to the latest action and information from the environment about the step, i.e. metrics, debug info.
    - `reset` - Resets the environment to an initial state, required before calling step. Returns the first agent observation for an episode and information, i.e. metrics, debug info.
    - `render` - Renders the environments to help visualise what the agent see, examples modes are "human", "rgb_array", "ansi" for text.
    - `close` - Closes the environment, important when external software is used, i.e. pygame for rendering, databases
    
    Environments have additional attributes for users to understand the implementation
    
    - `action_space` - The Space object corresponding to valid actions, all valid actions should be contained within the space.
    - `observation_space` - The Space object corresponding to valid observations, all valid observations should be contained within the space.
    - `reward_range` - A tuple corresponding to the minimum and maximum possible rewards for an agent over an episode. The default reward range is set to `(-\infty,+\infty)`.
    - `spec` - An environment spec that contains the information used to initialize the environment from `gymnasium.make`
    - `metadata` - The metadata of the environment, i.e. render modes, render fps
    - `np_random` - The random number generator for the environment. This is automatically assigned during `super().reset(seed=seed)` and when assessing `self.np_random`.
    
    Note:
    
    To get reproducible sampling of actions, a seed can be set with `env.action_space.seed(123)`.
    

각 env는 omni.isaac.Isaac Lab_tasks에 있음

gym.make로 만들어진 env객체는 entry_point에 지정된 RLTaskEnv 클래스로 wrapping됨.

---

## class: RLTaskEnv

`source/extensions/omni.isaac.Isaac Lab/omni/isaac/Isaac Lab/envs/rl_task_env.py`

RLTaskEnv클래스는  BaseEnv, gym.Env클래스로부터 다중상속을 받음.

이후 메서드 오버라이딩을 통해 여러 메서드를 RL Task에 맞게 다시 정의함. 

---

## Env Register

cartpole의 경우, init.py에서 register함 (gymnasium 메서드)

source/extensions/omni.isaac.Isaac Lab_tasks/omni/isaac/Isaac Lab_tasks/classic/cartpole/**init**.py

register시, env_cfg_entry_point인 CartpoleEnvCfg는 동일 폴더 내 py 파일의 클래스임.

해당 py파일 내에서는 RLTaskEnvCfg 클래스를 상속받아 CartpoleEnvCfg 클래스 만듦

entry_point가 RLTaskEnv 클래스이므로, 추후 gym.make에서 해당 클래스로 wrapping됨(RLTaskEnv 클래스 인스턴스로 리턴됨)

CartpoleEnvCfg 클래스에는, 부모 클래스인 RLTaskEnvCfg에서 필요로 하는 아래 각각의 클래스를 정의함 

Command
Actions
Observations
Rewards
Randomization
Terminations

---

## RL library 적용

### SB3

sb3의 VecEnv클래스를 상속받은 Sb3VecEnvWrapper로, gym.make를 통해 만들어진 env(gym.register를 통한 registration에서 entry_point가 RLTaskEnv 클래스였으므로, 해당 클래스로 wrapping되었음) 를  wrapping함

즉, 위에서 만들어진 Isaac Lab environment(vectorized, gym.make를 통해 만들어진 RLTaskEnv 인스턴스)를 sb3에 쓸 수 있도록 wrapping.

이후 sb3의 PPO 클래스(PPO implementaion)인스턴스(agent)를 생성하며, 최종 wrapping된 env를 arg로 넣어줌

## Isaac-Lift-Cube-Franka-IK-Abs-v0 구조 예시

- Isaac Lab 예제 중 하나, gym register 시 등록된 ID
- https://isaac-Isaac Lab.github.io/Isaac Lab/source/features/environments.html
- https://github.com/NVIDIA-Omniverse/Isaac Lab/blob/main/source/extensions/omni.isaac.Isaac Lab_tasks/omni/isaac/Isaac Lab_tasks/manipulation/lift/config/franka/ik_abs_env_cfg.py

### gym.register

- `source/extensions/omni.isaac.Isaac Lab_tasks/omni/isaac/Isaac Lab_tasks/manipulation/lift/config/franka/**init**.py`
- skrl_cfg_entry_point 사용

/home/kjs-dt/RL/Isaac Lab/logs/skrl/franka_lift/2024-04-01_20-31-16/checkpoints/best_agent.pt

### Observation

- `source/extensions/omni.isaac.Isaac Lab_tasks/omni/isaac/Isaac Lab_tasks/manipulation/lift/lift_env_cfg.py`
- `joint_pos`, `joint_vel`: panda_joint1~7, panda_finger_joint → **16**
    - The joint positions/velocities of the asset w.r.t. the default joint positions/velocities.
- `object_position`: x, y, z → **3**
    - The position of the object in the robot's root frame.
- `target_object_position`: x, y, z → **3**
    - The generated command from command term in the command manager with the given name.
- `actions`: x, y, z, qx, qy, qz, w → **7**
- 
    - The last input action to the environment.

### Reward

- `reaching_object`: 1 - torch.tanh(object_ee_distance / std)
    - Reward the agent for reaching the object using tanh-kernel
- `reaching_object`: torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    - Reward the agent for lifting the object above the minimal height.
- `object_goal_tracking`: (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
    - Reward the agent for tracking the goal pose using tanh-kernel.
- `object_goal_tracking_fine_grained`: (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
    - Reward the agent for tracking the goal pose using tanh-kernel.

### Penalty

- `action_rate`: Penalize the rate of change of the actions using L2-kernel.
- `joint_vel`: Penalize joint velocities on the articulation using L1-kernel.

### Action

- body_joint_pos: JointPositionActionCfg
- finger_joint_pos: BinaryJointPositionActionCfg

# RL Lib

## SKRL - SAC

https://skrl.readthedocs.io/en/latest/api/agents/sac.html

`/home/kjs-dt/RL/Isaac Lab/_isaac_sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/sac/sac.py`

sample a batch from memory

# SKRL - PPO

https://skrl.readthedocs.io/en/latest/api/agents/ppo.html

## RSL-RL

Actor MLP

Critic MLP

/home/kjs-dt/RL/Isaac Lab/_isaac_sim/kit/python/lib/python3.10/site-packages/skrl/agents/torch/ppo/ppo.py



# 자료정리

# RL 관련

## RL environments, frameworks

### Isaac sim+OIGE

[https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs](https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)

[Isaac Gym: The Next Generation — High-performance Reinforcement Learning in Omniverse | NVIDIA On-Demand](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41582/)

### OIGE 이용한 Mobile manipulator의 MARL implementation,([https://arxiv.org/abs/2309.14792](https://arxiv.org/abs/2309.14792))

[https://github.com/TIERS/isaac-marl-mobile-manipulation](https://github.com/TIERS/isaac-marl-mobile-manipulation)

### latest(2023.1.1) official repo에 MARL 적용

[https://github.com/kemjensak/OmniIsaacGymEnvs](https://github.com/kemjensak/OmniIsaacGymEnvs)

- 적용 완료, SARL, MARL train 구동 확인, SARL의 inference만 성공, MARL은 실패

### Omniverse Isaac Lab

[Isaac Lab: A Unified Simulation Framework for Interactive Robot Learning Environments](https://isaac-Isaac Lab.github.io/)

[AutoRT: Embodied Foundation Models for Large Scale Orchestration of Robotic Agents](https://auto-rt.github.io/)

[https://www.robot-learning.uk/dall-e-bot](https://www.robot-learning.uk/dall-e-bot)

[GenAug](https://genaug.github.io/)
    

## ETC

### RL tips in SB3

[Reinforcement Learning Tips and Tricks — Stable Baselines3 2.3.0a1 documentation](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)


