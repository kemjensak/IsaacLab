# Vectorization
## Caution
- IsaacLab에서 정의하는 env class에서 mdp 관련 정의 및 state등은 Isaac Sim의 environment instance 수 만큼 vectorization(tensor based)되어 있음
- 그러므로, 모든 함수나 클래스를 작성할 때, environment instance와 관련된 모든 데이터가 `torch.tensor`로 처리됨에 유의하여야 함.
  - `if`, `for`문과 같은 조건문, 반복문을 사용할 경우, GPU 기반 연산이 불가하므로, 이로 인한 오버헤드가 매우 커지며, 학습 속도가 저하됨.
  - **그러므로, 모든 계산은 `torch`에서 정의된 메서드를 사용하여야 함.**

## Example
- Unstructured env에서 사용된 reward를 예시로 들면 아래와 같음   

    ### [object_is_lifted](https://github.com/kemjensak/IsaacLab/blob/55ac82b8a84cb9e9b718f736fe17e8dd1944d38a/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/mdp/rewards.py#L399)   

    ```python
    def object_is_lifted(
        env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
    ) -> torch.Tensor:
        """Reward the agent for lifting the object above the minimal height from initial position."""
        object: RigidObject = env.scene[object_cfg.name]
        return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)
    ```

    - `object_is_lifted` 함수는 argument로 `env`, `minimal_height`, `object_cfg를` 받아, object의 z좌표가 `minimal_height`보다 높으면 1의 reward를 return 하는 함수이다.
    - `object.data.root_pos_w[:, 2]`은, 현재 env의 각 instance 별 object의 z축 좌표 값이며 `torch.tensor` 자료형이다.
    - **조건 판단에 `if` 문을 사용하지 않고, `torch.where`를 사용함에 유의할 것.**

    ### [object_is_lifted_from_initial](https://github.com/kemjensak/IsaacLab/blob/55ac82b8a84cb9e9b718f736fe17e8dd1944d38a/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/unstructured/mdp/rewards.py#L53)

    ```python
    class object_is_lifted_from_initial(ManagerTermBase):

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset: RigidObject = env.scene[asset_cfg.name]

        # store initial target object position
        self._initial_object_height = self._asset.data.root_pos_w[:, 2].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        minimal_height: float = 0.04,
    ):
        # Update initial object height where the environment's step is 1
        reset_mask = env.episode_length_buf == 1
        self._initial_object_height[reset_mask] = self._asset.data.root_pos_w[reset_mask, 2].clone()

        # return the reward
        return torch.where(self._asset.data.root_pos_w[:, 2] > (self._initial_object_height + minimal_height), 1.0, 0.0)
    ```

    - `object_is_lifted_from_initial` class의 `__call__` method는 argument로 `env`, `asset_cfg`, `minimal_height`를 받아, object의 z좌표가 `minimal_height + self._initial_object_height` 보다 높으면 1의 reward를 return 한다.
    - 위와는 다르게, `object_is_lifted_from_initial`는 function이 아닌 class이다.
    - 이는 object의 각 env instance 별 initial height을 멤버변수 `self._initial_object_height`에 저장하기 위함이다.
    - class의 `__call__` method를 통해, 매 step 마다, env instance별 step인 `env.episode_length_buf`가 `1`인 경우만을 가져와, `reset_mask`로 reset 여부를 판단하고 `self._initial_object_height`를 업데이트 한다.

    - **다수 env의 상태 판단 및 연산에 `for`문이 아닌 masking 기법을 사용하며, 조건 판단에 `if` 문을 사용하지 않고, `torch.where`를 사용함에 유의할 것.**


## Torch.tensor 관련 함수
- vectorization(tensor)연산에 사용되는 함수들은 아래 링크에서 확인가능


[Torch.tensor documentation] (https://pytorch.org/docs/stable/tensors.html)
