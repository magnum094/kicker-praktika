from src.sb3_contrib.sb3_contrib.common.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)

MlpLstmPolicy = RecurrentActorCriticPolicy
CnnLstmPolicy = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy
