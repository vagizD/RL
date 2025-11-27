### Код

Я сделал fork репозитория и закоммитил туда двух агентов и результаты экспериментов: https://github.com/vagizD/sumo-rl.

### Решение

#### Q-learning

В файле `sumo-rl/agents/ql_advanced_agent.py` из fork-а алгоритм Q-learning, у которого есть параметр
обучения - при обновлении
использовать классический Q-learning (награда следующего состояния как максимум), либо использовать SARSA,
который для определения награды следующего состояния также делает честный expolartion. Для этого
добавил в файл `sumo-rl.exploration.epsilon_greedy_with_state` почти копию обычного
`epsilon_greedy`, только в которой есть возможность не обновлять `eps` при вызове `self.choose()` -
это чтобы в SARSA при обновлении ценностей не менять уровень исследования.

Файл `experiments/ql_4x4grid.py` из fork-а содержит код моего эксперимента. Я поменял
только `num_seconds=10_000` вместо `num_seconds=80_000`, остальные параметры не менялись.
Также я прикрепил все эпизоды (папки `4x4` и `4x4-my` включительно).

Результат реализации из репозитория:

![img](ql-4x4.png)

Результат моей реализации (обновление SARSA):

![img](ql-4x4-my.png)

Кривая наград очень похожа. В целом, что у меня, что в репозитории - используется классический Q-learning
(только разное обновление) с одинаковыми уровнями исследования.

#### DQN

В файле `sumo-rl/models/dqn.py` из fork-а алгоритм DQN. Он сделан так, чтобы поддерживать API репозитория и метрики
логировались одинаково. Для этого класс `MyDQN` наследуется от `OffPolicyAlgorithm`, но в качестве политики
использует `DummyPolicy`, которая ничего не делает. Сама `QNet` - неглубокая сетка. Используется
replay buffer и обновление через max (классический Q-learning). В качестве лосса MSE.

Из интересного, я добавил некоторый трюк для уменьшения дисперсии. Вместо одной сетки, будет две
одинаковых `QNet` - `q_net` и `q_net_target`. Первая - та, которую мы постоянно обновляем
градиентным спуском. Вторая - используется только для обновления ценностей (ценность следующего
состояния считается как максимум по выходам этой сетки):

```python
with torch.no_grad():
    next_q = self.q_net_target(replay_data.next_observations)
    next_q_max = next_q.max(dim=1)[0]
    target_q = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * next_q_max

current_q = self.q_net(replay_data.observations)
current_q = current_q.gather(1, replay_data.actions).squeeze(-1)

loss = nn.functional.mse_loss(current_q, target_q)
```

В классическом варианте было бы `next_q = self.q_net(...)`. Это обучение нестабильно,
как и многие алгоритмы в RL. Однако, использование замороженной `q_net_target`
позволяет иметь консистентную оценку следующих состояний для `target_q`. Это
помогает стабилизировать обучение. Сама `q_net_target` просто копирует веса
`q_net` раз в сколько-то шагов обучения, когда `q_net` уже статзначимо обучилась.

Файл `experiments/dqn_big-intersection.py` из fork-а содержит код запуска. Я поменял только
`total_timesteps=100000` на `total_timesteps=20_000` в обучении модели. Эпизоды в
`episodes/big-intersection` - мои начинаются с `my-dqn`.

Результат sb3 `DQN`:

![img](big_intersection.png)

```bash
-----------------------------------
| rollout/            |           |
|    ep_len_mean      | 1.08e+03  |
|    ep_rew_mean      | -1.16e+03 |
|    exploration_rate | 0.01      |
| time/               |           |
|    episodes         | 16        |
|    fps              | 12        |
|    time_elapsed     | 1346      |
|    total_timesteps  | 17280     |
| train/              |           |
|    learning_rate    | 0.001     |
|    loss             | 10.9      |
|    n_updates        | 17279     |
-----------------------------------
```

Результат `MyDQN`:

![img](big_intersection_my.png)

```bash
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 1.08e+03 |
|    ep_rew_mean     | -22.7    |
| time/              |          |
|    episodes        | 16       |
|    fps             | 13       |
|    time_elapsed    | 1260     |
|    total_timesteps | 17280    |
| train/             |          |
|    learning_rate   | 0.001    |
|    loss            | 79.3     |
|    target_update   | 1        |
---------------------------------
```

Получилось достичь более быстрого схождения и лучшей награды, чем в sb3.

