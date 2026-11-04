"""Robot controller for a budget-constrained goal-reaching task.

Behavioural cloning of a few expert demonstrations, a learned dynamics model that is retrained online,
a goal-progress score learned from the observations, and model predictive control at test time
(random shooting with a gradient-based planner as fallback).

The class `Robot` at the bottom is the interface used by the simulation framework: `training_action`,
`receive_transition`, `receive_demo` and `testing_action`. All hyperparameters are set at the top of the file.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import constants

SEED = 1

NUM_DEMOS = 3
DEMO_LENGTH = 30
SINGLE_DEMO_COST = constants.COST_PER_DEMO + DEMO_LENGTH * constants.COST_PER_DEMO_STEP
TRAINING_MONEY_RESERVE = 3.0

DYNAMICS_HIDDEN_SIZE = 128
DYNAMICS_LEARNING_RATE = 3e-4
DYNAMICS_EPOCHS_INITIAL = 300
DYNAMICS_EPOCHS_ONLINE = 30
DYNAMICS_RETRAIN_INTERVAL = 100
MINIBATCH_SIZE = 64

POLICY_HIDDEN_SIZE = 128
POLICY_LEARNING_RATE = 3e-4
POLICY_EPOCHS = 400

EXPLORATION_NOISE_START = 0.5
EXPLORATION_NOISE_END = 0.05
EXPLORATION_NOISE_DECAY_STEPS = 3000

BC_IMPROVEMENT_THRESHOLD = 0.0

SHOOTER_NUM_RANDOM = 30
SHOOTER_HORIZON = 3


GRAD_REFINE_STEPS = 8
GRAD_REFINE_HORIZON = 4
GRAD_RECOVER_STEPS = 60
GRAD_RECOVER_HORIZON = 8
GRAD_LR = 0.1

STUCK_PATIENCE = 15

class DynamicsNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        in_sz = constants.OBSERVATION_DIMENSION + constants.ACTION_DIMENSION
        out_sz = constants.OBSERVATION_DIMENSION
        self.network = nn.Sequential(
            nn.Linear(in_sz, DYNAMICS_HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, DYNAMICS_HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, DYNAMICS_HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(DYNAMICS_HIDDEN_SIZE, out_sz),
        )

    def forward(self, x):
        return self.network(x)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(constants.OBSERVATION_DIMENSION, POLICY_HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_SIZE, POLICY_HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(POLICY_HIDDEN_SIZE, constants.ACTION_DIMENSION), nn.Tanh(),
        )

    def forward(self, x):
        return self.network(x) * constants.MAX_ACTION_MAGNITUDE


class ReplayBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.demo_observations = []
        self.demo_actions = []

    @property
    def num_transitions(self):
        return len(self.observations)

    @property
    def num_demo_steps(self):
        return len(self.demo_observations)

    def add_transition(self, obs, action, next_obs):
        self.observations.append(obs.copy())
        self.actions.append(action.copy())
        self.next_observations.append(next_obs.copy())

    def add_demo_transition(self, obs, action, next_obs):
        self.demo_observations.append(obs.copy())
        self.demo_actions.append(action.copy())
        self.add_transition(obs, action, next_obs)

    def dynamics_minibatches(self, batch_size):
        n = self.num_transitions
        if n < batch_size:
            return
        obs_arr = np.array(self.observations, dtype=np.float32)
        act_arr = np.array(self.actions, dtype=np.float32)
        delta_arr = np.array(self.next_observations, dtype=np.float32) - obs_arr
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i: i + batch_size]
            yield (
                torch.tensor(np.concatenate([obs_arr[b], act_arr[b]], axis=1), dtype=torch.float32),
                torch.tensor(delta_arr[b], dtype=torch.float32),
            )

    def policy_minibatches(self, batch_size):
        n = self.num_demo_steps
        if n == 0:
            return
        obs_arr = np.array(self.demo_observations, dtype=np.float32)
        act_arr = np.array(self.demo_actions, dtype=np.float32)
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i: i + batch_size]
            yield (
                torch.tensor(obs_arr[b], dtype=torch.float32),
                torch.tensor(act_arr[b], dtype=torch.float32),
            )


class DynamicsModel:
    def __init__(self):
        self.network = DynamicsNetwork()
        self.optimiser = optim.Adam(self.network.parameters(), lr=DYNAMICS_LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.is_trained = False

    def train(self, buffer, num_epochs):
        if buffer.num_transitions < MINIBATCH_SIZE:
            return
        for _ in range(num_epochs):
            for inputs, targets in buffer.dynamics_minibatches(MINIBATCH_SIZE):
                self.network.train()
                loss = self.loss_fn(self.network(inputs), targets)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
        self.is_trained = True

    def predict_next_obs(self, obs, action):
        x = np.concatenate([obs, action]).astype(np.float32)
        self.network.eval()
        with torch.no_grad():
            delta = self.network(torch.tensor(x)).numpy()
        return obs + delta

    def predict_next_obs_batch(self, obs, actions):
        n = actions.shape[0]
        obs_tiled = np.tile(obs, (n, 1)).astype(np.float32)
        x = np.concatenate([obs_tiled, actions.astype(np.float32)], axis=1)
        self.network.eval()
        with torch.no_grad():
            deltas = self.network(torch.tensor(x)).numpy()
        return obs_tiled + deltas

    def rollout_batch(self, obs, actions, horizon):
        self.network.eval()
        n = actions.shape[0]
        current_obs = np.tile(obs, (n, 1)).astype(np.float32)
        with torch.no_grad():
            for _ in range(horizon):
                x = torch.tensor(
                    np.concatenate([current_obs, actions.astype(np.float32)], axis=1),
                    dtype=torch.float32
                )
                deltas = self.network(x).numpy()
                current_obs = current_obs + deltas
        return current_obs

    def rollout_differentiable(self, obs_tensor, action_params, horizon):
        self.network.eval()
        max_a = constants.MAX_ACTION_MAGNITUDE
        curr = obs_tensor
        for t in range(horizon):
            a = torch.tanh(action_params[t]) * max_a
            curr = curr + self.network(torch.cat([curr, a]))
        return curr


class BehaviouralCloningPolicy:
    def __init__(self):
        self.network = PolicyNetwork()
        self.optimiser = optim.Adam(self.network.parameters(), lr=POLICY_LEARNING_RATE)
        self.loss_fn = nn.MSELoss()
        self.is_trained = False

    def train(self, buffer, num_epochs):
        if buffer.num_demo_steps == 0:
            return
        for _ in range(num_epochs):
            for inputs, targets in buffer.policy_minibatches(MINIBATCH_SIZE):
                self.network.train()
                loss = self.loss_fn(self.network(inputs), targets)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
        self.is_trained = True

    def predict_action(self, obs):
        self.network.eval()
        with torch.no_grad():
            a = self.network(torch.tensor(obs.astype(np.float32))).numpy()
        return np.clip(a, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)

class GoalProgressScorer:

    def __init__(self):
        self.weights = None
        self.goal_dimension = True

    def calibrate(self, obs_list, distance_list):
        if len(obs_list) < 10:
            return
        obs_arr = np.array(obs_list, dtype=np.float32)
        neg_dist = -np.array(distance_list, dtype=np.float32)
        D = obs_arr.shape[1]
        raw_weights = np.zeros(D, dtype=np.float32)
        for d in range(D):
            c = np.corrcoef(obs_arr[:, d], neg_dist)[0, 1]
            raw_weights[d] = max(0.0, 0.0 if np.isnan(c) else c)

        total = raw_weights.sum()
        if total > 1e-8:
            self.weights = raw_weights / total
        else:
            self.weights = np.ones(D, dtype=np.float32) / D

    def score(self, obs):
        if self.weights is None:
            return float(np.max(obs))
        return float(np.dot(self.weights, obs))

    def score_batch(self, obs_batch):
        if self.weights is None:
            return np.max(obs_batch, axis=1)
        return obs_batch @ self.weights

    def score_tensor(self, obs_tensor):
        if self.weights is None:
            return obs_tensor.max()
        w = torch.tensor(self.weights, dtype=torch.float32)
        return torch.dot(w, obs_tensor)


class RandomShootingPlanner:

    def __init__(self, scorer, dynamics):
        self.scorer = scorer
        self.dynamics = dynamics

    def shoot(self, obs):
        max_a = constants.MAX_ACTION_MAGNITUDE
        random_actions = np.random.uniform(
            -max_a, max_a,
            (SHOOTER_NUM_RANDOM, constants.ACTION_DIMENSION)
        ).astype(np.float32)
        biased_actions = np.array([
            [max_a * 0.8,  0.0],
            [max_a * 0.8,  max_a * 0.8],
            [max_a * 0.8, -max_a * 0.8],
            [0.0, max_a * 0.8],
            [0.0, -max_a * 0.8],
        ], dtype=np.float32)
        candidates = np.concatenate([random_actions, biased_actions], axis=0)

        final_obs_batch = self.dynamics.rollout_batch(obs, candidates, SHOOTER_HORIZON)
        scores = self.scorer.score_batch(final_obs_batch)
        return candidates[int(np.argmax(scores))]


class GradientPlanner:
    def __init__(self, scorer, dynamics):
        self.scorer = scorer
        self.dynamics = dynamics

    def _optimise(self, obs, init_params, num_steps, horizon):
        max_a = constants.MAX_ACTION_MAGNITUDE
        params = torch.tensor(init_params.copy(), dtype=torch.float32, requires_grad=True)
        opt = optim.Adam([params], lr=GRAD_LR)
        obs_t = torch.tensor(obs.astype(np.float32))
        best_score = -1e9
        best_action = np.zeros(constants.ACTION_DIMENSION, dtype=np.float32)
        for _ in range(num_steps):
            opt.zero_grad()
            final_obs = self.dynamics.rollout_differentiable(obs_t, params, horizon)
            score = self.scorer.score_tensor(final_obs)
            (-score).backward()
            opt.step()
            if score.item() > best_score:
                best_score = score.item()
                best_action = (torch.tanh(params[0]) * max_a).detach().numpy().copy()
        return np.clip(best_action, -max_a, max_a)

    def refine(self, obs, bc_action):
        max_a = constants.MAX_ACTION_MAGNITUDE
        ratio = np.clip(bc_action / max_a, -0.999, 0.999)
        warm = np.arctanh(ratio).astype(np.float32)
        init = np.zeros((GRAD_REFINE_HORIZON, constants.ACTION_DIMENSION), dtype=np.float32)
        init[0] = warm
        return self._optimise(obs, init, GRAD_REFINE_STEPS, GRAD_REFINE_HORIZON)

    def recover(self, obs):
        init = np.random.uniform(
            -1.0, 1.0, (GRAD_RECOVER_HORIZON, constants.ACTION_DIMENSION)
        ).astype(np.float32)
        return self._optimise(obs, init, GRAD_RECOVER_STEPS, GRAD_RECOVER_HORIZON)


class Robot:
    def __init__(self):
        self.environment = None
        self.visualisation_lines = []

        self.buffer = ReplayBuffer()
        self.dynamics = DynamicsModel()
        self.bc_policy = BehaviouralCloningPolicy()
        self.scorer = GoalProgressScorer()

        self._random_shooter = None
        self._gradient_planner = None

        self._demos_requested = 0
        self._demos_received = 0
        self._training_step_count = 0
        self._last_retrain_buffer_size = 0
        self._last_reset_step = -9999
        self._calibration_obs = []
        self._calibration_distances = []

        self._test_best_score = -1e9
        self._test_steps_without_progress = 0

    def _get_random_shooter(self):
        if self._random_shooter is None:
            self._random_shooter = RandomShootingPlanner(self.scorer, self.dynamics)
        return self._random_shooter

    def _get_gradient_planner(self):
        if self._gradient_planner is None:
            self._gradient_planner = GradientPlanner(self.scorer, self.dynamics)
        return self._gradient_planner

    def training_action(self, obs, money):
        if money < TRAINING_MONEY_RESERVE:
            return 4, None

        if (self._demos_requested < NUM_DEMOS
                and self._demos_requested == self._demos_received):
            if money >= SINGLE_DEMO_COST + TRAINING_MONEY_RESERVE:
                self._demos_requested += 1
                return 3, DEMO_LENGTH
            else:
                self._demos_requested = NUM_DEMOS

        steps_since_reset = self._training_step_count - self._last_reset_step
        if (self._training_step_count > 0
                and self._training_step_count % 300 == 0
                and steps_since_reset >= 150
                and money >= constants.COST_PER_RESET + TRAINING_MONEY_RESERVE):
            self._last_reset_step = self._training_step_count
            return 2, None

        if self.bc_policy.is_trained:
            action = self.bc_policy.predict_action(obs)
            decay = np.exp(-self._training_step_count / EXPLORATION_NOISE_DECAY_STEPS)
            noise_scale = (
                EXPLORATION_NOISE_END
                + (EXPLORATION_NOISE_START - EXPLORATION_NOISE_END) * decay
            )
            noise = np.random.normal(
                0, noise_scale * constants.MAX_ACTION_MAGNITUDE,
                size=constants.ACTION_DIMENSION
            )
            action = np.clip(
                action + noise,
                -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE
            )
        else:
            action = np.random.uniform(
                -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE,
                constants.ACTION_DIMENSION
            )

        self._training_step_count += 1
        return 1, action

    def receive_transition(self, obs, action, next_obs, distance_to_goal):
        self.buffer.add_transition(obs, action, next_obs)
        self._calibration_obs.append(obs.copy())
        self._calibration_distances.append(distance_to_goal)

        new_transitions = self.buffer.num_transitions - self._last_retrain_buffer_size
        if (self.buffer.num_transitions >= MINIBATCH_SIZE
                and new_transitions >= DYNAMICS_RETRAIN_INTERVAL):
            self.dynamics.train(self.buffer, DYNAMICS_EPOCHS_ONLINE)
            self._last_retrain_buffer_size = self.buffer.num_transitions
            if len(self._calibration_obs) >= 30:
                self.scorer.calibrate(self._calibration_obs, self._calibration_distances)

    def receive_demo(self, demo):
        self._demos_received += 1
        for i in range(len(demo) - 1):
            obs, action = demo[i]
            next_obs, _ = demo[i + 1]
            self.buffer.add_demo_transition(obs, action, next_obs)
        if len(demo) > 0:
            final_obs, final_action = demo[-1]
            self.buffer.demo_observations.append(final_obs.copy())
            self.buffer.demo_actions.append(final_action.copy())

        if self._demos_received == NUM_DEMOS:
            self.bc_policy.train(self.buffer, POLICY_EPOCHS)
            self.dynamics.train(self.buffer, DYNAMICS_EPOCHS_INITIAL)
            self._last_retrain_buffer_size = self.buffer.num_transitions
            n = self.buffer.num_transitions
            if n >= 5:
                synth = np.linspace(1.8, 0.05, n).tolist()
                self.scorer.calibrate(self.buffer.observations, synth)


    def testing_action(self, obs):
        current_score = self.scorer.score(obs)

        if current_score > self._test_best_score + 1e-4:
            self._test_best_score = current_score
            self._test_steps_without_progress = 0
        else:
            self._test_steps_without_progress += 1

        if (self._test_steps_without_progress >= STUCK_PATIENCE
                and self.dynamics.is_trained
                and self.scorer.weights is not None):
            self._test_steps_without_progress = 0
            return self._get_gradient_planner().recover(obs)

        if not self.bc_policy.is_trained:
            return np.random.uniform(
                -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE,
                constants.ACTION_DIMENSION
            )

        bc_action = self.bc_policy.predict_action(obs)

        if self.dynamics.is_trained and self.scorer.weights is not None:
            predicted_next = self.dynamics.predict_next_obs(obs, bc_action)
            bc_improvement = self.scorer.score(predicted_next) - current_score

            if bc_improvement < BC_IMPROVEMENT_THRESHOLD:
                shooter_action = self._get_random_shooter().shoot(obs)
                shooter_next = self.dynamics.predict_next_obs(obs, shooter_action)
                shooter_improvement = self.scorer.score(shooter_next) - current_score

                if shooter_improvement > bc_improvement:
                    return shooter_action

                return self._get_gradient_planner().refine(obs, bc_action)

        return bc_action