import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from DDP import *
import pdb


class CartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 250
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        action_high = np.array([self.force_mag * 100])

        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(-high, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _simulateReward(self, state):
        x, x_dot, theta, theta_dot = state
        #reward = - (99 * abs(theta - 0) + 1 * abs(x - 0))
        reward = - 100 * abs(theta - 0)
        return reward

    def _simulateStep(self, state, action):
        x, x_dot, theta, theta_dot = state

        force = action[0]
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot + xacc * self.tau**2 / 2
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot + thetaacc * self.tau**2 / 2
        theta_dot = theta_dot + self.tau * thetaacc

        next_state = np.array([x, x_dot, theta, theta_dot])

        reward = self._simulateReward(next_state)

        done = False

        return next_state, reward, done, {}

    def simulateStep(self, state, action):
        return self._simulateStep(state, action)

    def simulateReward(self, state):
        return self._simulateReward(state)

    def _step(self, action):
        state = self.state

        next_state, reward, done, _info = self._simulateStep(state, action)

        self.state = next_state

        return self.state, reward, done, _info

    def _reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')


def solve_actions(env, init_state, simulation_timesteps):
    SIMULATION_TIMESTEPS = simulation_timesteps
    INIT_ACTION = np.array([0.])
    DERIVATIVE_ACCURACY = 0.001
    LEARNING_RATE = 0.025

    # nodes
    nodes = []
    for i in range(SIMULATION_TIMESTEPS + 1):
        node = DynamicsSystemNode('timestep_' + str(i))
    
        if (i < SIMULATION_TIMESTEPS):
            node.next_primitive_name = 'move_' + str(i)
            node.ssa.updateActionElement('action_' + str(i), INIT_ACTION.copy())
        
        if (i > 0):
          node.prev_primitive_name = 'move_' + str(i-1)

        nodes.append(node)

    nodes[0].ssa.updateStateElement('state', init_state)

    # dynamics models
    dynamics_models = []
    for i in range(SIMULATION_TIMESTEPS):
        dynamics = DynamicsSystemEdgeDynamicsModel()

        def make_dynamics_func (i):
            def dynamics_func (ssa):
                next_ssa = ssa.copy()
                state = ssa.retrive('state')
                action = ssa.retrive('action_' + str(i))
                next_state, reward, done, _info = env.simulateStep(state, action)
                next_ssa.updateStateElement('state', next_state)
                ns = nodes
                return next_ssa
            return dynamics_func
        dynamics.dynamics_func = make_dynamics_func(i)

        def make_dynamics_dfunc (i):
            def dynamics_dfunc (ssa):
                state = ssa.retrive('state')
                action = ssa.retrive('action_' + str(i))
                next_state, reward, done, _info = env.simulateStep(state, action)
                shift_next_state, shift_reward, shift_done, _shift_info = env.simulateStep(state, action + DERIVATIVE_ACCURACY)
                d_state_action = np.array([(shift_next_state - next_state) / DERIVATIVE_ACCURACY])
                d = {}
                d['state'] = {}
                d['state']['action_' + str(i)] = d_state_action
                return d
            return dynamics_dfunc
        dynamics.dynamics_dfunc = make_dynamics_dfunc(i)

        def make_reward_func (i):
            def reward_func (next_ssa):
                next_state = next_ssa.retrive('state')
                reward = env.simulateReward(next_state)
                return reward
            return reward_func
        dynamics.reward_func = make_reward_func(i)

        def make_reward_dfunc (i):
            def reward_dfunc (next_ssa):
                next_state = next_ssa.retrive('state')
                reward = env.simulateReward(next_state)
                d_reward_state = np.zeros((next_state.shape[0]))
                for i in range(next_state.shape[0]):
                    shift_next_state_i = next_state.copy()
                    shift_next_state_i[i] += DERIVATIVE_ACCURACY
                    shift_reward_i = env.simulateReward(shift_next_state_i)
                    d_reward_state[i] = (shift_reward_i - reward) / DERIVATIVE_ACCURACY
                d = {}
                d['state'] = d_reward_state
                return d
            return reward_dfunc
        dynamics.reward_dfunc = make_reward_dfunc(i)

        dynamics_models.append(dynamics)

    # dynamics system
    dynamics_system = LinearDynamicsSystem()
    for i in range(len(nodes)):
        dynamics_system.updateNode('timestep_' + str(i), nodes[i])
    dynamics_system.updateRootNodeName('timestep_0')
    for i in range(len(dynamics_models)):
        dynamics_system.updateDynamics(
            'move_' + str(i), \
            'timestep_' + str(i), dynamics_models[i], 'timestep_' + str(i+1), \
            alpha=LEARNING_RATE)

    # execute
    def optimizeActions_shouldStop (indicators):
        if indicators['rounds'] > 120:
            return True
        if indicators['value'] > -50.0:
            return True
        return False
    dynamics_system.optimizeActions(optimizeActions_shouldStop, verbose=1)

    # export action sequence
    actions = []
    for i in range(SIMULATION_TIMESTEPS):
        node = nodes[i]
        actions.append(node.ssa.retrive('action_' + str(i)))
    return actions


if __name__ == '__main__':
    env = CartPoleEnv()
    init_state = env.reset()

    simulation_timesteps = 50
    actions = solve_actions(env, init_state, simulation_timesteps)
    #print(actions)

    print('init_state: {}'.format(init_state))
    env.render()
    timestep = 0
    for timestep in range(simulation_timesteps):
        action = actions[timestep]
        state, reward, done, _ = env.step(action)
        print('action: {}, reward: {}, state: {}'.format(action, np.round(reward, 2), np.round(state, 2)))
        env.render()
        if done:
            break


