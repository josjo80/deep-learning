"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

logger = logging.getLogger(__name__)

class AblationEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        
        self.dt = 10e-3 #time interval that agent has to calculate impedance, T, etc. and make decision
        self.N = .75 #
        #Tissue model
        #self.length = 7e-3 #length of tissue in m
        #self.width = 3e-3 #width of tissue in m
        #self.height = 3e-3 #height of tissue in m
        self.density = 1e3 #Density of water in kg/m^3
        #self.mass = self.length*self.width*self.height*self.density
        self.A = 5.6e63 #5.6e63 (old 7.39e39) 1/s for artery per "Rate process model for arterial tissue thermal damage: implications on vessel photocoagulation." by Agah R, Pearce JA, Welch AJ
        self.dE = 4.3e5 #4.3e5 (old 2.577e5) J/mol for artery per "Rate process model for arterial tissue thermal damage: implications on vessel photocoagulation." by Agah R, Pearce JA, Welch AJ
        self.R = 8.3144598 #Universal gas constant, J/K*mol
        #C is specific heat capacity
        #C is 4000 J/kgK per "Modeling and numerical simulation of bioheat transfer and biomechanics in soft tissue", Wensheng Shen
        #C is 3600 J/kgC per "Considerations for Thermal Injury Analysis for RF Ablation Devices" Isaac A Chang
        self.C = 3600
        #Initialized below in reset_(self)
        #self.n = 0 #step index used with time interval
        #self.T = 37. #initial temp of tissue, Body temp
        #self.D = 0. #initial damage to tissue
        self.P_mag = 1. # magnitude of power deposited to tissue in W, Use 2.5W for Q-learning-ablation.ipynb and 1W for Q-learning-ablation2

        
        # State limits: Time and ablation damage limits
        self.time_limit = 12.0
        self.damage_limit = 0.8
        self.low = 0.

        # High and low observation limits
        high = np.array([
            self.time_limit * 2,
            np.finfo(np.float32).max,
            self.damage_limit * 2,
            np.finfo(np.float32).max])

        low = np.array([self.low,np.finfo(np.float32).max,
            self.low,np.finfo(np.float32).max])

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low, high)

        self._seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        P, T, D, Z, n = state
        if action == 2:
            P += self.P_mag
        elif action == 1:
            P = P
        elif P > 0:
            P += -self.P_mag

        #P = self.P_mag if action==1 else -self.P_mag

        #Tissue Damage Solver - Integrates the damage to tissue over time for a given temperature (in Kelvin, hence adding 273.15 to T below).
        #Assumes dt timestep.
        D += (self.A*math.exp(-self.dE/(self.R*(T + 273.15)))*self.dt)

        #Thermal Solver - Assumes a mass of water, power is held constant as it is delivered to the mass, no phase change
        #and no perfusion of blood to the "tissue" (i.e. water mass).
        #Constant definitions
        T = (1/(self.mass*self.C))*((P*self.dt) + self.mass*self.C*T) + 1*D #Added in dependancy on D to mimic increase in T as tissue is cooked.
        if T > 100:
            T = 100 #T in Celcius.
        
        #ImpedanceSolver - Assumes saline solution that matches tissue conductance.  Takes in tissue temp, solution normalization,
        #and height/thickness of tissue.  Returns conductance, sig, and impedance, Z based on temperature.
        #sigma is based on Eq 6 from "Considerations for Thermal Injury Analysis for RF Ablation Devices" Isaac A Chang
        delta = 25.0 - T #delta in Celcius
        sig25N = self.N*(10.394 - 2.3776*self.N + 0.68258*self.N**2 - 9.13538*self.N**3 + 1.0086e-2*self.N**4)
        sig = sig25N*(1.0 - 1.962e-2*delta + 8.08e-5*delta**2 - self.N*delta*(3.020e-5 + 3.922e-5*delta + self.N*(1.721e-5 - 6.584e-6*delta)))
        Z = 1/(sig*self.height) + (10*D)**2 #Adding dependance on D damage, to mimic the increase in Z as tissue is cooked.
        
        n += 1

        #Update the states
        self.state = (P, T,D,Z,n)
        done =  n*self.dt > self.time_limit \
                or D > self.damage_limit
        done = bool(done)

        if not done:
            reward = -1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -10.*D
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        self.state = np.array([0., 37., 0., 65., 0])  #[5., self.np_random.uniform(low=35., high=39., size=(1,)), 0., self.np_random.uniform(low=55, high=75, size=(1,)), 0]
        self.length = self.np_random.uniform(low=5e-3, high=7e-3, size=(1,)) #7e-3 #length of tissue in m
        self.width = 3e-3 #width of tissue in m
        self.height = self.np_random.uniform(low=1e-3, high=3e-3, size=(1,)) #3e-3 #height of tissue in m
        self.mass = self.length*self.width*self.height*self.density
        self.steps_beyond_done = None
        return np.array(self.state), self.length, self.width, self.height, self.mass, self.steps_beyond_done

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.time_limit*2
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
