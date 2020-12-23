import gym
import numpy as np
import random
from gym import error, spaces, utils
from gym.utils import seeding

class CacheEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.task_n = 10  # the number of total tasks
    self.task = list(range(1, self.task_n + 1))  # the total tasks set

    self.user_n = 5  # the number of users
    self.each_user_task_n = 1  # the number of requested task by each user
    # if self.each_user_task_n == 1:
    #   self.users_tasks = np.zeros(self.user_n)  # the set of all users' tasks
    # else:
    #   self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))  # the set of all users' tasks
    self.users_tasks = np.zeros(shape=(self.user_n, self.each_user_task_n))


    self.edge_n = 3  # the number of agent(edge server)
    self.each_edge_cache_n = 3  # the number of caching tasks by each edge server
    # state : the set of all edge servers' caching tasks
    self.edges_caching_task = np.zeros(shape=(self.edge_n, self.each_edge_cache_n))
    # for i in range(self.edge_n):  # initial edge caching tasks randomly every episode
    #   self.edges_caching_task[i] = random.sample(self.task, self.each_edge_cache_n)
    self.state = self.edges_caching_task
    # action
    self.action_space = [spaces.Discrete(np.power(2, self.user_n)) for i in range(self.edge_n)]
    # observation
    self.observation_space = [spaces.Discrete(self.each_edge_cache_n) for i in range(self.edge_n)]

    # channel gain from edge server to user ！1.5不准确，可更换
    self.h_eu = np.zeros(shape=(self.edge_n, self.user_n))
    for i in range(self.edge_n):
      self.h_eu[i] = 1.5 * abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))

    # channel gain from cloud server to user
    self.h_cu = abs(1 / np.sqrt(2) * (np.random.randn(self.user_n) + 1j * np.random.randn(self.user_n)))

    # power bisection
    self.p_total = 19.953  # total power is 19.953w, namely 43dbm
    self.p = self.p_total/self.user_n
    # channel bandwidth
    self.bandwidth = 4500000  # 4.5MHZ

    self.seed()  # TODO control some features change randomly
    self.viewer = None  # whether open the visual tool
    self.steps_beyond_done = None  # current step = 0

  # Gaussian noise
  def compute_noise(self, NUM_Channel):
    ThermalNoisedBm = -174  # -174dBm/Hz
    var_noise = 10 ** ((ThermalNoisedBm - 30) / 10) * self.bandwidth / (
      NUM_Channel)  # envoriment noise is 1.9905e-15
    return var_noise

  '''
  compute signal to inference plus noise ratio
  h: channel gain from edge server e to all users with the shape of (1, self.user_n)
  x: serve success from edge server e to all users with the shape of (1, self.user_n)
  sinr: SINR from e to all users
  '''
  def compute_SINR(self, h, x):
    sinr = np.zeros(self.user_n)
    sum_hxp = 0
    for i in range(self.user_n):
      hxp = np.power(abs(h[i]*x[i]*self.p), 2)
      sum_hxp += hxp
    for i in range(self.user_n):
      sinr[i] = np.power(abs(h[i]*x[i]*self.p),2)/(sum_hxp - np.power(abs(h[i]*x[i]*self.p),2) + self.compute_noise(self.user_n))
    return sinr
  '''
  compute downlink rate
  sinr: SINR from an edge server to all users with the shape of (1, self.user_n)
  '''
  def compute_Rate(self, sinr):
    rate = np.zeros(self.user_n)
    for i in range(self.user_n):
      rate[i] = self.bandwidth * np.log2(1+sinr[i])
    return rate

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  # sample randomly from task_set
  def random_sample(self, task_set, num):
    sample_task = random.sample(task_set, num)
    return sample_task
  '''
  sample a task according to Zipf distribution
  task_set: the set we sample from
  num: the number of sampled tasks
  '''
  def Zipf_sample(self, task_set, num):
    # probability distribution
    task_num = len(task_set)
    p = np.zeros(task_num)
    for i in range(task_num):
      p[i] = int(0.1/(i+1)*100000)
    sampled_task = []
    for j in range(num):
      # sample & return index
      start = 0
      index = 0
      randnum = random.randint(1, sum(p))
      for index, scope in enumerate(p):
        start += scope
        if randnum <= start:
          break
      sampled_task.append(index)
    return sampled_task

  def step(self, action):
    edges_caching_task = self.state
    users_tasks = self.users_tasks
    serve_success = np.zeros(shape=(self.edge_n, self.user_n))
    # TODO action 这里怎么写？？multi-agent 怎么写action?
    # now set action as a matrix with the shape of edge_n * user_n
    ac = np.zeros(self.user_n)
    for i in range(self.user_n):
      for j in range(self.edge_n):
        ac = action[j]
        if (users_tasks[i] in edges_caching_task[j]) and (ac[i] == 1):
          serve_success[j][i] = 1
        elif users_tasks[i] not in edges_caching_task[j]:
          # choose new task according to zipf
          new_task = self.Zipf_sample(self.task, 1)  # zipf: sample a new task according to Zipf
          # choose new task according to user previous task
          #new_task = users_tasks[i]
          temp = np.delete(edges_caching_task[j], 0)  # delete the first one
          edges_caching_task[j] = np.append(temp, new_task)  # add the new one at the end
    self.state = edges_caching_task

    # cloud server serves users successfully
    serve_sucs_cu = np.zeros(self.user_n)
    for i in range(self.user_n):
      if 1 in serve_success[:, i]:
        serve_sucs_cu[i] = 0
      else:
        serve_sucs_cu[i] = 1
    # done
    done = 1

    # initial
    sinr_eu = np.zeros(shape=(self.edge_n, self.user_n))  # SINR from edge server to user
    R_eu = np.zeros(shape=(self.edge_n, self.user_n))  # downlink rate from edge server to user
    sinr_cu = np.zeros(self.user_n)  # SINR from cloud server to user
    R_cu = np.zeros(self.user_n)  # downlink rate from cloud server to user

    # compute
    for j in range(self.edge_n):
      sinr_eu[j] = self.compute_SINR(self.h_eu[j], serve_success[j])
      R_eu[j] = self.compute_Rate(sinr_eu[j])
    sinr_cu = self.compute_SINR(self.h_cu, serve_sucs_cu)
    R_cu = self.compute_Rate(sinr_cu)

    sinr = int(sum(sum(R_eu)) + sum(R_cu))
    # reward
    reward = sinr
    return self.state, reward, done, {}

  def reset(self):
    for i in range(self.user_n):  # initial users_tasks randomly
       self.users_tasks[i] = self.Zipf_sample(self.task, self.each_user_task_n)
    # update the state at the beginning of each episode
    for i in range(self.edge_n):
      self.edges_caching_task[i] = self.Zipf_sample(self.task, self.each_edge_cache_n)
    self.state = self.edges_caching_task
    self.steps_beyond_done = None  # set the current step as 0
    return np.array(self.state)  # return the initial state
  def render(self, mode='human'):
    ...
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None


