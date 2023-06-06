import gymnasium as gym

from gymnasium.utils.play import PlayPlot, play

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
       return [rew,]

keys_to_action = {
    'a': 0,
    'd': 1
}

plotter = PlayPlot(callback, 150, ["reward"])

play(gym.make("CartPole-v1", render_mode="rgb_array"), 
     callback=plotter.callback, 
     keys_to_action=keys_to_action)
