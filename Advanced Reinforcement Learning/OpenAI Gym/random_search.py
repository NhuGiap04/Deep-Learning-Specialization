import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0


def play_one_episode(env, params):
    observation, info = env.reset()
    done = False
    t = 0

    while not done and t < 10000:
        t += 1
        action = get_action(observation, params)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    return t


def play_multiple_episodes(env, T, params):
    episode_lengths = np.empty(T)

    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)

    avg_length = episode_lengths.mean()
    print("Avg Length:", avg_length)
    return avg_length


def random_search(env):
    episode_lengths = []
    best = 0
    params = None

    for t in range(100):
        new_params = np.random.random(4) * 2 - 1
        avg_length = play_multiple_episodes(env, 100, new_params)
        episode_lengths.append(avg_length)

        if avg_length > best:
            params = new_params
            best = avg_length
    return episode_lengths, params


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    episode_lengths, params = random_search(env)
    plt.plot(episode_lengths)
    plt.show()
