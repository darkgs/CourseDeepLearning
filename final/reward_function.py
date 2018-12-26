
import numpy as np

track_out_count = 0
low_wheel_spin_count = 0

def init_reward_f():
    global track_out_count
    global low_wheel_spin_count

    track_out_count = 0
    low_wheel_spin_count = 0

def calcReward(obs):
    ## write your own reward
    global track_out_count
    global low_wheel_spin_count

    track = np.array(obs['track'])
    trackPos = np.array(obs['trackPos'])
    sp = np.array(obs['speedX'])
    damage = np.array(obs['damage'])
    rpm = np.array(obs['rpm'])
    wheelSpinVel = np.mean(np.array(obs['wheelSpinVel']))
    distFromStart = np.array(obs['distFromStart'])

    if trackPos < 0.0 or trackPos > 1.0:
        track_out_count += 1
    else:
        track_out_count = 0

    if wheelSpinVel < 15.0:
        low_wheel_spin_count += 1
    else:
        low_wheel_spin_count = 0

    reward = sp*np.cos(obs['angle']) - np.abs(sp*np.sin(obs['angle'])) \
             - sp * np.abs(obs['trackPos'])

#[
#[    reward -= min(1.0 * (track_out_count ** 2), 50.0)
#[
#[    if wheelSpinVel > 80.0:
#[        reward -= min((wheelSpinVel - 80.0) ** 2, 20.0)
#[
    reward += min(wheelSpinVel - 20.0, 30.0)

    if low_wheel_spin_count > 20:
        reward -= min((low_wheel_spin_count-15)*15, 100.0)
    elif low_wheel_spin_count > 15:
        reward -= min((low_wheel_spin_count-15)*12, 60.0)

    if 0.3 < trackPos and trackPos < 0.7:
        reward += min(abs(trackPos-0.5)*50.0, 25.0)

    if trackPos < 0.0:
        reward -= min((0.0-trackPos) * 20.0, 100.0)
    elif trackPos > 1.0:
        reward -= min((trackPos-1.0) * 20.0, 100.0)

    reward -= min(damage / 20.0, 100.0)

#    print(reward)

    return reward

