import gym
import rsoccer_gym

# Using VSS Single Agent env
env = gym.make('VSSMA-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(1):
    done = False
    while not done:
        # Step using random actions
        action = env.action_space.sample()
        print('action', action)
        next_state, reward, done, _ = env.step(action)
        print('next_state', next_state)
        env.render()
    print(reward)
