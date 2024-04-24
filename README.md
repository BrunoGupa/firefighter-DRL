# Bulldozer RL Training with Stable Baselines

In this repository, we train instances of the Bulldozer reinforcement learning environment using Stable Baselines. Our work focuses on fixing bugs in the original Bulldozer environment repository. After experimenting with various reward functions, we found that the episodic function works best for solving small instances.

## Results

Below are some GIFs showing instances that have been successfully solved:

1. ![GIF 1](# Bulldozer RL Training with Stable Baselines

In this repository, we train instances of the Bulldozer reinforcement learning environment using Stable Baselines. Our work focuses on fixing bugs in the original Bulldozer environment repository. After experimenting with various reward functions, we found that the episodic function works best for solving small instances.

## Results

Below are GIFs showcasing instances with different velocities of the bulldozer containment and fire spread. When these hyperparameters are fixed, an agent is trained using the Proximal Policy Optimization (PPO) Deep Reinforcement Learning Algorithm to solve the instance by creating a containment strategy:

1. ![GIF 1](Solver_Firefighter/avatar-gifs/avatar-fire-master-1-b-0.gif)
2. ![GIF 2](Solver_Firefighter/avatar-gifs/avatar-fire-master-2-a-0.gif)
3. ![GIF 3](Solver_Firefighter/avatar-gifs/avatar-fire-master-6-b-1.gif)
   <!-- Add more GIFs as necessary -->

## Links

- [Moving Firefighter Problem (MFP)](https://www.mdpi.com/2038212)
- [MFP Repository](https://github.com/BrunoGupa/MovingFirefighterProblem)
- [Bulldozer Benchmark Repository](https://github.com/elbecerrasoto/gym-cellular-automata)
For translating instances from the Moving Firefighter Problem (MFP) into the Bulldozer RL environment, please refer to our instance translator available [here](https://github.com/BrunoGupa/instance_translator).
