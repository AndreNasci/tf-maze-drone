# MazeDrone Project with TF-Agents

![Static Badge](https://img.shields.io/badge/Project_Status:-Ongoing_(01/Jul/2025)-orange)

![Static Badge](https://img.shields.io/badge/TF_Agents-blue)
![Static Badge](https://img.shields.io/badge/Python-blue)
![Static Badge](https://img.shields.io/badge/Reinforcement_Learning-blue)
![Static Badge](https://img.shields.io/badge/DQN-blue)
![Static Badge](https://img.shields.io/badge/OpenAI_GYM-blue)
![Static Badge](https://img.shields.io/badge/Tiny_ML-blue)
![Static Badge](https://img.shields.io/badge/Curriculum_Learning-blue)
![Static Badge](https://img.shields.io/badge/Simulated_Environments-blue)

This repository contains the Python source code for an Artificial Intelligence (AI) agent trained with Reinforcement Learning (RL) techniques to assist autonomous navigation in unknown indoor environments.

The project aims to explore a viable alternative for indoor navigation, using simulated telemetry data and focusing on future application in embedded devices with limited computational resources and connectivity (TinyML).

### Key Features:
- **Simulated Environments:** Procedurally generated mazes using OpenAI Gym for agent training and evaluation. These mazes are designed to be without cycles, with a single solution, and fully connected.
- **RL Model (DQN):** Implementation of a Deep Q-Network (DQN) agent using the TensorFlow framework and TF-Agents to learn maze solving.
- **Phased Training (Curriculum Learning):** A training methodology segmented into phases, gradually increasing environment complexity to optimize agent learning and generalization.
- **Short-Term Memory:** Inclusion of a short-term memory mechanism to allow the agent to consider its last three states and actions, resulting in more precise and fluid navigation.
- **Results:** The agent demonstrated satisfactory capabilities in small-sized environments (3x3 grid), achieving an average completion rate of approximately 92% across all tested scenarios. It also showed low averages for collisions (2) and indecision (3.7) during evaluation.

This project is part of an Undergraduate Thesis (Trabalho Final de Graduação) for the Computer Engineering course at the Federal University of Itajubá (UNIFEI).

Evolution of my old personal project [MazeDrone](https://github.com/AndreNasci/MazeDrone). 

## Credits
This project uses a modified version of the maze generator [pymaze](https://github.com/jostbr/pymaze), by jostbr.
