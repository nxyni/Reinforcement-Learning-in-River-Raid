# Reinforcement-Learning-in-River-Raid
In this project, we apply reinforcment learning to the game named 'river raid' on Open AI platform.

# Game Background
The game, River Raid, is designed by Carol Shaw in 1982. It is a vertically scrolling shooter game with a top-down perspective. The player control a fighter jet over the river to raid the enemy lines. The jet can be controlled to move right or left to avoid getting hit or catch fuel bag. There is also a limitation on fuel, such that the jet needs to catch the fuel bag to maintain the game. Colliding on the surrounding walls or enemy’s jet, getting hit by enemy and running out of fuel will end the game. If the jet across the bridge or hit enemy’s jet, the points will increase.

# Methods
In the application, we firstly use Deep Q-Learning to this game. Since there exist some shortcomings of this algorithm, Double DQN and Deep Recurrent Q-Learning are applied for taking both the efficiency and performance into consideration. Although the training process is not adequate due to the limited computation resources, the performance of these algorithms is macroscopic, which means that we have successfully applied reinforcement learning in the interesting game area.
