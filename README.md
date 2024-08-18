# Calyber Q-Learning Decision-making

This repository contains the implementation of a decision-making model based on reinforcement learning, designed to address the challenges in pricing and matching in a double-channel environment. The model and associated scripts are used to simulate, train, and evaluate a neural network-based decision system.

## Structure

- **data/**
  - Contains the area information, user behavior data used to simulate the environment, and validation user behavior data for model performance evaluation.
  
- **dump/**
  - Includes an example of detailed simulated customer information and behavior, estimated action rewards, and model prediction rewards during a single episode of model training.
  
- **pictures/**
  - Contains examples showing the evolution of rewards during model training on an episode basis.

- **253 Calyber Report Group Shang_chun_shan.pdf**
  - The project report detailing the initial findings, model structure, and performance analysis.

- **calyber_decison.py**
  - The neural network structure used in the model.

- **calyber_env.py**
  - The simulation mechanism and reward design for the environment.

- **calyber_rl_torch.py**
  - The Q-learning training script used to train the model.

- **calyber_simulation.py**
  - Script to evaluate the performance of a trained model.

- **rider.py**
  - Defines the rider class used during simulation.

- **shang_chun_shan.py**
  - Contains helper functions and demonstrates how to use the trained model to make pricing and matching decisions.

## Dependencies

This project requires the following dependency:

- **PyTorch**: For building and training the neural network models.

You can install PyTorch using pip:

```bash
pip install torch torchvision
```

## Additional Notes

### Single Channel Drawbacks

The initial approach in the project was focused on a single channel network. However, this method encountered significant challenges:

- **Indistinguishable Rewards**: The model struggled to differentiate between matching rewards and pricing rewards, leading to suboptimal decision-making.
- **Matching Dilemma**: The single-channel network was unable to effectively address the matching dilemma described in the report, where balancing supply and demand while optimizing pricing posed a significant challenge.

### Transition to Double Channel

After recognizing these issues, the model was modified to utilize a double-channel approach. This adjustment involved:

- **Separate Channels for Rewards**: By separating the reward mechanisms into two distinct channels, the model could more effectively learn and distinguish between the different types of rewards.
- **Improved Model Performance**: The double-channel structure, combined with refined reward mechanisms, enabled the model to overcome the previous matching dilemma. The final model demonstrated strong performance on the validation set, showing a marked improvement in both pricing and matching decisions.

This modification highlights the importance of flexibility and adaptation in model design, and the effectiveness of multiple channels in complex decision-making environments.

## Usage

To simulate the environment, train the model, and evaluate its performance, follow these steps:

1. Run `calyber_env.py` to set up the simulation environment.
2. Train the model using `calyber_rl_torch.py`.
3. Evaluate the trained model's performance using `calyber_simulation.py`.
4. Use the helper functions in `shang_chun_shan.py` to apply the model to real-world decision-making scenarios.

For more detailed information, refer to the report included in this repository.
