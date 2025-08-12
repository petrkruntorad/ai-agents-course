import numpy as np
import gymnasium as gym
import setup

# Create environment
env = gym.make("Grid2DEnv-v0")

# Q-learning parameters
alpha = 0.2  # Learning rate
gamma = 0.5  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000  # Number of training episodes

# Q-table: 5x5 grid with 4 possible actions
q_table = np.zeros((5, 5, 4))


def choose_action(state):
    """Epsilon-greedy action selection"""
    row, col = map(int, state)  # Ensure indices are integers

    # ---- High epsilon (e.g. 0.8):

    # Agent explores a lot (80% random actions, 20% greedy/best).

    # Useful at the beginning when Q-table is mostly zeros/unexplored.

    # ---- Low epsilon (e.g. 0.05):

    # Agent mostly exploits (95% best-known actions, 5% random).

    # Useful later in training, so the agent uses what it has learned.

    # ---- Epsilon = 0:

    # Agent is 100% greedy (always exploits, never explores).

    # Risk: Might get stuck in a local optimum if it never tried other actions.

    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Random action

    return np.argmax(q_table[row, col])  # Greedy action


def train_q_learning():
    """Q-learning training loop (off-policy)"""
    for episode in range(episodes):
        state, _ = env.reset()
        state = state.astype(int)  # Convert to integer

        done = False

        while not done:
            action = choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = next_state.astype(int)  # Convert to integer

            row, col = state
            next_row, next_col = next_state

            # ----------------------------------------------
            # Q-learning update rule (off-policy)
            q_table[row, col, action] += alpha * (
                reward
                + gamma * np.max(q_table[next_row, next_col, :])
                - q_table[row, col, action]
            )

            state = next_state  # Move to next state
            # ----------------------------------------------

            done = terminated or truncated

        if episode % 100 == 0:
            print(f"Q-learning: Episode {episode} completed")

    np.save("q_learning_q_table.npy", q_table)
    print("Q-learning training complete. Q-table saved!")


if __name__ == "__main__":
    train_q_learning()
    env.close()
