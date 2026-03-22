import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    V = np.zeros(n_states)
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)

    for episode in episodes:
        T = len(episode)
        G_list = np.zeros(T)

        # Backward pass: compute full return at each timestep
        G = 0
        for t in reversed(range(T)):
            state, reward = episode[t]
            G = gamma * G + reward
            G_list[t] = G

        # Forward pass: first-visit check on complete returns
        visited = set()
        for t in range(T):
            state, _ = episode[t]
            if state not in visited:
                returns_sum[state] += G_list[t]
                returns_count[state] += 1
                visited.add(state)
    # avoid divide-by-zero
    for state in range(n_states):
        if returns_count[state] > 0:
            V[state] = returns_sum[state] / returns_count[state]


    return V