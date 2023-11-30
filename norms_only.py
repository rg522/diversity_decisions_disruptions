import random
import matplotlib.pyplot as plt

# Parameters
V_H = 1  # Value for high
V_L = 0  # Value for low
xi = 0.48  # Fidelity - (Probability of receiving the opposite signal)


# Location of V_H (True for Right, False for Left)
#V_H_location = random.choice([True, False])  # Randomly choose for simulation
#or choose manually:
V_H_location = False #for left
#V_H_location = True #for right

# Initialize counters
nL_s = 0  # Successful Left trials
nL = 0    # Total Left trials
nR_s = 0  # Successful Right trials
nR = 0    # Total Right trials


# Lists to track beliefs over time
pL_history = []
pR_history = []

# Function to simulate one decision period
def simulate_decision():
    global nL_s, nL, nR_s, nR

    # Agent's belief about success probabilities
    pL = nL_s / nL if nL > 0 else 0.5  # Avoid division by zero
    pR = nR_s / nR if nR > 0 else 0.5

    # Store the probabilities
    pL_history.append(pL)
    pR_history.append(pR)

    # Calculate probabilities for decision making
    P_L_greater_R = (nL_s + nR - nR_s) / (nR + nL) if (nR + nL) > 0 else 0.5
    P_R_greater_L = (nR_s + nL - nL_s) / (nR + nL) if (nR + nL) > 0 else 0.5

    # Agent makes a decision
    decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

    # Determine actual outcome
    if decision == 'L':
        nL += 1
        if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location):
            nL_s += 1
    else:
        nR += 1
        if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location):
            nR_s += 1


# Simulation loop
num_periods = 100  # Number of periods to simulate
for _ in range(num_periods):
    simulate_decision()

# Results
print(f"Location of V_H: {'Right' if V_H_location else 'Left'}")
print(f"Total Left Trials: {nL}, Successful Left Trials: {nL_s}")
print(f"Total Right Trials: {nR}, Successful Right Trials: {nR_s}")

# Plotting the beliefs over time
plt.plot(pL_history, label='Belief in Left (pL)')
plt.plot(pR_history, label='Belief in Right (pR)')
plt.xlabel('Time Periods')
plt.ylabel('Belief Probability')
plt.title('Agent Belief Convergence Over Time with Fidelity = {}'.format(xi))
plt.legend()
plt.grid()
plt.show()
