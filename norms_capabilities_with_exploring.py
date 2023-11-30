import random
import matplotlib.pyplot as plt

# Parameters
V_H = 1  # Value for high
V_L = 0  # Value for low
xi = 0.47 # Fidelity - (Probability of receiving the opposite signal)
C_L_0 = 0.5  # Initial capability for Left
C_R_0 = 0.5  # Initial capability for Right
delta = 2    # Difficulty in learning capability
exploration_rate = 0.05


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
C_L = C_L_0 # Left capability
C_R = C_R_0 # right capability
cumulative_profit= 0 # profit from decisions

# Lists to track beliefs over time
pL_history = []
pR_history = []
profit_history = []
cr_history = []
cl_history = []

# Function to simulate one decision period
def simulate_decision():
    global nL_s, nL, nR_s, nR,C_L, C_R, cumulative_profit

    # Agent's belief about success probabilities
    pL = nL_s / nL if nL > 0 else 0.5  # Avoid division by zero
    pR = nR_s / nR if nR > 0 else 0.5

    # Store the probabilities
    pL_history.append(pL)
    pR_history.append(pR)

    # Calculate probabilities for decision making
    P_L_greater_R = (nL_s + nR - nR_s) / (nR + nL) if (nR + nL) > 0 else 0.5
    P_R_greater_L = (nR_s + nL - nL_s) / (nR + nL) if (nR + nL) > 0 else 0.5

    # Agent makes a decision with a chance of exploration
    if random.random() < exploration_rate:
        decision = 'R' if random.random() < 0.5 else 'L'
    else:
        decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

    # Determine actual outcome and update capabilities
    if decision == 'L':
        nL += 1
        C_L = C_L_0 - delta / (delta + nL)
        profit = V_H - C_L if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location) else V_L - C_L
        if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location):
            nL_s += 1
    else:
        nR += 1
        C_R = C_R_0 - delta / (delta + nR)
        profit = V_H - C_R if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location) else V_L - C_R
        if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location):
            nR_s += 1

    cumulative_profit += profit
    profit_history.append(cumulative_profit)

    cl_history.append(C_L)
    cr_history.append(C_R)



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

# Plotting the Capability over time
plt.plot(cl_history, label='Cost in Left (pL)')
plt.plot(cr_history, label='Cost in Right (pR)')
plt.xlabel('Time Periods')
plt.ylabel('Cost of Decision')
plt.title('Agent Capability Over Time with Fidelity = {}'.format(xi))
plt.legend()
plt.grid()
plt.show()


# Plotting the profit over time
plt.plot(profit_history, label='Cumulative Profit')
plt.xlabel('Time Periods')
plt.ylabel('Profit')
plt.title('Agent Profit Over Time with Fidelity = {}'.format(xi))
plt.legend()
plt.grid()
plt.show()
