import random
import matplotlib.pyplot as plt
import numpy as np

#set disruption type:
# 1 for quadrant 1 (change to fidelity only)  [[Insider Better]]
# 2 for quadrant 2 (change to fidelity and Priority Shift) [[Outsider Better]]
# 3 for quadrant 3 (change to fidelity and NOPE) [[Insider Better]]
# 4 for quadrant 4 (change to fidelity, PSE, NOPE)  [[Outsider Better]
d_type = 2

# Location of V_H (True for Right, False for Left)
# V_H_location = random.choice([True, False])  # Randomly choose for simulation
# or choose manually:
V_H_location = False # for left
# V_H_location = True # for right
V_H_location_base = V_H_location # for use in reset

#number of periods:
big_t = 5000
t_D = 1000 #disruption at
num_simulations = 10

# Learning Parameters
V_H = 1  # Value for high
V_L = 0  # Value for low
xi = 0.0 # Fidelity - (Probability of receiving the opposite signal)
xi_base = xi
C_L_0 = 1  # Initial capability for Left
C_R_0 = 1  # Initial capability for Right
delta = 20   # Difficulty in learning capability
delta_in = 100  # Differential learning curve for insider and outsider
delta_out = 1000

#define the global variables
V_H_new= V_L_new = 0
reset_cap_left = reset_cap_right = reset_location = False

def set_d_type(d_type):
    # set the disruption type
    global t_D, V_H_new, V_L_new, V_H_location, reset_cap_left, reset_cap_right, reset_location
    V_H_new = 1  # How does the underlying value change?
    V_L_new = 0

    if d_type == 1:
        reset_location = False
        reset_cap_left = False  # Boolean for capability effect
        reset_cap_right = False

    elif d_type == 2:
        reset_location = True
        reset_cap_left = False  # Boolean for capability effect
        reset_cap_right = False

    elif d_type == 3:
        reset_location = False
        reset_cap_left = True  # Boolean for capability effect
        reset_cap_right = True

    elif d_type == 4:
        reset_location = True
        reset_cap_left = True  # Boolean for capability effect
        reset_cap_right = True

    print("Disruption set to Quadrant {}".format(d_type))

    return

set_d_type(d_type)

# Initialize counters for insider
nL_s = 0  # Successful Left trials
nL_belief = 0 # counter for the number of times left chosen
nL_capability = 0 # Total Left trials
nR_s = 0  # Successful Right trials
nR_belief = 0 # counter for the number of times right chosen
nR_capability = 0# Total Right trials
C_L = C_L_0 # Left capability
C_R = C_R_0 # right capability
cumulative_profit= 0 # profit from decisions

# Initialize variables for the outsider
nL_s_outsider = 0
nL_belief_outsider = 0
nL_capability_outsider = 0
nR_s_outsider = 0
nR_belief_outsider = 0
nR_capability_outsider = 0
C_L_outsider = C_L_0
C_R_outsider = C_R_0
cumulative_profit_outsider = 0

# Lists to track beliefs over time
pL_history = []
pR_history = []
profit_history = []
cr_history = []
cl_history = []
time_insider = []

pL_history_outsider = []
pR_history_outsider = []
profit_history_outsider = []
cr_history_outsider = []
cl_history_outsider = []
time_outsider = []

# Function to simulate one decision period
def simulate_decision_insider_randomizer(t):
    global nL_s, nL_belief, nR_s, nR_belief, nL_capability, nR_capability, C_L, C_R, cumulative_profit

    # Agent's belief about success probabilities (using belief counters)
    pL = nL_s / nL_belief if nL_belief > 0 else 0.5  # Avoid division by zero (assume no preference when n = 0)
    pR = nR_s / nR_belief if nR_belief > 0 else 0.5

    # Store the probabilities
    pL_history.append(pL)
    pR_history.append(pR)

    # Calculate probabilities for decision making
    P_L_greater_R = (nL_s + nR_belief - nR_s) / (nR_belief + nL_belief) if (nR_belief + nL_belief) > 0 else 0.5
    P_R_greater_L = (nR_s + nL_belief - nL_s) / (nR_belief + nL_belief) if (nR_belief + nL_belief) > 0 else 0.5

    # Agent makes a decision
    import random
    threshold = random.random()
    decision = 'L' if P_L_greater_R >  threshold else 'R'
    #decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

    #Capability
    C_L = C_L_0 - delta_in*C_L_0 / (delta_in + nL_capability)
    C_R = C_R_0 - delta_in*C_R_0 / (delta_in + nR_capability)

    # Determine actual outcome, update capabilities and profit
    if decision == 'L':
        nL_belief += 1
        nL_capability += 1
        profit = V_H - (1/(C_L+1)) if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location) else V_L - (1/(C_L+1))
        if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location):
            nL_s += 1
    else:
        nR_belief += 1
        nR_capability += 1
        profit = V_H - (1/(C_R+1)) if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location) else V_L - (1/(C_R+1))
        if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location):
            nR_s += 1


    cumulative_profit += profit
    profit_history.append(cumulative_profit)


    time_insider.append(t)

    # Fix initiation issues
    C_L = 0 if nL_capability == 0 else C_L
    C_R = 0 if nR_capability == 0 else C_R


    cl_history.append(C_L)
    cr_history.append(C_R)

def simulate_decision_insider(t):
    global nL_s, nL_belief, nR_s, nR_belief, nL_capability, nR_capability, C_L, C_R, cumulative_profit

    # Agent's belief about success probabilities (using belief counters)
    pL = nL_s / nL_belief if nL_belief > 0 else 0.5  # Avoid division by zero (assume no preference when n = 0)
    pR = nR_s / nR_belief if nR_belief > 0 else 0.5

    # Store the probabilities
    pL_history.append(pL)
    pR_history.append(pR)

    # Calculate probabilities for decision making
    P_L_greater_R = (nL_s + nR_belief - nR_s) / (nR_belief + nL_belief) if (nR_belief + nL_belief) > 0 else 0.5
    P_R_greater_L = (nR_s + nL_belief - nL_s) / (nR_belief + nL_belief) if (nR_belief + nL_belief) > 0 else 0.5

    # Agent makes a decision
    decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

    #Capability
    C_L = C_L_0 - delta_in*C_L_0 / (delta_in + nL_capability)
    C_R = C_R_0 - delta_in*C_R_0 / (delta_in + nR_capability)

    # Determine actual outcome, update capabilities and profit
    if decision == 'L':
        nL_belief += 1
        nL_capability += 1
        profit = V_H - (1/(C_L+1)) if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location) else V_L - (1/(C_L+1))
        if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location):
            nL_s += 1
    else:
        nR_belief += 1
        nR_capability += 1
        profit = V_H - (1/(C_R+1)) if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location) else V_L - (1/(C_R+1))
        if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location):
            nR_s += 1


    cumulative_profit += profit
    profit_history.append(cumulative_profit)


    time_insider.append(t)

    # Fix initiation issues
    C_L = 0 if nL_capability == 0 else C_L
    C_R = 0 if nR_capability == 0 else C_R


    cl_history.append(C_L)
    cr_history.append(C_R)


def simulate_decision_outsider_randomizer(t):
    global nL_s_outsider, nL_belief_outsider, nR_s_outsider, nR_belief_outsider, nL_capability_outsider, nR_capability_outsider, C_L_outsider, C_R_outsider, cumulative_profit_outsider

    # Agent's belief about success probabilities (using belief counters)
    pL = nL_s_outsider / nL_belief_outsider if nL_belief_outsider > 0 else 0.5  # Avoid division by zero
    pR = nR_s_outsider / nR_belief_outsider if nR_belief_outsider > 0 else 0.5

    # Store the probabilities
    pL_history_outsider.append(pL)
    pR_history_outsider.append(pR)

    # Calculate probabilities for decision making
    P_L_greater_R = (nL_s_outsider + nR_belief_outsider - nR_s_outsider) / (nR_belief_outsider + nL_belief_outsider) if (nR_belief_outsider + nL_belief_outsider) > 0 else 0.5
    P_R_greater_L = (nR_s_outsider + nL_belief_outsider - nL_s_outsider) / (nR_belief_outsider + nL_belief_outsider) if (nR_belief_outsider + nL_belief_outsider) > 0 else 0.5

    # Agent makes a decision
    threshold =  random.random()
    decision = 'L' if P_L_greater_R > threshold else 'R'

    #Capability
    C_L_outsider = C_L_0 - delta_out*C_L_0 / (delta_out + nL_capability_outsider)
    C_R_outsider = C_R_0 - delta_out*C_L_0 / (delta_out + nR_capability_outsider)

    # Determine actual outcome, update capabilities and profit
    if decision == 'L':
        nL_belief_outsider += 1
        nL_capability_outsider += 1
        profit = V_H - (1/(C_L_outsider+1)) if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location) else V_L - (1/(C_L_outsider+1))
        if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location):
            nL_s_outsider += 1
    else:

        nR_belief_outsider += 1
        nR_capability_outsider += 1
        profit = V_H - (1/(C_R_outsider+1)) if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location) else V_L - (1/(C_R_outsider+1))
        if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location):
            nR_s_outsider += 1


    cumulative_profit_outsider += profit
    profit_history_outsider.append(cumulative_profit_outsider)

    time = t
    time_outsider.append(time)

    cl_history_outsider.append(C_L_outsider)
    cr_history_outsider.append(C_R_outsider)

def simulate_decision_outsider(t):
    global nL_s_outsider, nL_belief_outsider, nR_s_outsider, nR_belief_outsider, nL_capability_outsider, nR_capability_outsider, C_L_outsider, C_R_outsider, cumulative_profit_outsider

    # Agent's belief about success probabilities (using belief counters)
    pL = nL_s_outsider / nL_belief_outsider if nL_belief_outsider > 0 else 0.5  # Avoid division by zero
    pR = nR_s_outsider / nR_belief_outsider if nR_belief_outsider > 0 else 0.5

    # Store the probabilities
    pL_history_outsider.append(pL)
    pR_history_outsider.append(pR)

    # Calculate probabilities for decision making
    P_L_greater_R = (nL_s_outsider + nR_belief_outsider - nR_s_outsider) / (nR_belief_outsider + nL_belief_outsider) if (nR_belief_outsider + nL_belief_outsider) > 0 else 0.5
    P_R_greater_L = (nR_s_outsider + nL_belief_outsider - nL_s_outsider) / (nR_belief_outsider + nL_belief_outsider) if (nR_belief_outsider + nL_belief_outsider) > 0 else 0.5

    # Agent makes a decision
    decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

    #Capability
    C_L_outsider = C_L_0 - delta_out*C_L_0 / (delta_out + nL_capability_outsider)
    C_R_outsider = C_R_0 - delta_out*C_L_0 / (delta_out + nR_capability_outsider)

    # Determine actual outcome, update capabilities and profit
    if decision == 'L':
        nL_belief_outsider += 1
        nL_capability_outsider += 1
        profit = V_H - (1/(C_L_outsider+1)) if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location) else V_L - (1/(C_L_outsider+1))
        if (random.random() > xi and not V_H_location) or (random.random() <= xi and V_H_location):
            nL_s_outsider += 1
    else:

        nR_belief_outsider += 1
        nR_capability_outsider += 1
        profit = V_H - (1/(C_R_outsider+1)) if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location) else V_L - (1/(C_R_outsider+1))
        if (random.random() > xi and V_H_location) or (random.random() <= xi and not V_H_location):
            nR_s_outsider += 1


    cumulative_profit_outsider += profit
    profit_history_outsider.append(cumulative_profit_outsider)

    time = t
    time_outsider.append(time)

    cl_history_outsider.append(C_L_outsider)
    cr_history_outsider.append(C_R_outsider)


def apply_disruption():
    global V_H, V_L, C_L, C_R, nL_capability, nR_capability, V_H_new, V_L_new, reset_location, V_H_location,xi

    xi = np.sqrt(xi) #update fidelity - proxy for making the problem harder after the disruption

    # Adjust V_H and V_L based on the disruption
    V_H, V_L = V_H_new, V_L_new

    # Reset capabilities if options change
    if reset_cap_left:
        nL_capability = 0
        C_L = C_L_0  # Reset Left capability to initial value

    if reset_cap_right:
        nR_capability = 0
        C_R = C_R_0  # Reset Right capability to initial value

    if reset_location:
        V_H_location = not V_H_location # Swap the value location

# New variables for aggregating results
average_profit_history = None
average_profit_history_outsider = []
average_belief_pL = []
average_belief_pR = []
average_belief_pL_outsider = []
average_belief_pR_outsider = []
average_capability_cL = []
average_capability_cR = []
average_capability_cL_outsider = []
average_capability_cR_outsider = []


for simulation in range(num_simulations):
    # Resetting variables for each simulation run
    nL_s = nL_belief = nL_capability = nR_s = nR_belief = nR_capability = 0
    nL_s_outsider = nL_belief_outsider = nL_capability_outsider = nR_s_outsider = nR_belief_outsider = nR_capability_outsider = 0
    C_L = C_L_0
    C_R = C_R_0
    cumulative_profit = 0
    cumulative_profit_outsider = 0
    pL_history = []
    pR_history = []
    pL_history_outsider = []
    pR_history_outsider = []
    profit_history = []
    profit_history_outsider = []
    cr_history = []
    cl_history = []
    cr_history_outsider = []
    cl_history_outsider = []
    time_insider = []
    xi = xi_base
    V_H_location = V_H_location_base
    #print(V_H_location)
    # Simulation loop
    for current_period in range(1, big_t):
        if current_period < t_D:
            simulate_decision_insider(current_period)

        else:
            if current_period == t_D:
                apply_disruption()
                #print(V_H_location)
            simulate_decision_insider(current_period)
            simulate_decision_outsider(current_period)

    # Aggregating results
    if average_profit_history is None:
        # profits:
        average_profit_history = np.array(profit_history)
        average_profit_history_outsider = np.array(profit_history_outsider)

        average_belief_pL = np.array(pL_history)
        average_belief_pR = np.array(pR_history)
        average_belief_pL_outsider = np.array(pL_history_outsider)
        average_belief_pR_outsider = np.array(pR_history_outsider)

        average_capability_cL = np.array(cl_history)
        average_capability_cR = np.array(cr_history)
        average_capability_cL_outsider = np.array(cl_history_outsider)
        average_capability_cR_outsider = np.array(cr_history_outsider)
    else:
        # profits:
        average_profit_history += np.array(profit_history)
        average_profit_history_outsider += np.array(profit_history_outsider)

        average_belief_pL += np.array(pL_history)
        average_belief_pR += np.array(pR_history)
        average_belief_pL_outsider += np.array(pL_history_outsider)
        average_belief_pR_outsider += np.array(pR_history_outsider)


        average_capability_cL += np.array(cl_history)
        average_capability_cR += np.array(cr_history)
        average_capability_cL_outsider += np.array(cl_history_outsider)
        average_capability_cR_outsider += np.array(cr_history_outsider)

# Averaging the results

# Profits:
average_profit_history /= num_simulations
average_profit_history_outsider /= num_simulations

# Beliefs:
average_belief_pL /= num_simulations
average_belief_pR /= num_simulations
average_belief_pL_outsider /= num_simulations
average_belief_pR_outsider /= num_simulations

# Capabilities:
average_capability_cL /= num_simulations
average_capability_cR /= num_simulations
average_capability_cL_outsider /= num_simulations
average_capability_cR_outsider /= num_simulations

# defining time for plotting
time_insider = np.arange(1, big_t )
time_outsider = np.arange(t_D, big_t)

# Print a statement that summarizes the loss due to an outsider
dif = average_profit_history[big_t - 2] - average_profit_history_outsider[(big_t - t_D - 1)] - average_profit_history[t_D-1]
print("cumulative insider advantage = {}".format(dif))

# Plotting the beliefs over time
plt.rcParams["figure.figsize"] = (10,5)
plt.plot(time_insider,average_belief_pL, label='Belief in Left INSIDER (pL)', color = "black", linestyle = "--")
plt.plot(time_insider,average_belief_pR, label='Belief in Right INSIDER(pR)', color = "black")
plt.plot(time_outsider,average_belief_pL_outsider, label='Belief in Left OUTSIDER (pL)', color = "green", linestyle = "--")
plt.plot(time_outsider,average_belief_pR_outsider, label='Belief in Right OUTSIDER (pR)', color = "green")
plt.axvline(t_D, color = "red", linestyle = "-.")
plt.xlabel('Time Periods')
plt.ylabel('Belief Probability')
plt.title('Agent Belief Convergence Over Time with Fidelity = {}'.format(xi_base))
plt.legend()
plt.grid()
plt.show()

# Plotting the Capability over time
plt.plot(time_insider,cl_history, label='Capability in Left INSIDER (cL)', color = "black", linestyle = "--")
plt.plot(time_insider,cr_history, label='Capability in Right INSIDER (cR)', color = "black")
plt.plot(time_outsider,cl_history_outsider, label='Capability in Left OUTSIDER (cL)', color = "green", linestyle = "--")
plt.plot(time_outsider,cr_history_outsider, label='Capability in Right OUTSIDER (cR)', color = "green")
plt.xlabel('Time Periods')
plt.ylabel('Capability')
plt.title('Agent Capability Over Time with Fidelity = {}'.format(xi_base))
plt.legend()
plt.grid()
plt.show()


# Plotting the profit over time
plt.plot(time_insider, average_profit_history, label='Cumulative Profit INSIDER ', color = "black", linestyle = ":")
plt.plot(time_outsider, average_profit_history_outsider, label='Cumulative Profit OUTSIDER', color = "green", linestyle = ":")
plt.xlabel('Time Periods')
plt.ylabel('Profit')
plt.title('Agent Profit Over Time with Fidelity = {}'.format(xi_base))
plt.legend()
plt.grid()
plt.show()

import pandas as pd

profit_series = pd.Series(profit_history)

# Plotting the profit over time
plt.plot(profit_series.diff(), label='Difference in Profit', color = "black", linestyle = ":")
plt.xlabel('Time Periods')
plt.ylabel('Profit')
plt.title('Agent Difference in Profit Over Time with Fidelity = {}'.format(xi_base))
plt.legend()
plt.grid()
#plt.show()

