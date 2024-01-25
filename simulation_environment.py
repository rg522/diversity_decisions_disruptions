import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

class SimulationEnvironment:
    def __init__(self, d_type, V_H_location, big_t, t_D, num_simulations, insider_class, outsider_class):
        # Initialize the simulation parameters
        self.d_type = d_type # option from 1 to 4 depending on the quadrant
        self.V_H_location = V_H_location # set the initial location True for Right,  False for Left
        self.big_t = big_t # set the simulation length
        self.t_D = t_D # set the time of disruption
        self.num_simulations = num_simulations # set the number of iterations
        self.C_initial = 5

        # Call to initialize other parameters
        self.initialize_parameters()

        # Initialize agents
        self.insider = insider_class(self,self.C_initial,self.C_initial)  # Passing the environment to the agents
        self.outsider = outsider_class(self,self.C_initial,self.C_initial)
        self.active_agents = [self.insider]  # Initially only the insider is active (prior to the disruption)

        self.set_d_type()  # Set the disruption type and its effects


    def initialize_parameters(self):
        # Set up initial simulation parameters
        self.V_H = 1  # Value for high
        self.V_L = 0  # Value for low
        self.xi = 0.0 # Fidelity - Probability of receiving the opposite signal
        self.C_L_0 = 5  # Initial capability for Left
        self.C_R_0 = 5  # Initial capability for Right
        self.delta_in = 100  # Differential learning curve for insider and outsider
        self.delta_out = 1000
        self.V_H_new = self.V_L_new = 0
        self.reset_cap_left = self.reset_cap_right = self.reset_location = False

    def set_d_type(self):
        # Set the disruption type and its effects
        self.V_H_new = 1  # How does the underlying value change?
        self.V_L_new = 0

        if self.d_type == 1:
            self.reset_location = False
            self.reset_cap_left = False
            self.reset_cap_right = False

        elif self.d_type == 2:
            self.reset_location = True
            self.reset_cap_left = False
            self.reset_cap_right = False

        elif self.d_type == 3:
            self.reset_location = False
            self.reset_cap_left = True
            self.reset_cap_right = True

        elif self.d_type == 4:
            self.reset_location = True
            self.reset_cap_left = True
            self.reset_cap_right = True

    def apply_disruption(self):
        self.xi = np.sqrt(self.xi)  # Update fidelity
        print("Disruption has occurred")
        print(self.reset_location)

        # Adjust V_H and V_L based on the disruption
        self.V_H, self.V_L = self.V_H_new, self.V_L_new

        # Reset capabilities if options change
        if self.reset_cap_left:
            self.nL_capability = 0
            self.C_L = self.C_L_0  # Reset Left capability to initial value

        if self.reset_cap_right:
            self.nR_capability = 0
            self.C_R = self.C_R_0  # Reset Right capability to initial value

        if self.reset_location:
            print("Value swap")
            self.V_H_location = not self.V_H_location  # Swap the value location

    def update_environment(self, current_period):
        if current_period == self.t_D:
            self.apply_disruption()
            self.active_agents.append(self.outsider)  # Include the outsider after disruption

    def simulate_period(self, current_period):
        for agent in self.active_agents:
            agent.take_action(current_period)  # Assuming each agent has a method 'take_action'
            # Collect data from the agents if needed

    def run_simulation(self):
        for period in range(self.big_t):
            self.simulate_period(period)
            self.update_environment(period)
            print(self.V_H_location)

    def visualize_results(self):
        # Convert the agent's history dictionary to a DataFrame for easier plotting
        df_insider = pd.DataFrame(self.insider.history)
        df_outsider = pd.DataFrame(self.outsider.history)

        # plotting beliefs over time
        plt.figure(figsize=(10, 6))
        plt.plot(df_insider['time'], df_insider['pL'], label='Insider Left Belief', color='black')
        plt.plot(df_insider['time'], df_insider['pR'], label='Insider Right Belief', color='black', linestyle=':')
        plt.plot(df_outsider['time'], df_outsider['pL'], label='Outsider Left Belief', color='red')
        plt.plot(df_outsider['time'], df_outsider['pR'], label='Outsider Right Belief', color='red', linestyle=':')
        plt.title('Agent Beliefs Over Time')
        plt.xlabel('Time')
        plt.ylabel('Probability of High Value at This Location')
        plt.legend()
        plt.show()

        # plotting capabilities over time
        plt.figure(figsize=(10, 6))
        plt.plot(df_insider['time'], df_insider['C_L'], label='Insider Left Capability', color='black')
        plt.plot(df_insider['time'], df_insider['C_R'], label='Insider Right Capability', color='black', linestyle=':')
        plt.plot(df_outsider['time'], df_outsider['C_L'], label='Outsider Left Capability', color='red')
        plt.plot(df_outsider['time'], df_outsider['C_R'], label='Outsider Right Capability', color='red', linestyle=':')
        plt.title('Agent Capabilities Over Time')
        plt.xlabel('Time')
        plt.ylabel('Capability in this Direction')
        plt.legend()
        plt.show()

        # plotting the cumulative profit over time
        plt.figure(figsize=(10, 6))
        plt.plot(df_insider['time'], df_insider['profit'], label='Insider Cumulative Profit', color='black')
        plt.plot(df_outsider['time'], df_outsider['profit'], label='Outsider Cumulative Profit', color='red')
        plt.title('Agent Cumulative Profit Over Time')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Profit')
        plt.legend()
        plt.show()



    def save_results(self, performance_data, filename="simulation_results.csv"):
        # Saving the results to a CSV file
        df_insider = pd.DataFrame(self.insider.history)
        df_insider.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


class InsiderAgent:
    def __init__(self, environment, C_L_initial, C_R_initial):
        self.env = environment
        # Initialize variables
        self.nL_s = 0
        self.nL_belief = 0
        self.nR_s = 0
        self.nR_belief = 0
        self.nL_capability = 0
        self.nR_capability = 0
        self.cumulative_profit = 0
        self.C_L = 0
        self.C_L = 0
        self.profit = 0
        self.C_L_0 = C_L_initial
        self.C_R_0 = C_R_initial

        # Initialize history dictionary
        self.history = {
            "pL": [], "pR": [],
            "profit": [], "time": [],
            "C_L": [], "C_R": []
        }

    def simulate_decision(self, t):
        # Calculate beliefs and probabilities
        pL_in = self.nL_s / self.nL_belief if self.nL_belief > 0 else 0.5
        pR_in = self.nR_s / self.nR_belief if self.nR_belief > 0 else 0.5


        # Decision making logic
        P_L_greater_R = (self.nL_s + self.nR_belief - self.nR_s) / (self.nR_belief + self.nL_belief) if (self.nR_belief + self.nL_belief) > 0 else 0.5
        P_R_greater_L = (self.nR_s + self.nL_belief - self.nL_s) / (self.nR_belief + self.nL_belief) if (self.nR_belief + self.nL_belief) > 0 else 0.5

        decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

        # Update capabilities
        delta_in = self.env.delta_in  # Set learning curve from parameters in environment
        C_L_0 = self.env.C_L_0  # Set initial capability levels
        C_R_0 = self.env.C_R_0
        self.C_L = C_L_0 - delta_in * C_L_0 / (delta_in + self.nL_capability) if self.nL_capability > 0 else 0
        self.C_R = C_R_0 - delta_in * C_R_0 / (delta_in + self.nR_capability) if self.nR_capability > 0 else 0

        # Determine actual outcome, update capabilities and profit
        if decision == 'L':
            self.nL_belief += 1
            self.nL_capability += 1
            self.profit = self.env.V_H - (1 / (self.C_L + 1)) if (random.random() > self.env.xi and not self.env.V_H_location) or (
                        random.random() <= self.env.xi and self.env.V_H_location) else self.env.V_L - (1 / (self.C_L + 1))
            if (random.random() > self.env.xi and not self.env.V_H_location) or (random.random() <= self.env.xi and self.env.V_H_location):
                self.nL_s += 1
        else:
            self.nR_belief += 1
            self.nR_capability += 1
            self.profit = self.env.V_H - (1 / (self.C_R + 1)) if (random.random() > self.env.xi and self.env.V_H_location) or (
                        random.random() <= self.env.xi and not self.env.V_H_location) else self.env.V_L - (1 / (self.C_R + 1))
            if (random.random() > self.env.xi and self.env.V_H_location) or (random.random() <= self.env.xi and not self.env.V_H_location):
                self.nR_s += 1

        # Update the profit history for the current period
        self.cumulative_profit += self.profit

        # Append to histories
        self.history["pL"].append(pL_in)
        self.history["pR"].append(pR_in)
        self.history["profit"].append(self.cumulative_profit)
        self.history["time"].append(t)
        self.history["C_L"].append(self.C_L)
        self.history["C_R"].append(self.C_R)

    def take_action(self, current_period):
        # Call the simulate_decision method for each period of the simulation
        # possibly unnecessary step to be refactored later
        self.simulate_decision(current_period)


class OutsiderAgent:
    def __init__(self, environment, C_L_initial, C_R_initial):
        self.env = environment
        # Initialize variables
        self.nL_s = 0
        self.nL_belief = 0
        self.nR_s = 0
        self.nR_belief = 0
        self.nL_capability = 0
        self.nR_capability = 0
        self.cumulative_profit = 0
        self.C_L = 0
        self.C_L = 0
        self.profit = 0
        self.C_L_0 = C_L_initial
        self.C_R_0 = C_R_initial

        # Initialize history dictionary
        self.history = {
            "pL": [], "pR": [],
            "profit": [], "time": [],
            "C_L": [], "C_R": []
        }

    def simulate_decision(self, t):
        # Calculate beliefs and probabilities
        pL_in = self.nL_s / self.nL_belief if self.nL_belief > 0 else 0.5
        pR_in = self.nR_s / self.nR_belief if self.nR_belief > 0 else 0.5

        # Decision making logic
        P_L_greater_R = (self.nL_s + self.nR_belief - self.nR_s) / (self.nR_belief + self.nL_belief) if (
                                                                                                                    self.nR_belief + self.nL_belief) > 0 else 0.5
        P_R_greater_L = (self.nR_s + self.nL_belief - self.nL_s) / (self.nR_belief + self.nL_belief) if (
                                                                                                                    self.nR_belief + self.nL_belief) > 0 else 0.5

        decision = 'L' if P_L_greater_R > P_R_greater_L else 'R'

        # Update capabilities
        delta = self.env.delta_out  # Set learning curve from parameters in environment
        C_L_0 = self.env.C_L_0  # Set initial capability levels
        C_R_0 = self.env.C_R_0
        self.C_L = C_L_0 - delta * C_L_0 / (delta + self.nL_capability) if self.nL_capability > 0 else 0
        self.C_R = C_R_0 - delta * C_R_0 / (delta + self.nR_capability) if self.nR_capability > 0 else 0

        # Determine actual outcome, update capabilities and profit
        if decision == 'L':
            self.nL_belief += 1
            self.nL_capability += 1
            self.profit = self.env.V_H - (1 / (self.C_L + 1)) if (
                                                                             random.random() > self.env.xi and not self.env.V_H_location) or (
                                                                         random.random() <= self.env.xi and self.env.V_H_location) else self.env.V_L - (
                        1 / (self.C_L + 1))
            if (random.random() > self.env.xi and not self.env.V_H_location) or (
                    random.random() <= self.env.xi and self.env.V_H_location):
                self.nL_s += 1
        else:
            self.nR_belief += 1
            self.nR_capability += 1
            self.profit = self.env.V_H - (1 / (self.C_R + 1)) if (
                                                                             random.random() > self.env.xi and self.env.V_H_location) or (
                                                                         random.random() <= self.env.xi and not self.env.V_H_location) else self.env.V_L - (
                        1 / (self.C_R + 1))
            if (random.random() > self.env.xi and self.env.V_H_location) or (
                    random.random() <= self.env.xi and not self.env.V_H_location):
                self.nR_s += 1

        # Update the profit history for the current period
        self.cumulative_profit += self.profit

        # Append to histories
        self.history["pL"].append(pL_in)
        self.history["pR"].append(pR_in)
        self.history["profit"].append(self.cumulative_profit)
        self.history["time"].append(t)
        self.history["C_L"].append(self.C_L)
        self.history["C_R"].append(self.C_R)

    def take_action(self, current_period):
        # Call the simulate_decision method for each period of the simulation
        # possibly unnecessary step to be refactored later
        self.simulate_decision(current_period)



# Example usage
# Initialize the environment with required parameters
simulation_env = SimulationEnvironment(d_type=3, V_H_location=False, big_t=5000, t_D=2000, num_simulations=10,
                                       insider_class=InsiderAgent, outsider_class=OutsiderAgent)

# Run the simulation
simulation_env.run_simulation()

# Visualize the results
simulation_env.visualize_results()