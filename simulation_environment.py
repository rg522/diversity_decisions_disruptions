import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import json

class SimulationEnvironment:
    def __init__(self, d_type, V_H_location, big_t, t_D, num_simulations, insider_class, outsider_class, config_file=None):
        # Initialize the simulation parameters
        self.d_type = d_type # option from 1 to 4 depending on the quadrant
        self.V_H_location = V_H_location # set the initial location True for Right,  False for Left
        self.big_t = big_t # set the simulation length
        self.t_D = t_D # set the time of disruption
        self.num_simulations = num_simulations # set the number of iterations
        self.C_initial = 5

        # Set parameters
        if config_file:
            self.initialize_parameters_from_file(config_file)
        else:
            self.initialize_parameters_default()

        # Initialize agents
        self.insider = insider_class(self,self.C_initial,self.C_initial,'static')  # Passing the environment to the agents
        self.outsider = outsider_class(self,self.C_initial,self.C_initial,'static')
        self.insider_rand = insider_class(self, self.C_initial, self.C_initial,'random')  # Random decision elements
        self.outsider_rand = outsider_class(self, self.C_initial, self.C_initial, 'random')
        self.active_agents = [self.insider, self.insider_rand]  # Initially only the insider is active (prior to the disruption)

        self.set_d_type()  # Set the disruption type and its effects


    def initialize_parameters_default(self):
        # Set up initial simulation parameters (default values)
        self.V_H = 1  # Value for high
        self.V_L = 0  # Value for low
        self.xi = 0.0 # Fidelity - Probability of receiving the opposite signal
        self.C_L_0 = 5  # Initial capability for Left
        self.C_R_0 = 5  # Initial capability for Right
        self.delta_in = 100  # Differential learning curve for insider and outsider
        self.delta_out = 1000
        self.V_H_new = self.V_L_new = 0
        self.reset_cap_left = self.reset_cap_right = self.reset_location = False


    def initialize_parameters_from_file(self, config_file="config.json"):
        # Load parameters from a JSON file (for non-default values)
        with open(config_file, 'r') as file:
            config = json.load(file)

        self.V_H = config["V_H"]
        self.V_L = config["V_L"]
        self.xi = config["xi"]
        self.C_L_0 = config["C_L_0"]
        self.C_R_0 = config["C_R_0"]
        self.delta_in = config["delta_in"]
        self.delta_out = config["delta_out"]


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

        # swap te value if required
        if self.reset_location:
            self.V_H_location = not self.V_H_location  # Swap the value location

    def update_environment(self, current_period):
        if current_period == self.t_D:
            self.apply_disruption()
            self.active_agents.append(self.outsider)  # Include the outsider after disruption
            self.active_agents.append(self.outsider_rand)

    def simulate_period(self, current_period):
        for agent in self.active_agents:
            # apply the disruption
            if current_period == self.t_D:
                agent.take_action(current_period, True)
            # otherwise do not
            else:
                agent.take_action(current_period, False)


    def run_simulation(self):
        for period in range(self.big_t):
            self.simulate_period(period)
            self.update_environment(period)
            print(self.V_H_location)

    def visualize_results(self, analysis = 1):
        self.analysis = analysis  # 1 for insider vs. outsider, 2 for random insider vs. non-random insider
        # Convert the agent's history dictionary to a DataFrame for easier plotting
        df_insider = pd.DataFrame(self.insider.history)
        df_outsider = pd.DataFrame(self.outsider.history)
        df_insider_rand = pd.DataFrame(self.insider_rand.history)
        df_outsider_rand = pd.DataFrame(self.outsider_rand.history)

        if self.analysis == 1:
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

            # plotting beliefs over time
            plt.figure(figsize=(10, 6))
            plt.plot(df_insider['time'], df_insider['pL>R'], label='Insider Left Better Belief', color='black')
            plt.plot(df_insider['time'], df_insider['pR>L'], label='Insider Right Better Belief', color='black', linestyle=':')
            plt.plot(df_outsider['time'], df_outsider['pL>R'], label='Outsider Left Better Belief', color='red')
            plt.plot(df_outsider['time'], df_outsider['pR>L'], label='Outsider Right Better Belief', color='red', linestyle=':')
            plt.title('Agent Beliefs Over Time')
            plt.xlabel('Time')
            plt.ylabel('Probability of Better Value at This Location')
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

        if self.analysis == 2:
            # plotting beliefs over time
            plt.figure(figsize=(10, 6))
            plt.plot(df_insider['time'], df_insider['pL'], label='Static Insider Left Belief', color='black')
            plt.plot(df_insider['time'], df_insider['pR'], label='Static Insider Right Belief', color='black', linestyle=':')
            plt.plot(df_insider_rand['time'], df_insider_rand['pL'], label='Random Insider Left Belief', color='green')
            plt.plot(df_insider_rand['time'], df_insider_rand['pR'], label='Random Insider Right Belief', color='green', linestyle=':')
            plt.title('Agent Beliefs Over Time')
            plt.xlabel('Time')
            plt.ylabel('Probability of High Value at This Location')
            plt.legend()
            plt.show()

            # plotting BETTER beliefs over time
            plt.figure(figsize=(10, 6))
            plt.plot(df_insider['time'], df_insider['pL>R'], label='Static Insider Left Better Belief', color='black')
            plt.plot(df_insider['time'], df_insider['pR>L'], label='Static Insider Right Better Belief', color='black', linestyle=':')
            plt.plot(df_insider_rand['time'], df_insider_rand['pL>R'], label='Random Insider Left Better Belief', color='green')
            plt.plot(df_insider_rand['time'], df_insider_rand['pR>L'], label='Random Insider Right Better Belief', color='green', linestyle=':')
            plt.title('Agent Beliefs Over Time')
            plt.xlabel('Time')
            plt.ylabel('Probability of Better Value at This Location')
            plt.legend()
            plt.show()

            # plotting capabilities over time
            plt.figure(figsize=(10, 6))
            plt.plot(df_insider['time'], df_insider['C_L'], label='Static Insider Left Capability', color='black')
            plt.plot(df_insider['time'], df_insider['C_R'], label='Static Insider Right Capability', color='black', linestyle=':')
            plt.plot(df_insider_rand['time'], df_insider_rand['C_L'], label='Random Insider Capability', color='green')
            plt.plot(df_insider_rand['time'], df_insider_rand['C_R'], label='Random Insider Right Capability', color='green', linestyle=':')
            plt.title('Agent Capabilities Over Time')
            plt.xlabel('Time')
            plt.ylabel('Capability in this Direction')
            plt.legend()
            plt.show()


            # plotting the cumulative profit over time for insiders: random and static
            plt.figure(figsize=(10, 6))
            plt.plot(df_insider['time'], df_insider['profit'], label='Static Insider Cumulative Profit', color='black')
            plt.plot(df_insider['time'], df_insider_rand['profit'], label='Random Insider Cumulative Profit', color='green')
            plt.title('Agent Cumulative Profit Over Time')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Profit for Insiders: Random and Static')
            plt.legend()
            plt.show()



    def save_results(self, performance_data, filename="simulation_results.csv"):
        # Saving the results to a CSV file
        df_insider = pd.DataFrame(self.insider.history)
        df_insider.to_csv(filename, index=False)
        print(f"Results saved to {filename}")


class InsiderAgent:
    def __init__(self, environment, C_L_initial, C_R_initial, type = 'static'):
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
        self.type = type

        # Initialize history dictionary
        self.history = {
            "pL": [], "pR": [],
            "pL>R": [], "pR>L": [],
            "profit": [], "time": [],
            "C_L": [], "C_R": []
        }

    def simulate_decision(self, t):
        # Calculate beliefs and probabilities
        pL_in = self.nL_s / self.nL_belief if self.nL_belief > 0 else 0.5
        pR_in = self.nR_s / self.nR_belief if self.nR_belief > 0 else 0.5


        # Decision making logic
        self.P_L_greater_R = (self.nL_s + self.nR_belief - self.nR_s) / (self.nR_belief + self.nL_belief + 1)
        self.P_R_greater_L = (self.nR_s + self.nL_belief - self.nL_s) / (self.nR_belief + self.nL_belief + 1)

        # Set the agent type
        if self.type == 'random':
            decision = 'L' if self.P_L_greater_R > random.uniform(0, 1) else 'R'

        else:
            decision = 'L' if self.P_L_greater_R > self.P_R_greater_L else 'R'

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
        self.history["pL>R"].append(self.P_L_greater_R)
        self.history["pR>L"].append(self.P_R_greater_L)
        self.history["profit"].append(self.cumulative_profit)
        self.history["time"].append(t)
        self.history["C_L"].append(self.C_L)
        self.history["C_R"].append(self.C_R)

    def take_action(self, current_period, disruption):
        # Run each step in the simulation
        # Check if the disruption has occurred and reset capabilities if required
        if disruption:
            if self.env.reset_cap_right:
                self.nL_capability = 0
                self.C_L = self.C_L_0  # Reset Left capability to initial value

            if self.env.reset_cap_right:
                self.nR_capability = 0
                self.C_R = self.C_R_0  # Reset Right capability to initial value

        self.simulate_decision(current_period)


class OutsiderAgent:
    def __init__(self, environment, C_L_initial, C_R_initial, type = 'static'):
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
        self.delta = self.env.delta_out
        self.type = type

        # Initialize history dictionary
        self.history = {
            "pL": [], "pR": [],
            "pL>R": [], "pR>L": [],
            "profit": [], "time": [],
            "C_L": [], "C_R": []
        }

    def simulate_decision(self, t):
        # Calculate beliefs and probabilities
        pL_in = self.nL_s / self.nL_belief if self.nL_belief > 0 else 0.5
        pR_in = self.nR_s / self.nR_belief if self.nR_belief > 0 else 0.5

        # Decision making logic
        self.P_L_greater_R = (self.nL_s + self.nR_belief - self.nR_s) / (self.nR_belief + self.nL_belief + 1)
        self.P_R_greater_L = (self.nR_s + self.nL_belief - self.nL_s) / (self.nR_belief + self.nL_belief + 1)

        # Set the agent type
        if self.type == 'random':
            decision = 'L' if self.P_L_greater_R > random.uniform(0.5, 1) else 'R'

        else:
            decision = 'L' if self.P_L_greater_R > self.P_R_greater_L else 'R'


        # Update capabilities
        C_L_0 = self.env.C_L_0  # Set initial capability levels
        C_R_0 = self.env.C_R_0
        self.C_L = C_L_0 - self.delta * C_L_0 / (self.delta + self.nL_capability) if self.nL_capability > 0 else 0
        self.C_R = C_R_0 - self.delta * C_R_0 / (self.delta + self.nR_capability) if self.nR_capability > 0 else 0

        # Determine actual outcome, update capabilities and profit
        if decision == 'L':
            self.nL_belief += 1
            self.nL_capability += 1
            self.profit = self.env.V_H - (1 / (self.C_L + 1)) if (random.random() > self.env.xi and not self.env.V_H_location) or (
             random.random() <= self.env.xi and self.env.V_H_location) else self.env.V_L - (1 / (self.C_L + 1))
            if (random.random() > self.env.xi and not self.env.V_H_location) or (
                random.random() <= self.env.xi and self.env.V_H_location):
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
        self.history["pL>R"].append(self.P_L_greater_R)
        self.history["pR>L"].append(self.P_R_greater_L)
        self.history["profit"].append(self.cumulative_profit)
        self.history["time"].append(t)
        self.history["C_L"].append(self.C_L)
        self.history["C_R"].append(self.C_R)

    def take_action(self, current_period, disruption):
        # Run each step in the simulation
        # Check if the disruption has occurred and reset capabilities if required
        if disruption:
            if self.env.reset_cap_right:
                self.nL_capability = 0
                self.C_L = self.C_L_0  # Reset Left capability to initial value

            if self.env.reset_cap_right:
                self.nR_capability = 0
                self.C_R = self.C_R_0  # Reset Right capability to initial value

        self.simulate_decision(current_period)



