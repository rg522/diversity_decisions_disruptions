from simulation_environment import SimulationEnvironment, OutsiderAgent, InsiderAgent

# Initialize the environment with required parameters
simulation_env = SimulationEnvironment(d_type=4, V_H_location=False, big_t=10000, t_D=2000, num_simulations=100,
                                       insider_class=InsiderAgent, outsider_class=OutsiderAgent, config_file='config.json')


# Run the simulation
simulation_env.run_simulation()

# Visualize the results
simulation_env.visualize_results(analysis=2)

# Being random starts to suck when the fidelity is not perfect