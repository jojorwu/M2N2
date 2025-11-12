"""
Main entry point for running the M2N2 evolutionary simulation.

This script initializes and runs the entire evolutionary experiment from start to
finish. It serves as the primary executable for the simulation backend.

To run the simulation, execute this script from the project's root directory:
    python3 -m src.main

The simulation's behavior is controlled by the `config.yaml` file.
"""
from .simulator import EvolutionSimulator

def main():
    """
    Initializes the EvolutionSimulator and starts the simulation run.

    This function creates an instance of the simulator, which automatically
    loads the configuration from `config.yaml`, and then calls the `run`
    method to begin the evolutionary process.
    """
    simulator = EvolutionSimulator(config_path='config.yaml')
    simulator.run()

if __name__ == '__main__':
    main()
