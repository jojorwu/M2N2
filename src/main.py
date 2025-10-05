"""
The main entry point for running the M2N2 evolutionary simulation.

This script creates an instance of the EvolutionSimulator and calls its
run() method to start the experiment.
"""
from .simulator import EvolutionSimulator

def main():
    """
    Initializes and runs the evolutionary simulation.
    """
    simulator = EvolutionSimulator(config_path='config.yaml')
    simulator.run()

if __name__ == '__main__':
    main()