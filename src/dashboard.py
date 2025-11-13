"""
Streamlit dashboard for monitoring and controlling the M2N2 simulation.

This script launches a web-based interface that provides real-time visualization
of the simulation's progress and allows for dynamic adjustment of its key
parameters. The dashboard is decoupled from the main simulation logic; it reads
data from a shared log file (`fitness_log.csv`) and sends control commands
via a JSON file (`command.json`).

To run the dashboard:
    streamlit run src/dashboard.py
"""
import streamlit as st
import pandas as pd
import os
import time
import json
import yaml
from src.constants import COMMAND_FILE, FITNESS_LOG_FILENAME


def _update_command_file(updates: dict):
    """
    Reads, updates, and writes the JSON command file.

    This helper function safely handles the command file, which acts as a
    communication channel between the dashboard and the running simulation. It
    reads the existing commands, updates them with new ones, and writes the
    result back to the file.

    Args:
        updates (dict): A dictionary of new commands or parameters to be
                        added or updated in the command file.
    """
    config = {}
    if os.path.exists(COMMAND_FILE):
        with open(COMMAND_FILE, 'r') as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError:
                st.warning("Could not parse command file, creating a new one.")

    config.update(updates)

    with open(COMMAND_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    st.toast("Commands sent to simulator!")


def show_monitoring_page():
    """
    Displays the main monitoring page of the dashboard.

    This function is responsible for rendering the real-time plots and data
    tables that visualize the simulation's fitness history. It continuously
    polls the `fitness_log.csv` file and updates the charts, providing a live
    view of the evolutionary process.
    """
    st.header("Live Simulation Feed")
    if not os.path.exists(FITNESS_LOG_FILENAME):
        st.warning(
            "The 'fitness_log.csv' file was not found. "
            "Please start the simulation by running `python3 -m src.main` in your terminal. "
            "The dashboard will automatically update once the simulation begins."
        )
        # Use a long sleep and rerun to avoid high CPU usage on a static page
        time.sleep(5)
        st.rerun()
        return

    # Read and Display Data
    try:
        df = pd.read_csv(FITNESS_LOG_FILENAME)

        if not df.empty:
            st.header("Fitness History")
            st.line_chart(df, x='generation', y=['best_fitness', 'average_fitness'])

            st.header("Raw Fitness Data")
            st.dataframe(df)

            last_gen = df['generation'].max()
            st.metric(label="Last Recorded Generation", value=int(last_gen))
        else:
            st.info("Waiting for the first generation data...")

    except pd.errors.EmptyDataError:
        st.info("Log file is empty. Waiting for data...")
    except Exception as e:
        st.error(f"An error occurred while reading the log file: {e}")

    # Auto-refresh logic to keep the dashboard live
    time.sleep(5)
    st.rerun()


def show_settings_page():
    """
    Displays the settings page for real-time simulation control.

    This function renders a form with various input widgets (sliders, number
    inputs, etc.) that allow the user to modify simulation parameters on the fly.
    When the "Update Configuration" button is pressed, it sends these new values
    to the simulator via the command file.
    """
    st.header("Simulation Settings")
    st.write("Modify the simulation parameters in real-time. Changes will be applied at the start of the next generation.")

    # Load config from file ONCE and store in session state for persistence
    if 'simulation_config' not in st.session_state:
        try:
            with open('config.yaml', 'r') as f:
                st.session_state.simulation_config = yaml.safe_load(f)
        except FileNotFoundError:
            st.error("`config.yaml` not found. Make sure it's in the root directory.")
            return
        except yaml.YAMLError as e:
            st.error(f"Error parsing `config.yaml`: {e}")
            return

    config = st.session_state.simulation_config

    st.subheader("Evolutionary Settings")
    col1, col2 = st.columns(2)

    with col1:
        config['num_generations'] = st.number_input("Number of Generations", min_value=1, value=config.get('num_generations', 10))
        config['population_size'] = st.number_input("Population Size", min_value=2, value=config.get('population_size', 10))
        config['mutation_rate'] = st.slider("Mutation Rate", 0.0, 1.0, value=config.get('mutation_rate', 0.05))

    with col2:
        options = ['average', 'fitness_weighted', 'layer-wise', 'sequential_constructive']
        current_strategy = config.get('merge_strategy', 'average')
        if current_strategy not in options:
            current_strategy = 'average'
        config['merge_strategy'] = st.selectbox("Merge Strategy",
                                                options=options,
                                                index=options.index(current_strategy))
        config['initial_mutation_strength'] = st.slider("Initial Mutation Strength", 0.0, 1.0, value=config.get('initial_mutation_strength', 0.1))
        config['mutation_decay_factor'] = st.slider("Mutation Decay Factor", 0.0, 1.0, value=config.get('mutation_decay_factor', 0.99))

    st.subheader("Optimizer and Scheduler")
    col1, col2 = st.columns(2)

    with col1:
        if 'optimizer_config' not in config: config['optimizer_config'] = {}
        config['optimizer_config']['learning_rate'] = st.number_input("Learning Rate", min_value=0.0001, format="%.4f", value=config.get('optimizer_config', {}).get('learning_rate', 0.001))

    with col2:
        if 'scheduler_config' not in config: config['scheduler_config'] = {}
        config['scheduler_config']['patience'] = st.number_input("Scheduler Patience", min_value=0, value=config.get('scheduler_config', {}).get('patience', 5))
        config['scheduler_config']['factor'] = st.slider("Scheduler Factor", 0.0, 1.0, value=config.get('scheduler_config', {}).get('factor', 0.5))

    if st.button("Update Configuration"):
        dynamic_config = {
            'num_generations': config.get('num_generations'),
            'population_size': config.get('population_size'),
            'mutation_rate': config.get('mutation_rate'),
            'merge_strategy': config.get('merge_strategy'),
            'initial_mutation_strength': config.get('initial_mutation_strength'),
            'mutation_decay_factor': config.get('mutation_decay_factor'),
            'optimizer_config': {'learning_rate': config.get('optimizer_config', {}).get('learning_rate')},
            'scheduler_config': {'patience': config.get('scheduler_config', {}).get('patience'), 'factor': config.get('scheduler_config', {}).get('factor')}
        }
        _update_command_file(dynamic_config)

def _handle_sidebar_commands():
    """Renders sidebar controls and handles command file updates."""
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Monitoring", "Settings"])
        st.divider()
        st.header("Live Simulation Controls")
        if st.button("Stop Simulation Gracefully"):
            _update_command_file({"stop_simulation": True})
        if st.button("Restart Simulation"):
            # Clear any stale stop commands and issue a restart
            _update_command_file({"restart_simulation": True, "stop_simulation": False})
    return page

def main():
    """
    Main function to run the Streamlit dashboard.

    This function sets up the page configuration, title, and sidebar navigation.
    It acts as the entry point for the dashboard application, routing the user
    to the appropriate page (`Monitoring` or `Settings`) based on their
    selection.
    """
    st.set_page_config(page_title="M2N2 Simulation Monitor", layout="wide")
    st.title("M2N2 Simulation Monitor & Control")

    page = _handle_sidebar_commands()

    if page == "Monitoring":
        show_monitoring_page()
    elif page == "Settings":
        show_settings_page()

if __name__ == "__main__":
    main()
