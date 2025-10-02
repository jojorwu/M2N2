import streamlit as st
import pandas as pd
import os
import time
import json
import yaml

def write_command_config(config_data):
    """Writes the given configuration data to command_config.json."""
    with open('command_config.json', 'w') as f:
        json.dump(config_data, f, indent=4)
    st.toast("Commands sent to simulator!")

def show_monitoring_page():
    # --- Main Content Area for Displaying Simulation State ---
    log_file = 'fitness_log.csv'

    if not os.path.exists(log_file):
        st.warning(
            "The 'fitness_log.csv' file was not found. "
            "Please start the simulation by running `python3 -m src.main` in your terminal. "
            "The dashboard will automatically update once the simulation begins."
        )
        time.sleep(5)
        st.rerun()
        return

    # Read and Display Data
    try:
        df = pd.read_csv(log_file)

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

    # Auto-refresh logic
    time.sleep(5)
    st.rerun()


def show_settings_page():
    st.header("Simulation Settings")
    st.write("Modify the simulation parameters in real-time. Changes will be applied at the start of the next generation.")

    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("`config.yaml` not found. Make sure it's in the root directory.")
        return

    # --- Display and Edit Settings ---
    st.subheader("Evolutionary Settings")

    # Using columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        config['num_generations'] = st.number_input("Number of Generations", min_value=1, value=config.get('num_generations', 10))
        config['population_size'] = st.number_input("Population Size", min_value=2, value=config.get('population_size', 10))
        config['mutation_rate'] = st.slider("Mutation Rate", 0.0, 1.0, value=config.get('mutation_rate', 0.05))

    with col2:
        config['merge_strategy'] = st.selectbox("Merge Strategy",
                                                options=['average', 'fitness_weighted', 'layer-wise', 'sequential_constructive'],
                                                index=['average', 'fitness_weighted', 'layer-wise', 'sequential_constructive'].index(config.get('merge_strategy', 'average')))
        config['initial_mutation_strength'] = st.slider("Initial Mutation Strength", 0.0, 1.0, value=config.get('initial_mutation_strength', 0.1))
        config['mutation_decay_factor'] = st.slider("Mutation Decay Factor", 0.0, 1.0, value=config.get('mutation_decay_factor', 0.99))

    st.subheader("Optimizer and Scheduler")
    col1, col2 = st.columns(2)

    with col1:
        config['optimizer_config']['learning_rate'] = st.number_input("Learning Rate", min_value=0.0001, format="%.4f", value=config.get('optimizer_config', {}).get('learning_rate', 0.001))

    with col2:
        config['scheduler_config']['patience'] = st.number_input("Scheduler Patience", min_value=0, value=config.get('scheduler_config', {}).get('patience', 5))
        config['scheduler_config']['factor'] = st.slider("Scheduler Factor", 0.0, 1.0, value=config.get('scheduler_config', {}).get('factor', 0.5))

    if st.button("Update Configuration"):
        write_command_config(config)

def main():
    """
    Defines the Streamlit dashboard for visualizing simulation results and
    providing real-time control over its parameters.
    """
    st.set_page_config(page_title="M2N2 Simulation Monitor", layout="wide")
    st.title("M2N2 Simulation Monitor & Control")

    # --- Sidebar for Navigation and Controls ---
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Monitoring", "Settings"])

        st.divider()

        st.header("Live Simulation Controls")

        if st.button("Stop Simulation Gracefully"):
            write_command_config({"stop_simulation": True})

        if st.button("Restart Simulation"):
            write_command_config({"restart_simulation": True})

    # --- Page Content ---
    if page == "Monitoring":
        show_monitoring_page()
    elif page == "Settings":
        show_settings_page()


if __name__ == "__main__":
    main()