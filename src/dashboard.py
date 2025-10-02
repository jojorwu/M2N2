import streamlit as st
import pandas as pd
import os
import time
import json

def write_command_config(config_data):
    """Writes the given configuration data to command_config.json."""
    with open('command_config.json', 'w') as f:
        json.dump(config_data, f, indent=4)
    st.toast("Commands sent to simulator!")

def main():
    """
    Defines the Streamlit dashboard for visualizing simulation results and
    providing real-time control over its parameters.
    """
    st.set_page_config(page_title="M2N2 Simulation Monitor", layout="wide")
    st.title("M2N2 Simulation Monitor & Control")

    # --- Sidebar for Interactive Controls ---
    with st.sidebar:
        st.header("Live Simulation Controls")

        # Define available merge strategies and a default mutation rate
        merge_strategies = ['average', 'fitness_weighted', 'layer-wise', 'sequential_constructive']
        default_mutation_rate = 0.05

        # Initialize session state if it doesn't exist
        if 'merge_strategy' not in st.session_state:
            st.session_state.merge_strategy = merge_strategies[0]
        if 'mutation_rate' not in st.session_state:
            st.session_state.mutation_rate = default_mutation_rate

        # Create the UI widgets
        selected_strategy = st.selectbox(
            "Merge Strategy for Next Child",
            options=merge_strategies,
            index=merge_strategies.index(st.session_state.get('merge_strategy', merge_strategies[0]))
        )

        selected_mutation_rate = st.slider(
            "Mutation Rate for Next Child",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('mutation_rate', default_mutation_rate),
            step=0.01
        )

        # Update session state and write to file if changes are detected
        if (selected_strategy != st.session_state.merge_strategy or
            selected_mutation_rate != st.session_state.mutation_rate):
            st.session_state.merge_strategy = selected_strategy
            st.session_state.mutation_rate = selected_mutation_rate

            command_config = {
                "merge_strategy": selected_strategy,
                "mutation_rate": selected_mutation_rate
            }
            write_command_config(command_config)

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

if __name__ == "__main__":
    main()