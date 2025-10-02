import streamlit as st
import pandas as pd
import os
import time

def main():
    """
    Defines the Streamlit dashboard for visualizing simulation results.
    This dashboard reads data from 'fitness_log.csv' and auto-refreshes.
    """
    st.set_page_config(page_title="M2N2 Simulation Monitor", layout="wide")
    st.title("M2N2 Simulation Monitor")

    log_file = 'fitness_log.csv'

    # --- Main Content Area ---
    if not os.path.exists(log_file):
        st.warning(
            "The 'fitness_log.csv' file was not found. "
            "Please start the simulation by running `python3 -m src.main` in your terminal. "
            "The dashboard will automatically update once the simulation begins."
        )
        # Keep retrying to find the file
        time.sleep(5)
        st.rerun()
        return

    # --- Read and Display Data ---
    try:
        # Read the latest data from the log file
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

    # --- Auto-refresh logic ---
    time.sleep(5) # Refresh every 5 seconds
    st.rerun()

if __name__ == "__main__":
    main()