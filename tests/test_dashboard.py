import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add the project root to the Python path to allow importing from `src`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.dashboard import show_settings_page

class TestDashboard(unittest.TestCase):
    """Unit tests for the Streamlit dashboard."""

    @patch('src.dashboard.st')
    @patch('src.dashboard.yaml')
    @patch('src.dashboard.open', new_callable=mock_open, read_data='{}')
    def test_settings_page_preserves_mate_selection_strategy(self, mock_file, mock_yaml, mock_st):
        """
        Tests that the settings page correctly preserves the user's choice
        for mate_selection_strategy across reruns.
        """
        # Arrange
        # 1. Mock the session state to simulate a pre-existing config
        initial_config = {
            'mate_selection_strategy': 'healing',
            'merge_strategy': 'average',
            'num_generations': 10
        }
        mock_st.session_state.simulation_config = initial_config
        mock_yaml.safe_load.return_value = initial_config

        # Mock st.columns to return two mock objects that can be used as context managers
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_st.columns.return_value = (mock_col1, mock_col2)

        # 2. Mock the return value of the selectbox to simulate a user selection
        def selectbox_side_effect(*args, **kwargs):
            # The first call is for merge_strategy, the second for mate_selection_strategy
            if 'Mate Selection Strategy' in args:
                return 'healing' # Simulate user selecting 'healing'
            return 'average' # Default for other selectboxes

        mock_st.selectbox.side_effect = selectbox_side_effect
        mock_st.number_input.return_value = 10
        mock_st.slider.return_value = 0.5
        mock_st.button.return_value = True # Simulate user clicking 'Update'

        # Act
        show_settings_page()

        # Assert
        # Check that st.selectbox for mate selection was called with the correct index
        # This confirms the UI correctly reflects the loaded state.
        mate_strategy_call = None
        for call in mock_st.selectbox.call_args_list:
            if 'Mate Selection Strategy' in call.args:
                mate_strategy_call = call
                break

        self.assertIsNotNone(mate_strategy_call, "selectbox for Mate Selection Strategy was not called.")
        self.assertEqual(
            mate_strategy_call.kwargs['index'],
            0, # 'healing' is at index 0
            "The selectbox was not initialized with the correct index for 'healing'."
        )

if __name__ == '__main__':
    unittest.main()