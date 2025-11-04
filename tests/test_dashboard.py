import unittest
import os
import sys
from unittest.mock import patch

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from src.dashboard import show_settings_page

class TestDashboard(unittest.TestCase):
    """Contains tests for the Streamlit dashboard's reliability."""

    @patch('src.dashboard.st')
    def test_show_settings_page_handles_corrupted_yaml(self, mock_st):
        """
        Verify that show_settings_page handles a yaml.YAMLError gracefully by
        displaying an error message and not crashing.
        """
        # Configure the mock `st` object's session_state for the test
        mock_st.session_state = {}

        # Patch yaml.safe_load to simulate a corrupted YAML file
        with patch('src.dashboard.yaml.safe_load', side_effect=yaml.YAMLError("Test YAML Error")):
            # Patch 'open' to simulate the file existing, preventing FileNotFoundError
            with patch('builtins.open', unittest.mock.mock_open(read_data='- invalid yaml :')):
                show_settings_page()

        # Assert that st.error was called with a message indicating a parsing error
        mock_st.error.assert_called_once()
        self.assertIn("Error parsing `config.yaml`", mock_st.error.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
