import time
import json
from playwright.sync_api import sync_playwright, expect

def run_verification(page):
    """
    This script verifies the interactive controls on the Streamlit dashboard
    using robust data-testid locators.
    """
    # 1. Go to the dashboard URL.
    page.goto("http://localhost:8501")

    # 2. Wait for the sidebar controls to be visible.
    expect(page.get_by_text("Live Simulation Controls")).to_be_visible(timeout=30000)

    # 3. Interact with the merge strategy dropdown using data-testid.
    # First, click the selectbox to open the dropdown options.
    page.locator('[data-testid="stSelectbox"]').click()
    # Then, click the desired option.
    page.get_by_role("option", name="fitness_weighted").click()

    # 4. Interact with the mutation rate slider.
    # Click on the slider rail at the 75% position to set the value.
    slider = page.locator('[data-testid="stSlider"]')
    slider_bounding_box = slider.bounding_box()

    if slider_bounding_box:
        # Calculate the x-coordinate for the 75% mark
        target_x = slider_bounding_box['x'] + slider_bounding_box['width'] * 0.75
        target_y = slider_bounding_box['y'] + slider_bounding_box['height'] / 2
        page.mouse.click(target_x, target_y)

    # 5. Wait for the changes to be written to the file.
    time.sleep(2)

    # 6. Take a screenshot to verify the final UI state.
    page.screenshot(path="jules-scratch/verification/interactive_dashboard.png")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            run_verification(page)
            print("Verification script completed successfully.")
        except Exception as e:
            print(f"An error occurred during verification: {e}")
            page.screenshot(path="jules-scratch/verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    main()