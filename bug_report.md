**Bug Report: Redundant Evaluation Pass in Simulation Loop**

*   **File(s):** `src/simulator.py`, `src/evolution.py`
*   **Line Number(s):** `_run_evaluation_phase` and `_run_evolution_phase` in `src/simulator.py`; `select_mates` in `src/evolution.py`.
*   **Description of the Bug:** The current simulation workflow contains a critical performance bottleneck. In each generation, the `_run_evaluation_phase` iterates over the entire test dataset to calculate the overall fitness for each model. Immediately after, the `_run_evolution_phase` calls `select_mates`, which in turn calls `evaluate_by_class`. This function then iterates over the *exact same* test dataset a second time to calculate per-class accuracies. This results in two full, expensive passes over the data when one would suffice.
*   **Impact:** This redundancy nearly doubles the evaluation time per generation, significantly slowing down the entire simulation. The impact is magnified with larger datasets, more complex models, or a larger population size.
*   **Proposed Fix Strategy:**
    1.  **Consolidate Metric Calculation:** Create a new function, `_calculate_metrics`, in `src/utils.py` that computes both overall accuracy and per-class accuracy in a single pass.
    2.  **Cache Per-Class Fitness:** Add a `per_class_fitness` attribute to the `ModelWrapper` class in `src/model_wrapper.py` to cache the per-class results, mirroring the existing `fitness` cache.
    3.  **Update Evaluation Phase:** Modify `_run_evaluation_phase` in `src/simulator.py` to call the new consolidated function, populating both the overall and per-class fitness caches for each model in one go.
    4.  **Refactor Mate Selection:** Modify `select_mates` in `src/evolution.py` to use the now-cached `per_class_fitness` from the model wrapper, completely eliminating the call to `evaluate_by_class` and the redundant evaluation pass.
