**Bug Report: Redundant Training Data Loading in Simulation Loop**

*   **File(s):** `src/evolution.py`, `src/data.py`
*   **Function(s):** `specialize`, `finetune`, `get_dataloaders`
*   **Description of the Bug:** A major performance bottleneck exists in the main simulation loop. The `specialize` and `finetune` functions are called in every generation, and both functions internally call `get_dataloaders`. The `get_dataloaders` function, in turn, reloads the entire training dataset from disk every time it is called. This results in redundant and expensive disk I/O operations in each generation of the simulation.
*   **Impact:** This repeated data loading significantly slows down the entire simulation. The performance degradation is especially severe for larger datasets, where disk I/O is a primary bottleneck. It leads to wasted computational resources and unnecessarily long runtimes for experiments.
*   **Proposed Fix Strategy:**
    1.  **Cache the Training Dataset:** I will modify the `EvolutionSimulator` in `src/simulator.py` to load the training dataset only once during initialization and cache it as a class attribute (e.g., `self.train_dataset`).
    2.  **Update `get_dataloaders`:** I will refactor the `get_dataloaders` function in `src/data.py` to accept an optional, pre-loaded `train_dataset`. If the dataset is provided, the function will skip the disk loading step and use the cached object directly. It will also be updated to return the dataset it uses, so it can be cached on the first call.
    3.  **Update Evolutionary Functions:** I will update the signatures of `specialize` and `finetune` in `src/evolution.py` to accept the cached `train_dataset`.
    4.  **Pass Cached Data:** I will modify the main simulation loop in `src/simulator.py` to pass the cached `self.train_dataset` to the `specialize` and `finetune` functions, thereby eliminating the redundant I/O operations.
