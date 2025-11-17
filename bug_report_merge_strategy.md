**Bug Report: Critical Performance Bottleneck in `SequentialConstructiveMergeStrategy`**

*   **File:** `src/merge_strategies.py`
*   **Class:** `SequentialConstructiveMergeStrategy`
*   **Description of the Bug:** The `SequentialConstructiveMergeStrategy` is a major performance bottleneck in the simulation. Its current implementation iterates through every layer of the neural network and performs a full validation pass for each layer to decide whether to swap it. For a ResNet model, this can result in over 50 separate, expensive validation runs *for a single merge operation*. This iterative, layer-by-layer evaluation is computationally prohibitive and dramatically slows down the entire evolutionary process.
*   **Impact:** The high computational cost of this merge strategy makes it impractical for real-world use, especially with deep learning models. It significantly increases the time required to complete a simulation, hindering rapid experimentation and analysis.
*   **Proposed Fix Strategy:**
    1.  **Replace with a Heuristic:** I will replace the iterative, layer-by-layer evaluation with a much faster heuristic-based approach. The new strategy will create a single hybrid model by swapping a random half of the layers from the weaker parent and then perform only **one** validation pass to decide whether the hybrid or the original fitter parent is better.
    2.  **Rename for Clarity:** I will rename the class from `SequentialConstructiveMergeStrategy` to `RandomHalfMergeStrategy` to accurately reflect its new, non-sequential, and heuristic-based nature.
    3.  **Maintain Compatibility:** I will update the `strategy_map` in `src/evolution.py` to ensure that the `'sequential_constructive'` key (used in `config.yaml`) maps to the new `RandomHalfMergeStrategy`. This will ensure that existing configurations continue to work seamlessly while benefiting from the massive performance boost.
