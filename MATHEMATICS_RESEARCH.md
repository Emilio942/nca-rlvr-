# Mathematical Optimization Roadmap: NCA-RLVR

## Open Research Question 1: Analytical Stability (Spectral Radius)
... (existing content) ...
3. Provide a formal bound on the required `NCA_HIDDEN_CHANNELS` relative to the structural complexity of the target.

## Open Research Question 3: Geometric Group Theory and SE(2) Equivariance

**Inquiry:**
"The current NCA implementation is translation-invariant due to its convolutional nature, but it lacks **Rotational Equivariance**. How can we analytically constrain the weights of the `NCA_Model_3x3` using **Geometric Group Theory** (specifically the Euclidean Group $SE(2)$) so that the update function $f$ commutes with the rotation operator $R_\theta$? 

Specifically, can we derive a mathematical basis for the $3 \times 3$ kernel weights as a linear combination of **Steerable Filters**? Can we prove that by restricting the kernels to this basis, the structure becomes 'Zero-Shot Rotationally Robust'—meaning it will grow and repair itself correctly at any arbitrary angle $\theta \in [0, 2\pi]$ without requiring any data augmentation or additional training?"

## Why this is needed
Current training requires the structure to always be in the same orientation. Solving this would:
1. Allow the NCA to grow in any direction or orientation.
2. Make the "Biological" analogy more accurate (cells don't care about 'up' or 'down' relative to a global grid).
3. Dramatically reduce the training complexity by removing the need for rotational data augmentation.
