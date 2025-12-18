# Methodology for stateless tunable keyed identity consistent pseudonymization

### Phase 1: Temporal stabilization
* Buffer input: The system captures a sequence of video frames containing the subject's face.
* Apply sliding window: These frames are grouped into a specific batch size (the sliding window size) to serve as a single unit of analysis.
* Average features: The system extracts facial features from every frame in the window and calculates the average. This creates a single, stable representation of the face, removing momentary sensor noise or jitter.

### Phase 2: Identity seed derivation
* Extract deep features: A biometric model converts the averaged facial representation into a complex numerical code.
* Apply fuzzy quantization: This numerical code is adjusted to the nearest point on a pre-defined coordinate grid (the quantization grid size). This ensures that slight natural variations in lighting or head angle result in the exact same code.
* Generate keyed hash: The stabilized code is combined with a secure secret key and processed through a one-way hashing function.
* Output identity seed: The result is a unique, semi-random alphanumeric string known as the identity seed. This seed is deterministic: the same person will always generate the same seed, but the seed cannot be reversed to reveal the person.

### Phase 3: Structural and contextual preparation
* Isolate structure: The system identifies facial landmarks essential for expression, specifically the jawline, nose bridge, and mouth.
* Discard identifiers: Landmarks relating to the eyes, eyebrows, and specific facial shape—which could reveal the original identity—are discarded.
* Mask context: A visual mask is created to erase the original facial skin pixels while strictly preserving the hair, neck, clothing, and background.

### Phase 4: Seeded generative inpainting
* Initialize generator: A generative diffusion model is initialized. Instead of starting with random chaos, the model's initial state is determined by the identity seed from phase 2.
* Condition generation: The model is given the structural landmarks from phase 3 to guide the pose and expression.
* Synthesize face: The model inpaints a new face into the masked area. Because the initialization is unique and constant for the subject, the generated face appears as a consistent, unique individual across all frames without needing to look up a stored avatar.

### Notes on tunable parameters
1. Sliding window size (phase 1): This controls stability versus latency. A large window averages more frames for stability but adds delay. A small window reacts instantly but risks flickering.
2. Quantization grid size (phase 2): This controls robustness versus uniqueness. A large grid makes the system robust to lighting changes but risks distinct people having the same ID. A small grid ensures uniqueness but risks the ID changing with shadow shifts.

## Implementation Details: Identity Seed Derivation

### 1. Face Detection and Alignment
Prior to feature extraction, faces must be isolated and normalized from the input image. The system handles multiple faces concurrently.
*   **Model Candidates**: RetinaFace (ResNet50 backbone) or MTCNN.
*   **Procedure**:
    1.  Detect all face bounding boxes in the image.
    2.  For each detected face:
        a.  Extract 5-point landmarks (eyes, nose, mouth).
        b.  Apply affine transformation to align the face to a standard 112x112 pixel template.

### 2. Deep Feature Extraction
This step converts each aligned face image into a latent identity vector.
*   **Model Candidates**:
    *   **ArcFace (r100-glint360k)**: Produces 512-d embeddings; highly robust to pose and age variations.
    *   **AdaFace**: Optimized for low-quality images, useful if input resolution varies.
*   **Output**: A normalized feature vector $v \in \mathbb{R}^{512}$ where $||v|| = 1$.

### 3. Quantization Logic
This step implements the "fuzzy quantization" to map continuous embeddings to discrete grid points.
*   **Algorithm**: Uniform Scalar Quantization.
*   **Parameter**: Grid size $\Delta$ (tunable hyperparameter).
*   **Operation**: For each dimension $v_i$ in the embedding vector:
    $$ q_i = \left\lfloor \frac{v_i}{\Delta} + 0.5 \right\rfloor \cdot \Delta $$
*   **Optional Pre-processing**: PCA dimensionality reduction to project 512-d vectors to a lower subspace (e.g., 128-d) before quantization to reduce sparsity.

### 4. Seed Generation
Converts the discrete code into the final identity seed.
*   **Algorithm**: HMAC-SHA256.
*   **Inputs**:
    *   Message: Byte-serialized quantized vector $q$.
    *   Key: A user-defined secret salt string.
*   **Output**: Fixed-length hexadecimal string.