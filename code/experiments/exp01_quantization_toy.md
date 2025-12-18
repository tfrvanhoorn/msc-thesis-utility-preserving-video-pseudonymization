# Experiment 01: Quantization toy

## Objective
Verify effectiveness of tunable quantization grid on identity stability and uniqueness using static images.

## Dataset
- Source: CelebA?.
- Composition: X unique identities.
- Redundancy: Multiple images per identity (varying lighting, slight pose).
- Total size: N images.

## Pipeline
1. Face detection & alignment (RetinaFace).
2. Embedding extraction (ArcFace, 512-d).
3. Quantization (uniform scalar) with variable grid size (delta).
4. Seed generation (HMAC-SHA256).

## Variables
- Independent: Quantization grid size (delta).
- Dependent:
  - Consistency (intra-class).
  - Uniqueness (inter-class).

## Metrics
- Consistency rate: Percentage of same-identity pairs producing identical seeds.
- Collision rate: Percentage of different-identity pairs producing identical seeds.

## Hypothesis
- Large delta: High consistency, high collision risk.
- Small delta: Low consistency, low collision risk.
- Optimal delta: Maximizes consistency while minimizing collisions.
