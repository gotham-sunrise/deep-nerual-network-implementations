# ✅ LeNet-5 Architecture (1998) Overview:

- Input: 32x32 grayscale image
- C1: Convolution (6@5x5) + ReLU → 28x28x6
- S2: Average Pooling (2x2) → 14x14x6
- C3: Convolution (16@5x5) + ReLU → 10x10x16
- S4: Average Pooling (2x2) → 5x5x16
- C5: Fully Connected (120 units)
- F6: Fully Connected (84 units)
- Output: Fully Connected (10 units - digits 0 to 9)
