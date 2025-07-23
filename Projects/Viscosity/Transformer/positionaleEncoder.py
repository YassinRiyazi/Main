import numpy as np
import matplotlib.pyplot as plt

def get_2d_sinusoidal_positional_encoding(height, width, dim):
    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4")

    pe = np.zeros((height, width, dim))
    d_model = dim // 2

    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pos_y = np.arange(height)[:, np.newaxis]
    pos_x = np.arange(width)[:, np.newaxis]

    pe_y = np.zeros((height, d_model))
    pe_x = np.zeros((width, d_model))

    pe_y[:, 0::2] = np.sin(pos_y * div_term)
    pe_y[:, 1::2] = np.cos(pos_y * div_term)
    pe_x[:, 0::2] = np.sin(pos_x * div_term)
    pe_x[:, 1::2] = np.cos(pos_x * div_term)

    for i in range(height):
        for j in range(width):
            pe[i, j] = np.concatenate([pe_y[i], pe_x[j]])

    return pe.reshape(height * width, dim)

# Parameters
height, width, dim = 170, 200, 128
pos_enc = get_2d_sinusoidal_positional_encoding(height, width, dim)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(pos_enc, aspect='auto', cmap='viridis')
plt.colorbar(label="Encoding value")
plt.title(f"2D Sinusoidal Positional Encoding (Flattened {height}x{width} grid, dim={dim})")
plt.xlabel("Encoding Dimension")
plt.ylabel("Patch Index")
plt.tight_layout()
plt.show()
