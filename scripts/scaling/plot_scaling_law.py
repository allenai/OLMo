#!/usr/bin/env python3

"""plot_scaling_law.py
Generate an xkcd-style plot of the scaling law:
    Loss(C) = A / C^alpha + E
The plot includes:
  • The loss curve over a logarithmic compute axis
  • A horizontal dashed line at E (irreducible loss) labeled "E"
  • Two additional dashed lines (one above and one below E) each labeled "L0 here?"
The figure is saved as 'scaling_law_plot.png' in the current directory.
"""

import numpy as np
import matplotlib.pyplot as plt


# --- Parameters ------------------------------------------------------------ #
A: float = 2e5       # Scaling coefficient
alpha: float = 0.3   # Scaling exponent
E: float = 1.0       # Irreducible loss

# Range of compute (C) values to plot (log-spaced)
C = np.logspace(18, 22, 500)  # From 1e18 to 1e22

# Compute the scaling-law loss values
loss = A / (C ** alpha) + E


# --- Plotting -------------------------------------------------------------- #
with plt.xkcd():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot the scaling law curve
    ax.plot(C, loss, label=r"$Loss(C) = A / C^{\alpha} + E$")

    # Configure axes
    ax.set_xscale("log")
    ax.set_xlabel("Compute (C)")
    ax.set_ylabel("Loss")
    ax.set_yticks([])
    ax.set_ylim(0.8, 1.8)
    ax.set_title("Scaling Law: Task Loss vs Compute")

    # Horizontal dashed line at E (irreducible loss)
    ax.axhline(E, color="gray", linestyle="--")
    ax.text(C[0], E, "  E", va="bottom", ha="left", color="gray")

    # Offset for additional dashed lines (5% of E or 0.05 if E is zero)
    offset = 0.1 * E

    # Dashed line above E
    ax.axhline(E + 3 * offset, color="teal", linestyle="--")
    ax.text(C[0], E + 3 * offset, "  L0 here?", va="bottom", ha="left", color="teal")
    ax.axhline(E + offset, color="teal", linestyle="--")
    ax.text(C[0], E + offset, "  L0 here?", va="bottom", ha="left", color="teal")

    # Dashed line below E
    ax.axhline(E - offset, color="teal", linestyle="--")
    ax.text(C[0], E - offset, "  L0 here???", va="top", ha="left", color="teal")

    # Legend and layout tweaks
    ax.legend()
    plt.tight_layout()

    # Save figure
    plt.savefig("scaling_law_plot.png", dpi=300)

    # Show plot when run interactively
    plt.show()