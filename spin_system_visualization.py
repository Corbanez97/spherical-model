"""
Spin System Visualization with Manim

This script creates an animation of the SpinSystem evolution, showing:
1. Nodes arranged in a circle representing spins
2. Node values (spin values) displayed as text
3. Connections between nodes with opacity based on the J matrix
4. Evolution of the system over optimization steps

Requirements:
- manim (pip install manim)
- tensorflow (pip install tensorflow)
- numpy (pip install numpy)
"""

import tensorflow as tf
import numpy as np
from manim import *

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


class SpinSystem:
    """A class representing a system of spins with arbitrary interactions."""

    def __init__(self, N, J=None):
        self.N = N

        # Create or validate the interaction matrix
        if J is None:
            # Create a random symmetric matrix with zero diagonal
            J = tf.random.normal([N, N], stddev=1.0)
            J = (J + tf.transpose(J)) / 2  # Make symmetric
            J = tf.linalg.set_diag(J, tf.zeros(N))  # No self-interactions
        else:
            # Convert to tensor if needed and validate shape
            J = tf.convert_to_tensor(J, dtype=tf.float32)
            if J.shape != [N, N]:
                raise ValueError(f"J must be of shape [{N}, {N}]")

        self.J = J

        # Initialize spins with spherical constraint
        self.sigma = tf.Variable(self._initialize_spins(), trainable=True)

    def _initialize_spins(self):
        """Initialize spins with spherical constraint."""
        sigma = tf.random.normal([self.N])
        return self._apply_spherical_constraint(sigma)

    def _apply_spherical_constraint(self, sigma):
        """
        Apply spherical constraint to spin vector.

        Returns:
            Normalized spin vector satisfying ∑σ_i² = N
        """
        return tf.sqrt(tf.cast(self.N, tf.float32)) * tf.math.l2_normalize(sigma)

    def hamiltonian(self):
        """Compute the Hamiltonian H = Σ_{i,j} J_{ij} σ_i σ_j."""
        return tf.einsum('i,ij,j->', self.sigma, self.J, self.sigma)

    def optimize_step(self, learning_rate=0.01):
        """Perform one optimization step."""
        with tf.GradientTape() as tape:
            energy = self.hamiltonian()

        # Compute gradients - FIXED: Get gradient for self.sigma directly
        gradients = tape.gradient(energy, self.sigma)

        # Apply gradient descent - FIXED: gradients is now a tensor, not a list
        if gradients is not None:
            self.sigma.assign_sub(learning_rate * gradients)

        # Reapply spherical constraint after update
        self.sigma.assign(self._apply_spherical_constraint(self.sigma))

        return energy.numpy()


class SpinSystemVisualization(Scene):
    """Manim scene to visualize the SpinSystem evolution."""

    def construct(self):
        # System parameters
        N = 12  # Number of spins (smaller for visualization)
        steps = 10  # Number of optimization steps

        # Create spin system
        system = SpinSystem(N)

        # Get the J matrix for connection opacities
        J_matrix = system.J.numpy()
        max_abs_J = np.max(np.abs(J_matrix))

        # Create a circle of nodes
        radius = 2
        nodes = VGroup()
        node_positions = []
        node_circles = []
        node_texts = []

        for i in range(N):
            angle = 2 * PI * i / N
            x = radius * np.cos(angle)
            y = radius * np.sin(angle) - 1
            position = np.array([x, y, 0])
            node_positions.append(position)

            # Circle
            circle = Circle(radius=2/N, color=WHITE, fill_opacity=0.8)
            circle.move_to(position)
            node_circles.append(circle)

            # Initial value as Text
            value = system.sigma.numpy()[i]
            text = Text(f"{value:.2f}", font_size=100/N, color=BLACK)
            text.move_to(position)
            node_texts.append(text)

            nodes.add(circle, text)

        # Connections
        connections = VGroup()
        for i in range(N):
            for j in range(i + 1, N):
                if abs(J_matrix[i, j]) > 0.01:
                    opacity = min(1.0, abs(J_matrix[i, j]) / max_abs_J)
                    color = RED if J_matrix[i, j] > 0 else BLUE
                    line = Line(
                        node_positions[i],
                        node_positions[j],
                        stroke_width=2,
                        color=color,
                        stroke_opacity=opacity,
                    )
                    connections.add(line)

        # Title
        title = Text("Spin System Evolution", font_size=36)
        title.to_edge(UP, buff=0.5)

        # Energy and step labels
        energy_text = Text(
            f"Energy: {system.hamiltonian().numpy():.4f}", font_size=24)
        energy_text.next_to(title, DOWN, buff=0.3)

        step_text = Text("Step: 0", font_size=24)
        step_text.next_to(energy_text, DOWN, buff=0.3)

        # Add to scene
        self.play(Write(title))
        self.play(Write(energy_text), Write(step_text))
        self.play(Create(nodes), Create(connections))
        self.wait(1)

        # Optimization steps
        for step in range(steps):
            energy = system.optimize_step(learning_rate=0.1)

            animations = []

            # Update node values with ReplacementTransform
            new_node_texts = []
            for i in range(N):
                value = system.sigma.numpy()[i]
                new_text = Text(f"{value:.2f}", font_size=100/N, color=BLACK)
                new_text.move_to(node_positions[i])
                animations.append(ReplacementTransform(
                    node_texts[i], new_text))
                new_node_texts.append(new_text)
            node_texts = new_node_texts

            # Update energy text
            new_energy_text = Text(f"Energy: {energy:.4f}", font_size=24)
            new_energy_text.next_to(title, DOWN, buff=0.3)
            animations.append(ReplacementTransform(
                energy_text, new_energy_text))
            energy_text = new_energy_text

            # Update step text
            new_step_text = Text(f"Step: {step+1}", font_size=24)
            new_step_text.next_to(energy_text, DOWN, buff=0.3)
            animations.append(ReplacementTransform(step_text, new_step_text))
            step_text = new_step_text

            self.play(*animations, run_time=0.5)
            self.wait(0.1)

        self.wait(2)


if __name__ == "__main__":
    # For testing without Manim
    system = SpinSystem(3)
    print("Initial energy:", system.hamiltonian().numpy())

    for step in range(5):
        energy = system.optimize_step()
        print(f"Step {step}: Energy = {energy:.4f}")
