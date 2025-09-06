from manim import (
    Scene,
    Axes,
    BLUE,
    Text,
    UP,
    LEFT,
    VGroup,
    RIGHT,
    Circle,
    WHITE,
    Arrow,
    YELLOW,
    TracedPath,
    RED,
    ValueTracker,
    Write,
    linear,
    PI,
    GREEN
)
import numpy as np

class NutsFourierScene(Scene):
    def construct(self):
        # 1. Convert "Nuts" to a binary signal
        text = "Nuts"
        binary_string = ''.join(format(ord(char), '08b') for char in text)
        signal = [1 if bit == '1' else -1 for bit in binary_string]
        
        # Define the periodic function for the Fourier series
        def f(t):
            # t is expected to be in [0, 1]
            index = int(t * len(signal)) % len(signal)
            return signal[index]

        # 2. Calculate Fourier Series Coefficients
        n_terms = 50  # Number of harmonics to use
        coeffs = []
        
        # We perform numerical integration to get the coefficients
        T = 1.0  # Period
        w0 = 2 * PI / T
        dt = T / 10000
        t_samples = np.arange(0, T, dt)
        f_samples = np.array([f(t) for t in t_samples])

        # Calculate a0
        a0 = (1 / T) * np.sum(f_samples * dt)

        for n in range(1, n_terms + 1):
            # an and bn coefficients
            an = (2 / T) * np.sum(f_samples * np.cos(n * w0 * t_samples) * dt)
            bn = (2 / T) * np.sum(f_samples * np.sin(n * w0 * t_samples) * dt)
            coeffs.append((an, bn))

        # 3. Manim Animation Setup
        axes = Axes(
            x_range=[0, 2.5, 0.5],
            y_range=[-2, 2, 1],
            x_length=8,
            y_length=4,
            axis_config={"color": BLUE},
        ).to_edge(RIGHT)

        # Draw the target waveform ("Nuts" signal)
        target_graph = axes.plot(lambda x: f(x / 2.5), x_range=[0, 2.5], color=GREEN)
        target_label = Text("Target Waveform: 'Nuts'").next_to(target_graph, UP, buff=0.2).scale(0.5)

        # Setup for the circles and vectors
        circles_origin = LEFT * 4
        circles = VGroup()
        vectors = VGroup()
        
        # Add the DC offset (a0) as the starting point
        current_center = circles_origin + RIGHT * a0
        
        # Create circles and vectors for each harmonic
        for n, (an, bn) in enumerate(coeffs):
            k = n + 1
            radius = np.sqrt(an**2 + bn**2)
            phase = np.arctan2(bn, an)

            circle = Circle(radius=radius, color=WHITE, stroke_width=2).move_to(current_center)
            
            # Start vector from the center of the circle
            vector = Arrow(
                start=circle.get_center(),
                end=circle.get_right(),
                buff=0,
                color=YELLOW,
                stroke_width=3
            )
            vector.rotate(phase, about_point=circle.get_center())
            
            circles.add(circle)
            vectors.add(vector)

            # The center of the next circle is the tip of the current vector
            current_center = vector.get_end()

        # Path traced by the tip of the last vector
        path = VGroup()
        path.add(TracedPath(vectors[-1].get_end, stroke_color=RED, stroke_width=3))

        # Time tracker
        time = ValueTracker(0)

        # Updater functions for animation
        def update_vectors(vecs):
            last_end = circles_origin + RIGHT * a0
            for i, vec in enumerate(vecs):
                k = i + 1
                an, bn = coeffs[i]
                phase = np.arctan2(bn, an)

                circle = circles[i]
                circle.move_to(last_end)
                
                vec.put_start_and_end_on(last_end, last_end + RIGHT * (circle.width / 2))
                vec.rotate(phase + time.get_value() * k * w0, about_point=last_end)
                
                last_end = vec.get_end()

        vectors.add_updater(update_vectors)

        # Add all elements to the scene
        self.add(axes, target_graph, target_label)
        self.add(circles, vectors)
        self.add(path)

        # Display the text being encoded
        text_label = Text(f"Encoding: '{text}'").to_edge(UP)
        self.play(Write(text_label))

        # Animate over one full period
        self.play(time.animate.set_value(T), run_time=15, rate_func=linear)
        
        vectors.clear_updaters()
        self.wait()