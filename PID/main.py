# pid_simulator_largePI.py
import sys
import numpy as np
from collections import deque
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

class PIDSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PID Control Simulator")
        self.resize(1000, 700)

        # ---------------- Simulation parameters ----------------
        self.fs = 1000.0           # sampling frequency (Hz)
        self.dt = 1.0 / self.fs
        self.window_duration = 0.2 # seconds for rolling window
        self.f_signal = 0.5        # sine wave frequency (Hz)
        self.noise_std = 0.05
        self.output_min = -6.0
        self.output_max = 6.0

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        # Buffers
        self.N_window = int(self.fs * self.window_duration)
        self.time_buf = deque(maxlen=self.N_window)
        self.input_buf = deque(maxlen=self.N_window)
        self.output_buf = deque(maxlen=self.N_window)

        # Delay buffer for actuator
        self.delay_steps = 0
        self.delay_buffer = deque([0.0], maxlen=1)

        # ---------------- Build UI ----------------
        self._build_ui()

        # ---------------- Timer ----------------
        self.timer = QtCore.QTimer()
        self.timer.setInterval(20)  # ms
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Plot
        self.plot = pg.PlotWidget(title="PID Control Simulator")
        self.plot.showGrid(x=True, y=True)
        self.plot.setBackground('k')
        self.plot.setLabel('left', 'Amplitude')
        self.plot.setLabel('bottom', 'Time [s]')
        self.plot.setYRange(-6, 6)
        self.input_curve = self.plot.plot(pen=pg.mkPen(width=1.5, color='w'))
        self.output_curve = self.plot.plot(pen=pg.mkPen(width=2, color='r'))
        layout.addWidget(self.plot, stretch=4)

        # Controls
        controls = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(controls)

        # P gain
        grid.addWidget(QtWidgets.QLabel("P Gain"), 0, 0)
        self.sl_p = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_p.setRange(0, 1000)
        self.sl_p.setValue(50)
        self.lbl_p = QtWidgets.QLabel("2.0")
        self.sl_p.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_p, 0, 1)
        grid.addWidget(self.lbl_p, 0, 2)

        # I gain
        grid.addWidget(QtWidgets.QLabel("I Gain"), 1, 0)
        self.sl_i = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_i.setRange(0, 1000)
        self.sl_i.setValue(0)
        self.lbl_i = QtWidgets.QLabel("1.0")
        self.sl_i.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_i, 1, 1)
        grid.addWidget(self.lbl_i, 1, 2)

        # D gain
        grid.addWidget(QtWidgets.QLabel("D Gain"), 2, 0)
        self.sl_d = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_d.setRange(0, 1000)
        self.sl_d.setValue(0)
        self.lbl_d = QtWidgets.QLabel("0.01")
        self.sl_d.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_d, 2, 1)
        grid.addWidget(self.lbl_d, 2, 2)

        # Setpoint
        grid.addWidget(QtWidgets.QLabel("Setpoint"), 3, 0)
        self.sl_setpoint = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_setpoint.setRange(-60, 60)
        self.sl_setpoint.setValue(0)
        self.lbl_setpoint = QtWidgets.QLabel("0.0")
        self.sl_setpoint.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_setpoint, 3, 1)
        grid.addWidget(self.lbl_setpoint, 3, 2)

        # Actuator Delay
        grid.addWidget(QtWidgets.QLabel("Actuator Delay (ms)"), 4, 0)
        self.sl_delay = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_delay.setRange(0, 200)
        self.sl_delay.setValue(0)
        self.lbl_delay = QtWidgets.QLabel("0 ms")
        self.sl_delay.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_delay, 4, 1)
        grid.addWidget(self.lbl_delay, 4, 2)

        layout.addWidget(controls, stretch=1)
        self._on_slider_change()

    # ---------------- Slider helpers ----------------
    def _on_slider_change(self):
        self.lbl_p.setText(f"{self._p_gain():.3f}")
        self.lbl_i.setText(f"{self._i_gain():.3f}")
        self.lbl_d.setText(f"{self._d_gain():.3f}")
        self.lbl_setpoint.setText(f"{self._setpoint():.2f}")
        self.lbl_delay.setText(f"{self.sl_delay.value()} ms")

        # Update delay buffer length dynamically
        self.delay_steps = int(self.sl_delay.value() / 1000.0 * self.fs)
        if self.delay_steps < 1:
            self.delay_steps = 1
        current_values = list(self.delay_buffer)
        self.delay_buffer = deque(current_values, maxlen=self.delay_steps)

    def _p_gain(self):
        return self.sl_p.value() * 20.0 / 1000.0  # 0–20

    def _i_gain(self):
        return self.sl_i.value() * 20.0 / 1000.0  # 0–20

    def _d_gain(self):
        return self.sl_d.value() / 1000.0  # 0–1

    def _setpoint(self):
        return self.sl_setpoint.value() / 10.0

    # ---------------- Main simulation ----------------
    def update_simulation(self):
        # Generate input sine
        t = self.time_buf[-1] + self.dt if self.time_buf else 0.0
        sine_val = np.sin(2 * np.pi * self.f_signal * t)
        noisy_input = sine_val + np.random.normal(0, 0.05)

        # PID computation
        error = self._setpoint() - noisy_input
        derivative = (error - self.prev_error) / self.dt
        pid_unsat = self._p_gain() * error + self._i_gain() * self.integral + self._d_gain() * derivative

        # Anti-windup: only integrate if output not saturated
        if self.output_min < pid_unsat + noisy_input < self.output_max:
            self.integral += error * self.dt

        # Compute PID output
        pid_output = self._p_gain() * error + self._i_gain() * self.integral + self._d_gain() * derivative

        # --- Actuator delay ---
        self.delay_buffer.append(pid_output)
        delayed_output = self.delay_buffer[0]  # oldest value
        controlled = np.clip(noisy_input + delayed_output, self.output_min, self.output_max)

        self.prev_error = error

        # Store in buffers
        self.time_buf.append(t)
        self.input_buf.append(noisy_input)
        self.output_buf.append(controlled)

        # Keep rolling window
        while self.time_buf and self.time_buf[0] < t - self.window_duration:
            self.time_buf.popleft()
            self.input_buf.popleft()
            self.output_buf.popleft()

        # Update plot
        x = np.array(self.time_buf)
        self.input_curve.setData(x, np.array(self.input_buf))
        self.output_curve.setData(x, np.array(self.output_buf))
        self.plot.setXRange(max(0, t - self.window_duration), t)
        self.plot.setYRange(-6, 6)

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PIDSimulator()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
