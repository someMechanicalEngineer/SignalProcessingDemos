# pressure_one_period.py
import sys
import numpy as np
from scipy.signal import butter, lfilter_zi, lfilter
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg

class PressureSimulator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Processing Chain Simulator")
        self.resize(1000, 700)

        # ---------------- Simulation parameters ----------------
        self.f_signal = 0.5          # sine frequency (Hz)
        self.fs_base = 1000.0        # internal high-rate sampling (Hz)
        self.noise_std = 0.05        # additive white noise
        self.v_ref = 5.0             # ADC full-scale ±Vref
        self.update_interval_ms = 50 # GUI update interval

        # Filter state
        self.filter_order = 4
        self.b, self.a = butter(self.filter_order, 20.0 / (0.5 * self.fs_base), btype='low')
        self.zi = None

        # ---------------- Build UI ----------------
        self._build_ui()

        # ---------------- Timer ----------------
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.update_interval_ms)
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start()

    # ---------------- UI ----------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # Plot
        self.plot = pg.PlotWidget(title="Signal Processing Chain Simulator")
        self.plot.showGrid(x=True, y=True)
        self.plot.setBackground('k')
        self.plot.setLabel('left', 'Amplitude', units='')
        self.plot.setLabel('bottom', 'Time', units='s')
        self.true_curve = self.plot.plot(pen=pg.mkPen(width=1.5, color='w'))
        self.proc_curve = self.plot.plot(pen=pg.mkPen(width=2, color='r'))
        layout.addWidget(self.plot, stretch=4)

        # Controls
        controls = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(controls)

        # Gain
        grid.addWidget(QtWidgets.QLabel("Amplification (gain)"), 0, 0)
        self.sl_gain = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_gain.setRange(10, 1000)
        self.sl_gain.setValue(100)
        self.lbl_gain = QtWidgets.QLabel("1.00x")
        self.sl_gain.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_gain, 0, 1)
        grid.addWidget(self.lbl_gain, 0, 2)

        # Low-pass cutoff
        grid.addWidget(QtWidgets.QLabel("Low-pass cutoff (Hz)"), 1, 0)
        self.sl_cutoff = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_cutoff.setRange(1, 2000)  # scaled ×10 → 0.1–200
        self.sl_cutoff.setValue(2000)
        self.lbl_cutoff = QtWidgets.QLabel("20.0 Hz")
        self.sl_cutoff.valueChanged.connect(self._on_cutoff_change)
        grid.addWidget(self.sl_cutoff, 1, 1)
        grid.addWidget(self.lbl_cutoff, 1, 2)

        # Capture frequency
        grid.addWidget(QtWidgets.QLabel("Capture frequency (Hz)"), 2, 0)
        self.sl_capture = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_capture.setRange(1, 5000)  # scaled ×10 → 0.1–500
        self.sl_capture.setValue(1000)
        self.lbl_capture = QtWidgets.QLabel("100.0 Hz")
        self.sl_capture.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_capture, 2, 1)
        grid.addWidget(self.lbl_capture, 2, 2)

        # Discretization
        grid.addWidget(QtWidgets.QLabel("Discretization (bits)"), 3, 0)
        self.sl_bits = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.sl_bits.setRange(4, 16)
        self.sl_bits.setValue(12)
        self.lbl_bits = QtWidgets.QLabel("12 bits")
        self.sl_bits.valueChanged.connect(self._on_slider_change)
        grid.addWidget(self.sl_bits, 3, 1)
        grid.addWidget(self.lbl_bits, 3, 2)

        layout.addWidget(controls, stretch=1)
        self._on_cutoff_change()
        self._on_slider_change()

    # ---------------- Slider helpers ----------------
    def _on_slider_change(self):
        self.lbl_gain.setText(f"{self._gain_from_slider():.2f}x")
        self.lbl_capture.setText(f"{self._capture_from_slider():.2f} Hz")
        self.lbl_bits.setText(f"{self.sl_bits.value()} bits")

    def _on_cutoff_change(self):
        cutoff = self._cutoff_from_slider()
        self.lbl_cutoff.setText(f"{cutoff:.2f} Hz")
        nyq = 0.5 * self.fs_base
        normalized = min(max(cutoff / nyq, 1e-6), 0.99)
        self.b, self.a = butter(self.filter_order, normalized, btype='low')
        self.zi = None

    def _gain_from_slider(self):
        val = self.sl_gain.value()
        return 0.1 + (val - 10) * (10.0 - 0.1) / (1000 - 10)

    def _cutoff_from_slider(self):
        val = self.sl_cutoff.value()
        return 0.1 + (val - 1) * (200 - 0.1) / (2000 - 1)

    def _capture_from_slider(self):
        val = self.sl_capture.value()
        return 0.1 + (val - 1) * (500 - 0.1) / (5000 - 1)

    # ---------------- DSP ----------------
    def _apply_lowpass(self, x):
        if x.size == 0:
            return x
        # reset filter state each update for clean per-period plot
        init_zi = lfilter_zi(self.b, self.a) * x[0]
        y, _ = lfilter(self.b, self.a, x, zi=init_zi)
        return y

    def _quantize(self, x, bits):
        levels = 2 ** bits
        step = (2 * self.v_ref) / (levels - 1)
        x_clipped = np.clip(x, -self.v_ref, self.v_ref)
        return np.round(x_clipped / step) * step

    # ---------------- Main update ----------------
    def update_simulation(self):
        T_period = 1.0 / self.f_signal
        N_points = int(self.fs_base * T_period)
        t = np.linspace(0, T_period, N_points)

        # True sine + noise
        true_sig = np.sin(2 * np.pi * self.f_signal * t)
        noisy = true_sig + np.random.normal(0, self.noise_std, N_points)

        # Apply gain, filter
        amplified = noisy * self._gain_from_slider()
        filtered = self._apply_lowpass(amplified)

        # Sample and quantize
        fs_capture = self._capture_from_slider()
        step = max(1, int(round(self.fs_base / fs_capture)))
        sampled_t = t[::step]
        quantized = self._quantize(filtered[::step], self.sl_bits.value())

        # Update plot
        self.true_curve.setData(t, true_sig)
        self.proc_curve.setData(sampled_t, quantized)
        self.plot.setXRange(0, T_period, padding=0.01)
        self.plot.setYRange(-6, 6)  # <-- fixed y-axis

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PressureSimulator()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
