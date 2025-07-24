
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QRadioButton, QButtonGroup, QLabel, QLineEdit, QPushButton, QSlider, QComboBox,
    QMessageBox, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class DiscreteSignal:
    def __init__(self, values, start):
        self.values = np.array(values)
        self.start = int(start)

    def time_indices(self):
        return np.arange(self.start, self.start + len(self.values))

class ContinuousSignal:
    def __init__(self, kind, t_range, amp):
        self.kind = kind
        self.t_range = t_range
        self.amp = amp

def parse_discrete_signal(values_str, start_str):
    try:
        values = [float(v.strip()) for v in values_str.split(',')]
        start = int(start_str)
        return DiscreteSignal(values, start)
    except (ValueError, IndexError):
        raise ValueError("Invalid discrete signal input. Use comma-separated numbers for values and an integer for the start index.")

def parse_continuous_signal(kind, t0_str, t1_str, amp_str):
    try:
        t0 = float(t0_str)
        t1 = float(t1_str) if t1_str else t0 + 1.0
        amp = float(amp_str)
        return ContinuousSignal(kind, (t0, t1), amp)
    except ValueError:
        raise ValueError("Invalid continuous signal parameters. Use numbers for t0, t1, and amplitude.")

def discrete_convolve(x, h):
    y_vals = np.convolve(x.values, h.values, mode='full')
    y_start = x.start + h.start
    return DiscreteSignal(y_vals, y_start)
# --- End of Dummy Implementations ---


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Convolution & Correlation GUI")
        self._setup_ui()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self.discrete = True
        
        self.xi = self.hi = None
        self.n_full = self.y_full = None
        self.frame_idx = 0
        self.shift_positions = None
        self.output_values = None
        self.unified_xlim_left = 0.0
        self.unified_xlim_right = 0.0
        self.dt = 0.005
        self.is_correlation_mode = False

    def _setup_ui(self):
        container = QWidget()
        main_layout = QVBoxLayout(container)
        
        mode_layout = QHBoxLayout()
        self.discrete_rb = QRadioButton("Discrete")
        self.continuous_rb = QRadioButton("Continuous")
        self.discrete_rb.setChecked(True)
        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.discrete_rb)
        mode_layout.addWidget(self.continuous_rb)
        group = QButtonGroup(self)
        group.addButton(self.discrete_rb)
        group.addButton(self.continuous_rb)
        self.discrete_rb.toggled.connect(self._on_mode)
        main_layout.addLayout(mode_layout)
        
        self.stack_inputs = QStackedWidget()
        self.stack_inputs.addWidget(self._make_discrete_panel("x"))
        self.stack_inputs.addWidget(self._make_continuous_panel("x"))
        main_layout.addWidget(QLabel("Input Signal:"))
        main_layout.addWidget(self.stack_inputs)
        
        self.stack_impulses = QStackedWidget()
        self.stack_impulses.addWidget(self._make_discrete_panel("h"))
        self.stack_impulses.addWidget(self._make_continuous_panel("h"))
        main_layout.addWidget(QLabel("Impulse Response / Second Signal:"))
        main_layout.addWidget(self.stack_impulses)
        
        ctrl_layout = QHBoxLayout()
        self.compute_btn = QPushButton("Convolution")
        self.reset_btn = QPushButton("Reset")
        self.corr_btn = QPushButton("Correlation")
        ctrl_layout.addWidget(self.compute_btn)
        ctrl_layout.addWidget(self.corr_btn)
        ctrl_layout.addWidget(self.reset_btn)
        
        ctrl_layout.addWidget(QLabel("FPS:"))
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setRange(1, 120) 
        self.fps_slider.setValue(30)
        ctrl_layout.addWidget(self.fps_slider)
        
        ctrl_layout.addWidget(QLabel("Frame Skip:"))
        self.frame_skip_edit = QLineEdit("1") 
        self.frame_skip_edit.setFixedWidth(40)
        ctrl_layout.addWidget(self.frame_skip_edit)
        
        main_layout.addLayout(ctrl_layout)
        
        self.fig = Figure(figsize=(12, 8))
        self.ax1 = self.fig.add_subplot(311)
        self.ax2 = self.fig.add_subplot(312)
        self.ax3 = self.fig.add_subplot(313)
        self.canvas = FigureCanvasQTAgg(self.fig)
        main_layout.addWidget(self.canvas)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        main_layout.addWidget(toolbar)
        
        self.compute_btn.clicked.connect(self._on_compute)
        self.reset_btn.clicked.connect(self._on_reset)
        self.corr_btn.clicked.connect(self._on_correlation)
        self.fps_slider.valueChanged.connect(self._on_fps_change)
        self.setCentralWidget(container)
        self._on_mode()

    def _make_discrete_panel(self, prefix: str) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        setattr(self, f"{prefix}_values_edit", QLineEdit())
        setattr(self, f"{prefix}_start_edit", QLineEdit())
        
        if prefix == "x":
            getattr(self, f"{prefix}_values_edit").setText("1,2,3,1")
            getattr(self, f"{prefix}_start_edit").setText("0")
        else:
            getattr(self, f"{prefix}_values_edit").setText("1,1,1")
            getattr(self, f"{prefix}_start_edit").setText("0")
        
        layout.addWidget(QLabel("Values (comma-separated):"), 0, 0)
        layout.addWidget(getattr(self, f"{prefix}_values_edit"), 0, 1)
        layout.addWidget(QLabel("Start index (n0):"), 1, 0)
        layout.addWidget(getattr(self, f"{prefix}_start_edit"), 1, 1)
        return w

    def _make_continuous_panel(self, prefix: str) -> QWidget:
        w = QWidget()
        layout = QGridLayout(w)
        
        setattr(self, f"{prefix}_kind_combo", QComboBox())
        combo = getattr(self, f"{prefix}_kind_combo")
        combo.addItems(["rectangular", "triangular", "step", "impulse", "sawtooth"])
        combo.currentTextChanged.connect(lambda: self._on_signal_type_change(prefix))
        
        setattr(self, f"{prefix}_t0_edit", QLineEdit())
        setattr(self, f"{prefix}_t1_edit", QLineEdit())
        setattr(self, f"{prefix}_amp_edit", QLineEdit())
        
        setattr(self, f"{prefix}_t1_label", QLabel("t1:"))
        
        getattr(self, f"{prefix}_t0_edit").setText("0")
        getattr(self, f"{prefix}_t1_edit").setText("2")
        getattr(self, f"{prefix}_amp_edit").setText("1")
        
        layout.addWidget(QLabel("Type:"), 0, 0)
        layout.addWidget(combo, 0, 1)
        layout.addWidget(QLabel("t0:"), 1, 0)
        layout.addWidget(getattr(self, f"{prefix}_t0_edit"), 1, 1)
        layout.addWidget(getattr(self, f"{prefix}_t1_label"), 2, 0)
        layout.addWidget(getattr(self, f"{prefix}_t1_edit"), 2, 1)
        layout.addWidget(QLabel("Amplitude:"), 3, 0)
        layout.addWidget(getattr(self, f"{prefix}_amp_edit"), 3, 1)
        
        self._update_t1_visibility(prefix)
        
        return w

    def _on_signal_type_change(self, prefix: str):
        self._update_t1_visibility(prefix)

    def _update_t1_visibility(self, prefix: str):
        combo = getattr(self, f"{prefix}_kind_combo")
        t1_label = getattr(self, f"{prefix}_t1_label")
        t1_edit = getattr(self, f"{prefix}_t1_edit")
        
        signal_type = combo.currentText()
        
        if signal_type in ["impulse", "step"]:
            t1_label.hide()
            t1_edit.hide()
            t1_edit.setText("")
        else:
            t1_label.show()
            t1_edit.show()
            if not t1_edit.text():
                t1_edit.setText("2")

    def _on_mode(self):
        if self.discrete_rb.isChecked():
            self.stack_inputs.setCurrentIndex(0)
            self.stack_impulses.setCurrentIndex(0)
            self.discrete = True
        else:
            self.stack_inputs.setCurrentIndex(1)
            self.stack_impulses.setCurrentIndex(1)
            self.discrete = False
            self._update_t1_visibility("x")
            self._update_t1_visibility("h")

    def _on_fps_change(self):
        if self.timer.isActive():
            fps = self.fps_slider.value()
            self.timer.start(int(1000/fps))

    def _create_synchronized_animation(self):
        if self.discrete:
            xn = self.xi.time_indices()
            hn = self.hi.time_indices()
            x_start, x_end = xn[0], xn[-1]
            h_start, h_end = hn[0], hn[-1]
            y_start, y_end = self.n_full[0], self.n_full[-1]

            signal_start = min(x_start, h_start, y_start)
            signal_end = max(x_end, h_end, y_end)
            center = (signal_start + signal_end) / 2.0
            slide_range_width = (x_end - x_start) + (h_end - h_start) + 5
        else:
            x_t0, x_t1 = self.xi.t_range
            h_t0, h_t1 = self.hi.t_range
            y_start = self.n_full[0]
            y_end = self.n_full[-1]

            signal_start = min(x_t0, h_t0, y_start)
            signal_end = max(x_t1, h_t1, y_end)
            center = (signal_start + signal_end) / 2.0
            slide_range_width = (x_t1 - x_t0) + (h_t1 - h_t0) + 5.0
        
        slide_start = center - slide_range_width
        slide_end = center + slide_range_width
        
        padding = 2.0
        self.unified_xlim_left = slide_start - padding
        self.unified_xlim_right = slide_end + padding
        
        range_span = abs(slide_end - slide_start)
        num_frames = min(400, max(150, int(range_span * 10)))
        self.shift_positions = np.linspace(slide_start, slide_end, num_frames)
        
        self.output_values = np.interp(self.shift_positions, self.n_full, self.y_full)

    def _parse_continuous_signal(self, prefix: str):
        kind = getattr(self, f"{prefix}_kind_combo").currentText()
        t0 = getattr(self, f"{prefix}_t0_edit").text()
        t1 = getattr(self, f"{prefix}_t1_edit").text()
        amp = getattr(self, f"{prefix}_amp_edit").text()
        
        if kind in ["step", "impulse"]:
            t1 = str(float(t0) + 1.0) if t0 else "1.0"
            
        return parse_continuous_signal(kind, t0, t1, amp)

    def _calculate_correlation(self, x, h):
        corr_vals = np.correlate(x.values, h.values, mode='full')
        h_end = h.start + len(h.values) - 1
        corr_start = x.start - h_end
        return DiscreteSignal(corr_vals, corr_start)

    def _on_compute(self):
        self._run_process(is_correlation=False)

    def _on_correlation(self):
    
        
        self._run_process(is_correlation=True)
       

    def _run_process(self, is_correlation: bool):
        self.timer.stop()
        self.is_correlation_mode = is_correlation
        try:
            if self.discrete:
                self.xi = parse_discrete_signal(self.x_values_edit.text(), self.x_start_edit.text())
                self.hi = parse_discrete_signal(self.h_values_edit.text(), self.h_start_edit.text())
                
                if self.is_correlation_mode:
                    result = self._calculate_correlation(self.xi, self.hi)
                else: # Convolution
                    result = discrete_convolve(self.xi, self.hi)
                
                self.n_full, self.y_full = result.time_indices(), result.values
            else: # Continuous
                self.xi = self._parse_continuous_signal("x")
                self.hi = self._parse_continuous_signal("h")
                self.master_t = np.arange(-20, 20, self.dt)
                def sample_on_grid(signal, t_grid):
                    y = np.zeros_like(t_grid)
                    t0, t1 = signal.t_range
                    amp = signal.amp
                    kind = signal.kind
                    if kind == 'rectangular': y[(t_grid >= t0) & (t_grid <= t1)] = amp
                    elif kind == 'step': y[t_grid >= t0] = amp
                    elif kind == 'triangular':
                        mid = (t0 + t1) / 2.0
                        mask1 = (t_grid >= t0) & (t_grid < mid)
                        mask2 = (t_grid >= mid) & (t_grid <= t1)
                        if mid > t0: y[mask1] = amp * (t_grid[mask1] - t0) / (mid - t0)
                        if t1 > mid: y[mask2] = amp * (t1 - t_grid[mask2]) / (t1 - mid)
                    elif kind == 'sawtooth':
                        mask = (t_grid >= t0) & (t_grid <= t1)
                        if t1 > t0: y[mask] = amp * (t_grid[mask] - t0) / (t1 - t0)
                    elif kind == 'impulse':
                        idx = np.argmin(np.abs(t_grid - t0)); y[idx] = amp / self.dt
                    return y
                self.xv = sample_on_grid(self.xi, self.master_t)
                self.hv = sample_on_grid(self.hi, self.master_t)
                self.tx = self.th = self.master_t
                
                
                # Branch logic for continuous calculation
                if self.is_correlation_mode:
                    # Correlation(x,h) is equivalent to Convolution(x, h[-t])
                    # h[-t] is represented by the time-reversed array hv[::-1]
                    self.y_full = np.convolve(self.xv, self.hv[::-1], 'full') * self.dt
                else: # Convolution
                    self.y_full = np.convolve(self.xv, self.hv, 'full') * self.dt
                    # These attributes are only needed for the convolution animation
                    self.h_flip = self.hv[::-1]
                    self.t_flip = -self.th[::-1]
                

                output_start_time = self.master_t[0] + self.master_t[0]
                self.n_full = output_start_time + np.arange(len(self.y_full)) * self.dt

            self._create_synchronized_animation()
            
            self.frame_idx = 0
            self._update_frame()
            fps = self.fps_slider.value()
            self.timer.start(int(1000/fps))
            
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", str(e))

    def _update_frame(self):
        if self.n_full is None or self.frame_idx >= len(self.shift_positions):
            self.timer.stop()
            return
        
        current_shift = self.shift_positions[self.frame_idx]
        current_output = self.output_values[self.frame_idx]
        
        for ax in (self.ax1, self.ax2, self.ax3): 
            ax.clear()
        
        process_name = "Correlation" if self.is_correlation_mode else "Convolution"
        output_symbol = "R" if self.is_correlation_mode else "y"

        if self.discrete:
            if not isinstance(self.xi, DiscreteSignal):
                self.timer.stop()
                return
            
            self.ax1.stem(self.xi.time_indices(), self.xi.values, linefmt='b-', markerfmt='bo', basefmt=" ", label='x[n]')
            self.ax1.stem(self.hi.time_indices(), self.hi.values, linefmt='r-', markerfmt='ro', basefmt=" ", label='h[n]')
            self.ax1.set_title("Original Signals", fontsize=12, fontweight='bold')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_xlim(self.unified_xlim_left, self.unified_xlim_right)

            self.ax2.set_xlim(self.unified_xlim_left, self.unified_xlim_right)
            self.ax2.stem(self.xi.time_indices(), self.xi.values, linefmt='b-', markerfmt='bo', basefmt=" ", label='x[n] (fixed)')
            
            k = int(round(current_shift))
            if self.is_correlation_mode:
                shifted_h_n = self.hi.time_indices() + k
                h_shifted_vals = self.hi.values
                self.ax2.stem(shifted_h_n, h_shifted_vals, linefmt='r-', markerfmt='ro', basefmt=" ", label=f'h[n-{k}]')
            else: # Convolution
                h_shifted_n = -self.hi.time_indices()[::-1] + k
                h_shifted_vals = self.hi.values[::-1]
                self.ax2.stem(h_shifted_n, h_shifted_vals, linefmt='r-', markerfmt='ro', basefmt=" ", label=f'h[{k}-n]')

            self.ax2.set_title(f"Sliding Window at shift k = {current_shift:.1f}", fontsize=12, fontweight='bold')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)

            self.ax3.set_xlim(self.unified_xlim_left, self.unified_xlim_right)
            self.ax3.stem(self.n_full, self.y_full, linefmt='k:', markerfmt='ko', basefmt=" ", label=f'Full {output_symbol}[n]')
            indices_to_plot = self.n_full <= round(current_shift)
            if any(indices_to_plot):
                 self.ax3.stem(self.n_full[indices_to_plot], self.y_full[indices_to_plot], linefmt='b-', markerfmt='bo', basefmt=" ", label=f'{output_symbol}[n]')
            self.ax3.plot([current_shift], [current_output], 'go', markersize=8, label=f'{output_symbol}({current_shift:.1f}) = {current_output:.3f}')
            self.ax3.axvline(x=current_shift, color='red', linestyle='--', linewidth=2)
            self.ax3.set_title(f"{process_name} Output (Synchronized)", fontsize=12, fontweight='bold')
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)

        else: # Continuous Mode
            if not hasattr(self, 'tx'):
                self.timer.stop()
                return

            self.ax1.plot(self.tx, self.xv, 'b-', linewidth=2, label='x(t)')
            self.ax1.plot(self.th, self.hv, 'r-', linewidth=2, label='h(t)')
            self.ax1.set_title("Original Signals", fontsize=12, fontweight='bold')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_xlim(self.unified_xlim_left, self.unified_xlim_right)

            self.ax2.set_xlim(self.unified_xlim_left, self.unified_xlim_right)
            self.ax2.plot(self.tx, self.xv, 'b-', linewidth=2, label='x(t) (fixed)')
            
           
            # Branch logic for drawing the sliding window and overlap
            if self.is_correlation_mode:
                # For correlation, we slide an UN-FLIPPED h(t)
                t_h_shifted = self.th + current_shift
                h_vals_sliding = self.hv
                self.ax2.plot(t_h_shifted, h_vals_sliding, 'r-', linewidth=2, label=f'h(t - {current_shift:.1f})')
            else: # Convolution
                # For convolution, we slide a FLIPPED h(t)
                t_h_shifted = self.t_flip + current_shift
                h_vals_sliding = self.h_flip
                self.ax2.plot(t_h_shifted, h_vals_sliding, 'r-', linewidth=2, label=f'h({current_shift:.1f} - t)')

            # Generalized overlap logic
            h_shifted_on_grid = np.interp(self.master_t, t_h_shifted, h_vals_sliding, left=0, right=0)
            fill_y = np.minimum(self.xv, h_shifted_on_grid)
            fill_condition = (self.xv > 1e-9) & (h_shifted_on_grid > 1e-9)
            self.ax2.fill_between(
                self.master_t, 0, fill_y, where=fill_condition, 
                color='green', alpha=0.4, label='Overlap'
            )
            
            
            self.ax2.set_title(f"Sliding Window at shift t = {current_shift:.1f}", fontsize=12, fontweight='bold')
            self.ax2.legend()
            self.ax2.grid(True, alpha=0.3)

            self.ax3.set_xlim(self.unified_xlim_left, self.unified_xlim_right)
            trace_indices = self.shift_positions <= current_shift
            self.ax3.plot(self.n_full, self.y_full, 'k--', alpha=0.5, label=f'Full {output_symbol}(t)')
            self.ax3.plot(self.shift_positions[trace_indices], self.output_values[trace_indices], 'k-', linewidth=2, label=f'{output_symbol}(t)')
            self.ax3.plot([current_shift], [current_output], 'go', markersize=8, label=f'{output_symbol}({current_shift:.1f}) = {current_output:.3f}')
            self.ax3.axvline(x=current_shift, color='red', linestyle='--', linewidth=2)
            self.ax3.set_title(f"{process_name} Output (Synchronized)", fontsize=12, fontweight='bold')
            self.ax3.legend()
            self.ax3.grid(True, alpha=0.3)

        try:
            self.fig.tight_layout(rect=[0, 0, 1, 0.96])
        except UserWarning:
            pass 
            
        self.fig.suptitle(f"{process_name} Animation", fontsize=14, fontweight='bold')
        self.canvas.draw()
        
        try:
            skip = int(self.frame_skip_edit.text())
            if skip < 1: skip = 1
        except (ValueError, TypeError):
            skip = 1 
            
        self.frame_idx += skip

    def _on_reset(self):
        self.timer.stop()
        self.frame_idx = 0
        self.n_full = self.y_full = None
        
        self.frame_skip_edit.setText("1")
        self.fps_slider.setValue(30)
        
        if self.discrete_rb.isChecked():
            self.x_values_edit.setText("1,2,3,1")
            self.x_start_edit.setText("0")
            self.h_values_edit.setText("1,1,1")
            self.h_start_edit.setText("0")
        else:
            self.x_t0_edit.setText("0")
            self.x_t1_edit.setText("2")
            self.x_amp_edit.setText("1")
            self.h_t0_edit.setText("0")
            self.h_t1_edit.setText("1")
            self.h_amp_edit.setText("0.5")
            self.x_kind_combo.setCurrentText("rectangular")
            self.h_kind_combo.setCurrentText("rectangular")
        
        for ax in (self.ax1, self.ax2, self.ax3): 
            ax.clear()
            ax.grid(True, alpha=0.3)
        
        self.fig.suptitle("Convolution & Correlation GUI", fontsize=14, fontweight='bold')
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())