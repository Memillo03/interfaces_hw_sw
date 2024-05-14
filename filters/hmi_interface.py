"""Code for the HMI interface of the audio processing application."""

import sys

import numpy as np
from hmi_processing import AudioSignal
from matplotlib import gridspec
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class AudioUploader(QMainWindow):
    """Main window of the audio processing application."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        self.setWindowTitle("Audio Uploader")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.upload_button = QPushButton("Upload Audio (.wav, .mp3, .aac)", self)
        self.layout.addWidget(self.upload_button)
        self.upload_button.clicked.connect(self.upload_audio)

        self.plot_button = QPushButton("Plot", self)
        self.plot_button.clicked.connect(self.plot_audio)
        self.layout.addWidget(self.plot_button)
        self.plot_button.hide()

        self.fft_button = QPushButton("FFT", self)
        self.fft_button.clicked.connect(self.apply_fft)
        self.layout.addWidget(self.fft_button)
        self.fft_button.hide()

        self.filter_button = QPushButton("Apply Filter", self)
        self.filter_button.clicked.connect(self.apply_filter)
        self.layout.addWidget(self.filter_button)
        self.filter_button.hide()

        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_plot)
        self.layout.addWidget(self.close_button)
        self.close_button.hide()

        self.save_button = None

        self.canvas = None
        self.ax = None

        self.filepath = None
        self.audio = None

    def upload_audio(self):
        """Open a file dialog to upload an audio file."""
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Audio files (*.mp3 *.wav *.aac)")
        file_dialog.setWindowTitle("Select Audio File")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.file_path = selected_files[0]
                self.audio = AudioSignal(self.file_path)
                print(
                    "Selected audio file:", self.file_path
                )  # You can do further processing here
                self.show_buttons()

    def show_buttons(self):
        """Show the buttons for plotting, FFT, and filtering."""
        self.upload_button.hide()
        self.plot_button.show()
        self.fft_button.show()
        self.filter_button.show()

    def plot_audio(self):
        """Plot the original audio signal."""
        # Prepare layout for plotting
        self.reset_plot()

        # Plot original audio signal
        time, signal = self.audio.get_signal("original")
        self.ax.plot(time, signal)
        self.ax.grid()
        self.ax.set_xlabel("Time [s]")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def apply_fft(self):
        """Plot the Fourier Transform of the original audio signal."""
        # Prepare layout for plotting
        self.reset_plot()

        # Plot Fourier Transform of original signal
        freq, ft = self.audio.fourier_transform("original")
        self.ax.plot(freq, np.abs(ft))
        self.ax.grid()
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel("Amplitude")
        self.canvas.draw()

    def apply_filter(self):
        """Show options for applying a filter to the audio signal."""
        self.remove_options()
        self.add_filter_options()

    def remove_options(self):
        """Hide the buttons for plotting, FFT, and filtering."""
        self.plot_button.hide()
        self.fft_button.hide()
        self.filter_button.hide()

    def add_filter_options(self):
        """Show options for applying a filter to the audio signal."""
        self.close_plot()

        self.filter_label = QLabel("Filter Type:", self)
        self.filter_type = QComboBox(self)
        self.filter_type.addItems(["FIR", "IIR"])
        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.filter_label)
        self.hbox1.addWidget(self.filter_type)
        self.hbox1.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.addLayout(self.hbox1)

        self.filter_type.currentIndexChanged.connect(self.add_filter_design)

        self.order_label = QLabel("Filter Order:", self)
        self.order = QLineEdit(self)
        self.order.setText("3")
        self.hbox2 = QHBoxLayout()
        self.hbox2.addWidget(self.order_label)
        self.hbox2.addWidget(self.order)
        self.hbox2.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.addLayout(self.hbox2)

        self.bandpass_label = QLabel("Bandpass Frequency:", self)
        self.bandpass_type = QComboBox(self)
        self.bandpass_type.addItems(["lowpass", "highpass", "bandpass"])
        self.hbox3 = QHBoxLayout()
        self.hbox3.addWidget(self.bandpass_label)
        self.hbox3.addWidget(self.bandpass_type)
        self.hbox3.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.addLayout(self.hbox3)

        self.cutoff_label_1 = QLabel("Cutoff Frequency 1 [Hz]:", self)
        self.cutoff_slider_1 = QSlider(Qt.Orientation.Horizontal, self)
        self.cutoff_slider_1.setMinimum(1)
        self.cutoff_slider_1.setMaximum(self.audio.sampFreq // 2 - 1)
        self.cutoff_slider_1.setValue(5000)
        self.cutoff_slider_1.setTickInterval(1000)
        self.cutoff_slider_1.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cutoff_value_1 = QLineEdit(str(self.cutoff_slider_1.value()), self)
        self.cutoff_value_1.setFixedWidth(50)
        self.hbox4 = QHBoxLayout()
        self.hbox4.addWidget(self.cutoff_label_1)
        self.hbox4.addWidget(self.cutoff_value_1)
        self.hbox4.addWidget(self.cutoff_slider_1)
        self.hbox4.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.layout.addLayout(self.hbox4)
        self.cutoff_slider_1.valueChanged.connect(
            lambda: self.cutoff_value_1.setText(str(self.cutoff_slider_1.value()))
        )
        self.cutoff_value_1.textChanged.connect(
            lambda: self.cutoff_slider_1.setValue(int(self.cutoff_value_1.text()))
        )

        self.bandpass_type.currentIndexChanged.connect(self.add_second_cutoff)

        self.return_button = QPushButton("Return", self)
        self.return_button.clicked.connect(self.return_to_options)

        self.apply_filter_button = QPushButton("Apply Filter", self)
        self.apply_filter_button.clicked.connect(self.show_filtered_signal)

        self.hbox6 = QHBoxLayout()
        self.hbox6.addWidget(self.return_button)
        self.hbox6.addWidget(self.apply_filter_button)
        self.hbox6.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addLayout(self.hbox6)

    def add_filter_design(self, index):
        """Add filter design options for IIR filters."""
        if hasattr(self, "design_type"):
            self.design_label.hide()
            self.design_type.hide()

        if index == 1:  # IIR
            self.design_label = QLabel("Filter Design:", self)
            self.design_type = QComboBox(self)
            self.design_type.addItems(["butter", "cheby1", "cheby2", "ellip", "bessel"])

            self.hbox1.addWidget(self.filter_type)
            self.hbox1.addWidget(self.design_label)
            self.hbox1.addWidget(self.design_type)

    def add_second_cutoff(self, index):
        """Add a second cutoff frequency for bandpass filters."""
        if hasattr(self, "cutoff_slider_2"):
            self.cutoff_label_2.hide()
            self.cutoff_value_2.hide()
            self.cutoff_slider_2.hide()

        if index == 2:
            self.cutoff_label_2 = QLabel("Cutoff Frequency 2 [Hz]:", self)
            self.cutoff_slider_2 = QSlider(Qt.Orientation.Horizontal, self)
            self.cutoff_slider_2.setMinimum(1)
            self.cutoff_slider_2.setMaximum(self.audio.sampFreq // 2 - 1)
            self.cutoff_slider_2.setValue(5000)
            self.cutoff_slider_2.setTickInterval(1000)
            self.cutoff_slider_2.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.cutoff_value_2 = QLineEdit(str(self.cutoff_slider_2.value()), self)
            self.cutoff_value_2.setFixedWidth(50)
            self.cutoff_value_1.textChanged.connect(
                lambda: self.cutoff_slider_2.setMinimum(
                    int(self.cutoff_value_1.text()) + 1
                )
            )

            self.layout.removeWidget(self.apply_filter_button)
            self.layout.removeWidget(self.return_button)
            self.hbox6 = None

            self.hbox5 = QHBoxLayout()
            self.hbox5.addWidget(self.cutoff_label_2)
            self.hbox5.addWidget(self.cutoff_value_2)
            self.hbox5.addWidget(self.cutoff_slider_2)
            self.hbox5.setAlignment(Qt.AlignmentFlag.AlignLeft)

            self.hbox6 = QHBoxLayout()
            self.hbox6.addWidget(self.return_button)
            self.hbox6.addWidget(self.apply_filter_button)
            self.hbox6.setAlignment(Qt.AlignmentFlag.AlignCenter)

            if self.canvas is not None:
                index = self.layout.indexOf(self.canvas)
                self.layout.insertLayout(index, self.hbox5)
                self.layout.insertLayout(index + 1, self.hbox6)
            else:
                self.layout.addLayout(self.hbox5)
                self.layout.addLayout(self.hbox6)

            # self.layout.addLayout(self.hbox5)
            self.cutoff_slider_2.valueChanged.connect(
                lambda: self.cutoff_value_2.setText(str(self.cutoff_slider_2.value()))
            )
            self.cutoff_value_2.textChanged.connect(
                lambda: self.cutoff_slider_2.setValue(int(self.cutoff_value_2.text()))
            )

    def return_to_options(self):
        """Return to the main options screen."""
        self.close_plot()
        self.remove_filter_options()
        self.show_buttons()

    def remove_filter_options(self):
        """Remove the filter options from the layout."""
        self.remove_layouts(self.hbox1)
        self.remove_layouts(self.hbox2)
        self.remove_layouts(self.hbox3)
        self.remove_layouts(self.hbox4)
        if hasattr(self, "hbox5"):
            self.remove_layouts(self.hbox5)
        self.remove_layouts(self.hbox6)

    def remove_layouts(self, layout):
        """Remove a hbox layout from the main layout."""
        for i in reversed(range(layout.count())):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

    def show_filtered_signal(self):
        """Plot the filtered audio signal."""
        self.create_plot()
        self.filtering()

        time, signal = self.audio.get_signal("processed")
        self.ax1.plot(time, signal)
        self.ax1.grid()
        self.ax1.set_xlabel("Time [s]")
        self.ax1.set_ylabel("Amplitude")

        freq, ft = self.audio.fourier_transform("processed")
        freq = freq[: len(freq) // 2]
        ft = ft[: len(ft) // 2]
        self.ax2.plot(freq, np.abs(ft))
        self.ax2.grid()
        self.ax2.set_xlabel("Frequency [Hz]")
        self.ax2.set_ylabel("Amplitude")
        self.canvas.draw()

    def filtering(self):
        """Pass the filter parameters to the audio signal object."""
        filter_type = self.filter_type.currentText().lower()
        print(filter_type)
        band_type = self.bandpass_type.currentText().lower()
        print(band_type)

        order_text = self.order.text()
        if order_text:  # If order_text is not an empty string
            order = int(order_text)
        else:
            # Handle the case where order_text is an empty string
            # For example, set order to a default value
            order = 1
        order = int(self.order.text())
        print(order)
        print(self.filter_type.currentIndex())

        if self.filter_type.currentIndex() == 1:  # IIR
            ftype = self.design_type.currentText().lower()
        else:
            ftype = None

        cutoff_freqs = [self.cutoff_slider_1.value()]
        if band_type == "bandpass":
            cutoff_freqs.append(self.cutoff_slider_2.value())
        print(cutoff_freqs)

        self.audio.apply_filter(filter_type, band_type, cutoff_freqs, order, ftype)

    def reset_plot(self):
        """Reset the plot layout."""
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.close_button)
        self.close_button.show()

    def create_plot(self):
        """Create a plot layout for the filtered signal."""
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.figure = Figure()
        self.figure.set_figheight(10)

        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=self.figure)
        self.ax1 = self.figure.add_subplot(gs[0])
        self.ax2 = self.figure.add_subplot(gs[1])
        self.figure.subplots_adjust(hspace=0.5)
        self.canvas = FigureCanvas(self.figure)
        self.close_button.hide()
        self.close_button = None
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.close_plot)
        self.layout.addWidget(self.canvas)
        self.layout.addWidget(self.close_button)
        self.close_button.show()
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_processed_audio)
        self.layout.addWidget(self.save_button)
        self.save_button.show()

    def close_plot(self):
        """Close the plot layout."""
        if self.canvas is not None:
            self.layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
            self.ax = None
            self.ax1 = None
            self.ax2 = None
            self.canvas = None
            self.layout.removeWidget(self.close_button)
            self.close_button.hide()
            if self.save_button is not None:
                self.layout.removeWidget(self.save_button)
                self.save_button.hide()

    def save_processed_audio(self):
        """Open a dialog to save the processed audio signal."""
        self.save_dialog = SaveDialog(self)
        self.save_dialog.show()


class SaveDialog(QDialog):
    """Dialog to save the processed audio signal."""

    def __init__(self, parent=None):
        """Initialize the save dialog."""
        super().__init__(parent)
        self.setWindowTitle("Save Audio File")
        layout = QVBoxLayout()

        self.save_button = QPushButton("Select Save Path")
        self.save_button.clicked.connect(self.select_save_path)
        layout.addWidget(self.save_button)

        self.format_combo = QComboBox()
        self.format_combo.addItems([".mp3", ".wav", ".aac"])
        layout.addWidget(QLabel("Select File Format:"))
        layout.addWidget(self.format_combo)

        self.file_name_edit = QLineEdit()
        layout.addWidget(QLabel("Enter File Name:"))
        layout.addWidget(self.file_name_edit)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept_save)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.ok_button)
        self.hbox.addWidget(self.cancel_button)

        self.hbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(self.hbox)
        self.setLayout(layout)

        self.path_folder = None

    def select_save_path(self):
        """Open a dialog to select the save path."""
        options = QFileDialog.Option.ShowDirsOnly
        save_folder = QFileDialog.getExistingDirectory(
            self, "Select Save Folder", options=options
        )
        self.path_folder = save_folder

    def accept_save(self):
        """Save the processed audio signal."""
        print(self.path_folder)
        self.parent().audio.save_signal(
            self.path_folder,
            self.file_name_edit.text(),
            self.format_combo.currentText(),
        )
        self.close()


def main():
    """Run the audio processing application."""
    app = QApplication(sys.argv)
    window = AudioUploader()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
