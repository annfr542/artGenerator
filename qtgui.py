#!/bin/python
import os
import matplotlib.pyplot as plt
import random
import sys
import time
import numpy as np
import scipy.misc
from enum import Enum
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import uic
from PyQt5.QtCore import * 
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from image_generator import generate


noise_width = 255
noise_height = 255

noise_file='.noise.png'

class InputType(Enum):
    NOISE = 0
    IMAGE = 1

def white_noise():
    pil_map = Image.new("RGBA", (noise_width, noise_height), 255)
    random_grid = map(lambda x: (
        random.randrange(0,255),
        random.randrange(0,255),
        random.randrange(0,255)
    ), [0] * 255 * 255)
    pil_map.putdata(list(random_grid))
    pil_map.save(noise_file)
    return ImageQt(pil_map)

def preprocess_image(pixmap):
    pixmap = pixmap.scaled(255, 255, Qt.KeepAspectRatio)
    channels_count = 3
    image = pixmap.toImage()
    b = image.bits()
    b.setsize(pixmap.height() * pixmap.width() * channels_count)
    return np.frombuffer(b, np.uint8).reshape((pixmap.height(), pixmap.width(), channels_count))

class QtGui(QMainWindow):
    def __init__(self):
        super(QtGui, self).__init__()
        ui_file = os.path.dirname(os.path.abspath(__file__)) + '/mainwindow.ui'
        uic.loadUi(ui_file, self)
        self.image = self.findChild(QLabel, 'imageLabel')
        self.image_path = ''
        self.data_path = noise_file

        self.feedback_label = self.findChild(QLabel, 'feedbackLabel')

        self.image_width = 1280
        self.image_height = 720

        self.iter_min = 20
        self.iter_max = 2000

        self.step_min = 0.001
        self.step_max = 0.1

        self.scale_min = 1.1
        self.scale_max = 2.5

        self.epoch_min = 1
        self.epoch_max = 5

        self.model_dict = {
            "Slug": "model3.h5",
            "Buildings": "model_places.h5",
            "Blobs": "model_hybrid.h5",
            "Lighthouse": "model_places_flower_simple.h5"
        }
        self.models = ["Slug", "Buildings", "Blobs", "Lighthouse"]
        input_types = ["Noise", "Image"]
        self.current_type = InputType.NOISE

        self.imageChooser = self.findChild(QPushButton, 'inputButton')
        self.imageChooser.clicked.connect(self.open_file_dialog)

        self.launchButton = self.findChild(QPushButton, 'launchButton')
        self.launchButton.clicked.connect(self.run)

        self.saveButton = self.findChild(QPushButton, 'saveButton')
        self.saveButton.clicked.connect(self.save)
        self.saveButton.setEnabled(0)

        self.iterVal = self.findChild(QLabel, 'iterVal')

        self.iterSlider = self.findChild(QSlider, 'iterSlider')
        self.iterSlider.valueChanged.connect(self.iter_slider_change)

        self.stepVal = self.findChild(QLabel, 'stepSizeVal')

        self.stepSlider = self.findChild(QSlider, 'stepSizeSlider')
        self.stepSlider.valueChanged.connect(self.step_slider_change)

        self.scaleVal = self.findChild(QLabel, 'scaleVal')

        self.scaleSlider = self.findChild(QSlider, 'scaleSlider')
        self.scaleSlider.valueChanged.connect(self.scale_slider_change)

        self.epochSlider = self.findChild(QSlider, 'epochSlider')
        self.epochSlider.valueChanged.connect(self.epoch_slider_change)
        self.epochVal = self.findChild(QLabel, 'epochLabel')

        self.modelCombo = self.findChild(QComboBox, 'modelCombo')
        self.modelCombo.addItems(self.models)

        self.inputType = self.findChild(QComboBox, 'inputType')
        self.inputType.addItems(input_types)
        self.inputType.currentTextChanged.connect(self.input_type_change)

        self.iter_slider_change()
        self.step_slider_change()
        self.scale_slider_change()
        self.epoch_slider_change()
        self.input_type_change()

        self.load_image()
        self.show()

    def get_pixmap(self):
        if str(self.inputType.currentText()) == "Noise":
            return QPixmap.fromImage(white_noise()).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio)
        return QPixmap(self.image_path).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio)

    def load_image(self):
        pixmap = self.get_pixmap()
        self.image.setPixmap(pixmap)
        self.image.setScaledContents(1)
        self.set_output("")

    def show_numpy_image(self, image):
        self.pil_image = image
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimg = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(self.image_width, self.image_height, Qt.KeepAspectRatio)
        self.image.setPixmap(pixmap)
        self.image.setScaledContents(1)
        self.image.repaint()

    def open_file_dialog(self):
        if os.name == 'nt':
            root_path = '%USERPROFILE%'
        else:
            root_path = '$HOME'
        filenames = QFileDialog.getOpenFileName(self, 'Choose image', root_path, "Image files (*.jpg *.png)")
        if filenames[0]:
            self.image_path = filenames[0]
            self.imageChooser.setText(self.image_path)
            self.load_image()
            return 1

        print("No image name")
        if self.image_path == '':
            idx = self.inputType.findText("Noise", Qt.MatchFixedString)
            self.inputType.setCurrentIndex(idx)

        return 0

    def iter_slider_change(self):
        value = self.iterSlider.value()
        nv = str(self.iter_min + (self.iter_max - self.iter_min) * value / 99)
        self.iterVal.setText(nv.split('.')[0])

    def step_slider_change(self):
        value = self.stepSlider.value()
        nv = str(self.step_min + (self.step_max - self.step_min) * value / 99)
        self.stepVal.setText(nv[0:5])

    def scale_slider_change(self):
        value = self.scaleSlider.value()
        nv = str(self.scale_min + (self.scale_max - self.scale_min) * value / 99)
        self.scaleVal.setText(nv[0:4])

    def epoch_slider_change(self):
        value = self.epochSlider.value()
        nv = str(int(self.epoch_min + (self.epoch_max - self.epoch_min) * value / 99))
        self.epochVal.setText(nv[0:1])

    def input_type_change(self):
        if str(self.inputType.currentText()) == "Noise":
            self.data_path = noise_file
            self.imageChooser.setEnabled(0)
            if self.current_type != InputType.NOISE:
                self.current_type = InputType.NOISE
                self.load_image()
        elif self.image_path == '':
            if self.open_file_dialog():
                self.current_type = InputType.IMAGE
                self.imageChooser.setEnabled(1)
                self.load_image()
                self.data_path = self.image_path
        else:
            self.current_type = InputType.IMAGE
            self.imageChooser.setEnabled(1)
            self.load_image()
            self.data_path = self.image_path

    def set_output(self, text):
        self.feedbackLabel.setText(text)
        self.feedbackLabel.repaint()

    def save(self):
        filename = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG files (*.png)")[0]
        plt.imsave(filename, self.pil_image)
        data_file = filename.rsplit('.', 1)[0]
        data_file += ".data"
        dfile = open(data_file, "w")
        dfile.write("Model: %s\n" % self.r_model)
        dfile.write("Iters: %s\n" % self.r_iters)
        dfile.write("Step size: %s\n" % self.r_step)
        dfile.write("Scale: %s\n" % self.r_scale)
        dfile.write("Epochs: %s\n" % self.r_epochs)
        dfile.close()

    def run(self):
        self.r_iters = self.iter_min + (self.iter_max - self.iter_min) * self.iterSlider.value() / 99
        self.r_step = self.step_min + (self.step_max - self.step_min) * self.stepSlider.value() / 99
        self.r_scale = self.scale_min + (self.scale_max - self.scale_min) * self.scaleSlider.value() / 99
        self.r_epochs = self.epoch_min + (self.epoch_max - self.epoch_min) * self.epochSlider.value() / 99
        self.r_model = self.model_dict[str(self.modelCombo.currentText())]

        self.launchButton.setText("Running...")
        self.iterSlider.setEnabled(0)
        self.stepSlider.setEnabled(0)
        self.scaleSlider.setEnabled(0)
        self.modelCombo.setEnabled(0)
        self.launchButton.setEnabled(0)
        self.imageChooser.setEnabled(0)
        self.saveButton.setEnabled(0)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        generate(int(self.r_iters), self.r_step, self.r_scale, self.r_model, self.data_path, self, int(self.r_epochs))
        self.iterSlider.setEnabled(1)
        self.stepSlider.setEnabled(1)
        self.scaleSlider.setEnabled(1)
        self.modelCombo.setEnabled(1)
        self.launchButton.setEnabled(1)
        self.saveButton.setEnabled(1)
        self.launchButton.setText("Run")
        if str(self.inputType.currentText()) == "Image":
            self.imageChooser.setEnabled(1)
        QApplication.restoreOverrideCursor()


if __name__ == "__main__":
    app = QApplication([])
    window = QtGui()
    sys.exit(app.exec_())
