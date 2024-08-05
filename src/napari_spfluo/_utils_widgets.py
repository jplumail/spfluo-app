from __future__ import annotations

import napari
import napari.layers
import numpy as np
from magicgui import magic_factory
from magicgui.widgets import ComboBox, create_widget
from napari.layers import Labels
from qtpy.QtCore import Qt
from qtpy.QtGui import (
    QColor,
    QIcon,
    QImage,
    QPainter,
    QPixmap,
    QStandardItem,
    QStandardItemModel,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)
from skimage.util import img_as_float


@magic_factory(
    threshold={
        "widget_type": "FloatSlider",
        "min": 0,
        "max": 1,
        "label": "threshold",
    }
)
def threshold_widget(
    image: napari.layers.Image, threshold: float = 0.5
) -> napari.layers.Labels:
    d = img_as_float(image.data)
    return Labels(
        d > threshold * d.max(),
        scale=image.scale,
        name=image.name + " thresholded",
    )


class ColorDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        # Peindre le texte
        text = index.model().data(index, Qt.DisplayRole)
        painter.drawText(option.rect, Qt.AlignLeft, text)

        # Peindre la couleur
        color = index.model().data(index, Qt.UserRole)
        color_rect = option.rect
        color_rect.setLeft(
            option.rect.width() - 50
        )  # Ajuster selon la taille souhaitée
        painter.fillRect(color_rect, color)


class QtColorBox(QWidget):
    """A widget that shows a square with a color.

    Parameters
    ----------
    color : np.ndarray (4,) float32
        A color.
    """

    def __init__(self, color) -> None:
        super().__init__()

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)

        self._height = 24
        self.setFixedWidth(self._height)
        self.setFixedHeight(self._height)

        self.color = color

    def paintEvent(self, event):
        painter = QPainter(self)
        self.drawColorBox(painter)

    def drawColorBox(self, painter):
        """Draw the color box with the given painter."""
        if self.color is None:
            for i in range(self._height // 4):
                for j in range(self._height // 4):
                    if (i % 2 == 0 and j % 2 == 0) or (
                        i % 2 == 1 and j % 2 == 1
                    ):
                        painter.setPen(QColor(230, 230, 230))
                        painter.setBrush(QColor(230, 230, 230))
                    else:
                        painter.setPen(QColor(25, 25, 25))
                        painter.setBrush(QColor(25, 25, 25))
                    painter.drawRect(i * 4, j * 4, 5, 5)
        else:
            color = np.round(255 * self.color).astype(int)
            painter.setPen(QColor(*list(color)))
            painter.setBrush(QColor(*list(color)))
            painter.drawRect(0, 0, self._height, self._height)

    def toIcon(self):
        """Convert the color box to a QIcon."""
        pixmap = QPixmap(self._height, self._height)
        pixmap.fill(Qt.transparent)  # Ensure the background is transparent
        painter = QPainter(pixmap)
        self.drawColorBox(painter)  # Use the new drawing method
        painter.end()
        return QIcon(pixmap)


class MergeLabelsWidget(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer: napari.Viewer = viewer
        # self.viewer.layers.events.inserted.connect()
        self.setLayout(QVBoxLayout())

        # select labels layer
        magicgui_combo: ComboBox = create_widget(
            label="labels", annotation="napari.layers.Labels"
        )
        self.viewer.layers.events.inserted.connect(
            magicgui_combo.reset_choices
        )
        self.viewer.layers.events.removed.connect(magicgui_combo.reset_choices)
        self.labels_combo: QComboBox = magicgui_combo.native
        labels_widget = QWidget()
        labels_widget.setLayout(QHBoxLayout())
        labels_widget.layout().addWidget(QLabel("labels:"))
        labels_widget.layout().addWidget(self.labels_combo)

        # Choose merge labels
        colors_widget = QWidget()
        colors_widget.setLayout(QHBoxLayout())
        self.current_labels = QStandardItemModel()
        self.color1_combo = QComboBox()
        self.color2_combo = QComboBox()
        self.color1_combo.setModel(self.current_labels)
        self.color2_combo.setModel(self.current_labels)
        colors_widget.layout().addWidget(self.color1_combo)
        colors_widget.layout().addWidget(self.color2_combo)

        self.labels_combo.currentIndexChanged.connect(self._on_layer_changed)

        # Merge button
        merge_button = QPushButton("Merge")
        merge_button.clicked.connect(self._merge)

        merge_widget = QWidget()
        merge_widget.setLayout(QVBoxLayout())
        merge_widget.layout().addWidget(labels_widget)
        merge_widget.layout().addWidget(colors_widget)
        merge_widget.layout().addWidget(merge_button)

        self.layout().addWidget(merge_widget)

    def _on_layer_data_changed(self):
        self._on_layer_changed(self.labels_combo.currentIndex())

    def _on_layer_changed(self, index: int):
        labels_layer: Labels = self.labels_combo.itemData(index)
        self.current_labels.clear()
        for label in range(labels_layer.data.max() + 1):
            if label > 0 and label in labels_layer.data:
                self.current_labels.appendRow(
                    QStandardItem(
                        QtColorBox(
                            labels_layer.colormap.map(
                                labels_layer._map_labels_to_colors(label)
                            )[0]
                        ).toIcon(),
                        str(label),
                    )
                )
        self.color1_combo.setCurrentIndex(0)
        if labels_layer.data.max() > 1:
            self.color2_combo.setCurrentIndex(1)

        labels_layer.events.set_data.connect(self._on_layer_data_changed)

    def _merge(self):
        labels_layer: Labels = self.labels_combo.currentData()
        label1 = int(self.color1_combo.currentText())
        label2 = int(self.color2_combo.currentText())
        labels_layer.data[labels_layer.data == label2] = label1
        labels_layer.refresh()


def ndarray_to_QImage(numpy_array: np.ndarray):
    assert numpy_array.dtype == np.uint8
    channels, height, width = numpy_array.shape
    bytesPerLine = width * channels
    concatenated_channels = np.concatenate((*numpy_array,), axis=1)
    qimage = QImage(
        concatenated_channels.data,
        width * channels,
        height,
        bytesPerLine,
        QImage.Format.Format_Grayscale8,
    )
    return qimage


class FilterSetWidget(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__()
        self.setLayout(QVBoxLayout(parent))

        self.scroll_area = QScrollArea()
        self.layout().addWidget(self.scroll_area)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.scroll_area.setWidget(self.list_widget)
        self.scroll_area.setWidgetResizable(True)

        self.image_width = 100

    def set_data(self, image_data: list[np.ndarray], clear=False):
        for _ in range(len(image_data) - self.list_widget.count()):
            item = QListWidgetItem(self.list_widget)
            image_label = QLabel()
            self.list_widget.setItemWidget(item, image_label)

        self.mask_indices_image = np.arange(len(image_data))
        for i, image in enumerate(image_data):
            multichannel = image.ndim == 4
            data = image
            if not multichannel:
                data = data[None]  # C=1
            item = self.list_widget.item(i)
            im = data
            im = np.sum(np.asarray(im, dtype=float), axis=1)
            im = (im - np.min(im, axis=(-2, -1), keepdims=True)) / (
                np.max(im, axis=(-2, -1), keepdims=True)
                - np.min(im, axis=(-2, -1), keepdims=True)
            )
            im = np.asarray(im * 255, dtype=np.uint8)

            image_label = QLabel()
            pixmap = QPixmap.fromImage(ndarray_to_QImage(im)).scaled(
                self.image_width * im.shape[0],
                self.image_width,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            image_label.setPixmap(pixmap)
            item.setSizeHint(image_label.sizeHint())
            item.setText(str(i))
            self.list_widget.setItemWidget(item, image_label)

    def _on_delete(self):
        indices = []
        for item in self.list_widget.selectedItems():
            index = self.list_widget.row(item)
            indices.append(index)
            self.list_widget.takeItem(index)
            self.mask_indices_image = np.delete(
                self.mask_indices_image, index, axis=0
            )
        return indices


class FilterLayerWidget(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer):
        super().__init__()
        self.viewer: napari.Viewer = viewer
        self.setLayout(QVBoxLayout())

        # select image layer
        self.image_layer_combo: ComboBox = create_widget(
            label="set of particles", annotation="napari.layers.Image"
        )
        self.viewer.layers.events.inserted.connect(
            self.image_layer_combo.reset_choices
        )
        self.viewer.layers.events.removed.connect(
            self.image_layer_combo.reset_choices
        )
        self.image_layer_qcombo: QComboBox = self.image_layer_combo.native
        self.current_image_layer: napari.layers.Image | None = None
        self.layout().addWidget(self.image_layer_qcombo)

        self.filter_set_widget = FilterSetWidget(self)
        self.layout().addWidget(self.filter_set_widget)

        self.image_layer_combo.changed.connect(self._on_layer_changed)
        self.viewer.bind_key("Delete", self._on_delete)
        self.filter_set_widget.list_widget.currentItemChanged.connect(
            self._on_item_changed
        )

    def _on_layer_changed(self, image_layer: napari.layers.Image):
        if image_layer and (image_layer.ndim == 4 or image_layer.ndim == 5):
            self.current_image_layer = image_layer
            self.original_data = np.copy(self.current_image_layer.data)
            self.filter_set_widget.set_data(list(image_layer.data))

    def _on_item_changed(
        self, item: QListWidgetItem, previous: QListWidgetItem
    ):
        i = self.filter_set_widget.list_widget.row(item)
        self.viewer.dims.current_step = (i, *self.viewer.dims.current_step[1:])

    def _on_delete(self, viewer: napari.Viewer):
        self.filter_set_widget._on_delete()
        self.current_image_layer.data = self.original_data[
            self.filter_set_widget.mask_indices_image
        ]
