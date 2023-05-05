"""
example taken from: https://github.com/napari/napari/blob/main/examples/multiple_viewer_widget.py

Multiple viewer widget
======================

This is an example on how to have more than one viewer in the same napari window.
Additional viewers state will be synchronized with the main viewer.
Switching to 3D display will only impact the main viewer.

This example also contain option to enable cross that will be moved to the
current dims point (`viewer.dims.point`).

.. tags:: gui
"""

from copy import deepcopy

import numpy as np
from packaging.version import parse as parse_version
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Image, Labels, Layer, Vectors, Points
from napari.qt import QtViewer
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info

NAPARI_GE_4_16 = parse_version(napari.__version__) > parse_version("0.4.16")


def copy_layer_le_4_16(layer: Layer, name: str = ""):
    res_layer = deepcopy(layer)
    # this deepcopy is not optimal for labels and images layers
    if isinstance(layer, (Image, Labels)):
        res_layer.data = layer.data

    res_layer.metadata["viewer_name"] = name

    res_layer.events.disconnect()
    res_layer.events.source = res_layer
    for emitter in res_layer.events.emitters.values():
        emitter.disconnect()
        emitter.source = res_layer
    return res_layer


def copy_layer(layer: Layer, name: str = ""):
    if NAPARI_GE_4_16:
        return copy_layer_le_4_16(layer, name)

    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata["viewer_name"] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ("thumbnail", "name"):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


def center_cross_on_mouse(
    viewer_model: napari.components.viewer_model.ViewerModel,
):
    """move the cross to the mouse position"""

    if not getattr(viewer_model, "mouse_over_canvas", True):
        # There is no way for napari 0.4.15 to check if mouse is over sending canvas.
        show_info(
            "Mouse is not over the canvas. You may need to click on the canvas."
        )
        return

    viewer_model.dims.current_step = tuple(
        np.round(
            [
                max(min_, min(p, max_)) / step
                for p, (min_, max_, step) in zip(
                    viewer_model.cursor.position, viewer_model.dims.range
                )
            ]
        ).astype(int)
    )


action_manager.register_action(
    name='napari:move_point',
    command=center_cross_on_mouse,
    description='Move dims point to mouse position',
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut('napari:move_point', 'C')


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: str = None,
        layer_type: str = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class CrossWidget(QCheckBox):
    """
    Widget to control the cross layer. because of the performance reason
    the cross update is throttled
    """

    def __init__(self, viewer: napari.Viewer, image_layer: Image) -> None:
        super().__init__("Add cross layer")
        self.viewer = viewer
        self.setChecked(False)
        self.stateChanged.connect(self._update_cross_visibility)
        self.layer = None
        self.viewer.dims.events.order.connect(self.update_cross)
        self.viewer.dims.events.ndim.connect(self._update_ndim)
        self.viewer.dims.events.current_step.connect(self.update_cross)
        self._extent = image_layer.extent
        self.layer = Vectors(name=".cross", ndim=image_layer.ndim)
        self.layer.edge_width = 0.5

        self.viewer.dims.events.connect(self.update_cross)

    def _update_ndim(self, event):
        if self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)
        self.layer = Vectors(name=".cross", ndim=event.value)
        self.layer.edge_width = 0.5
        self.update_cross()

    def _update_cross_visibility(self, state):
        if state:
            self.viewer.layers.append(self.layer)
        else:
            self.viewer.layers.remove(self.layer)
        self.update_cross()

    def update_cross(self):
        if self.layer not in self.viewer.layers:
            return

        point = self.viewer.dims.current_step
        vec = []
        for i, (lower, upper) in enumerate(self._extent.world.T):
            if (upper - lower) / self._extent.step[i] == 1:
                continue
            point1 = list(point)
            point1[i] = (lower + self._extent.step[i] / 2) / self._extent.step[
                i
            ]
            point2 = [0 for _ in point]
            point2[i] = (upper - lower) / self._extent.step[i]
            vec.append((point1, point2))
        if np.any(self.layer.scale != self._extent.step):
            self.layer.scale = self._extent.step
        self.layer.data = vec


class ExampleWidget(QWidget):
    """
    Dummy widget showcasing how to place additional widgets to the right
    of the additional viewers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.btn = QPushButton("Perform action")
        self.spin = QDoubleSpinBox()
        layout = QVBoxLayout()
        layout.addWidget(self.spin)
        layout.addWidget(self.btn)
        layout.addStretch(1)
        self.setLayout(layout)


class MultipleViewerWidget(QSplitter):
    def __init__(self, viewer: ViewerModel, dim: int) -> None:
        super().__init__()
        self.viewer = viewer
        self.dim = dim
        self.viewer_model = ViewerModel(title="model")
        self._block = False
        self.qt_viewer = QtViewerWrap(viewer, self.viewer_model)
        self.addWidget(self.qt_viewer)

        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model.events.status.connect(self._status_update)

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        self.viewer_model.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model.layers.selection.active = None
            return

        self.viewer_model.layers.selection.active = self.viewer_model.layers[
            event.value.name
        ]

    def _point_update(self, event):
        for model in [self.viewer, self.viewer_model]:
            if model.dims is event.source:
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:
            self.viewer_model.dims.order = order
            return

        if self.dim == 1:
            order[-3:] = order[-2], order[-3], order[-1]
        elif self.dim == 2:
            order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model.dims.order = order

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        self.viewer_model.layers.insert(
            event.index, copy_layer(event.value, "model")
        )
        for name in get_property_names(event.value):
            getattr(event.value.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(event.value, Labels) or isinstance(event.value, Points):
            event.value.events.set_data.connect(self._set_data_refresh)
            self.viewer_model.layers[
                event.value.name
            ].events.set_data.connect(self._set_data_refresh)
        #if isinstance(event.value, Points): # TODO
        #    event.value.events.current_size.connect(self._on_current_size)
        if event.value.name != ".cross":
            self.viewer_model.layers[event.value.name].events.data.connect(
                self._sync_data
            )

        event.value.events.name.connect(self._sync_name)

        self._order_update()
    
    def _on_current_size(self, event):
        self.viewer_model

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        self.viewer_model.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        for model in [self.viewer, self.viewer_model]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model]:
            layer = model.layers[event.source.name]
            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        self.viewer_model.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.viewer_model.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True
            setattr(
                self.viewer_model.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
        finally:
            self._block = False


if __name__ == "__main__":
    import tifffile
    from skimage.data import cells3d
    im = cells3d()
    #im = tifffile.imread("/home/plumail/Téléchargements/1.tif")

    from qtpy import QtCore, QtWidgets
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    # above two lines are needed to allow to undock the widget with
    # additional viewers
    view = napari.Viewer()
    dock_widget1 = MultipleViewerWidget(view, dim=1)
    dock_widget2 = MultipleViewerWidget(view, dim=2)
    
    def _update_zoom(event):
        for model in [view, dock_widget1.viewer_model, dock_widget2.viewer_model]:
            model: ViewerModel
            model.camera.zoom = event.value
    
    # Mouse callbacks
    view.camera.events.zoom.connect(_update_zoom)
    dock_widget1.viewer_model.camera.events.zoom.connect(_update_zoom)
    dock_widget2.viewer_model.camera.events.zoom.connect(_update_zoom)

    view.window.add_dock_widget(dock_widget1, name="Sample1", area="bottom")
    view.window.add_dock_widget(dock_widget2, name="Sample2", area="right")

    image_layer = view.add_image(
        im,
        channel_axis=1,
        name=["membrane", "nuclei"],
        colormap=["green", "magenta"],
        contrast_limits=[[1000, 20000], [1000, 50000]],
    )

    cross = CrossWidget(view, image_layer[0])
    view.window.add_dock_widget(cross, name="Cross", area="left")

    points_layer = Points(
        ndim=3,
        edge_color=[0,0,255,255],
        face_color=[0,0,0,0],
        out_of_slice_display=True,
        size=10
    )
    view.add_layer(points_layer)

    # Add callbacks when dragging points
    def on_move_point(event):
        layer = event.source
        if len(layer.selected_data) > 0:
            idx = list(layer.selected_data)[0]
            pos = tuple(layer.data[idx])
            viewers = []
            if not dock_widget1.underMouse():
                viewers.append(dock_widget1.viewer_model)
            if not dock_widget2.underMouse():
                viewers.append(dock_widget2.viewer_model)
            if len(viewers) == 1:
                viewers.append(view)
            # move current_step
            for model in viewers:
                model: ViewerModel
                pos_reordered = tuple(np.array(pos)[list(model.dims.order)])
                model.camera.center = pos_reordered
            view.dims.current_step = tuple(
                np.round(
                    [
                        max(min_, min(p, max_)) / step
                        for p, (min_, max_, step) in zip(
                            pos, model.dims.range
                        )
                    ]
                ).astype(int)
            )
    points_layer.events.set_data.connect(on_move_point)


    def on_size_change(event):
        print("Constant size!!!!")
        layer = event.source
        layer.size = layer.current_size
    
    points_layer.events.size.connect(on_size_change)

    view.camera.zoom = 10. # dumb value


    napari.run()
