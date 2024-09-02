# A copy of napari.layers.shapes._shape_list
from collections.abc import Iterable
from typing import Sequence, Union

import numpy as np

from napari.layers.shapes._shapes_utils import triangles_intersect_box
from napari.utils.geometry import (
    inside_triangles,
    intersect_line_with_triangles,
    line_in_triangles_3d,
)
from napari.utils.translations import trans
from .bounding_box import BoundingBox
from ._mesh import Mesh


class BoundingBoxList:
    """List of bounding boxes class.

    Parameters
    ----------
    data : list
        List of BoundingBox objects
    ndisplay : int
        Number of displayed dimensions.

    Attributes
    ----------
    bounding_boxes : (N, ) list
        Bounding box objects.
    data : (N, ) list of (M, D) array
        Data arrays for each bounding box.
    ndisplay : int
        Number of displayed dimensions.
    slice_keys : (N, 2, P) array
        Array of slice keys for each bounding box. Each slice key has the min and max
        values of the P non-displayed dimensions, useful for slicing
        multidimensional bounding boxes. If the both min and max values of bounding box are
        equal then the bounding box is entirely contained within the slice specified
        by those values.
    edge_color : (N x 4) np.ndarray
        Array of RGBA edge colors for each bounding box.
    face_color : (N x 4) np.ndarray
        Array of RGBA face colors for each bounding box.
    edge_widths : (N, ) list of float
        Edge width for each bounding box.
    z_indices : (N, ) list of int
        z-index for each bounding box.

    Notes
    -----
    _vertices : np.ndarray
        Mx2 array of all displayed vertices from all bounding boxes
    _index : np.ndarray
        Length M array with the index (0, ..., N-1) of each bounding box that each
        vertex corresponds to
    _z_index : np.ndarray
        Length N array with z_index of each bounding box
    _z_order : np.ndarray
        Length N array with z_order of each bounding box. This must be a permutation
        of (0, ..., N-1).
    _mesh : Mesh
        Mesh object containing all the mesh information that will ultimately
        be rendered.
    """

    def __init__(self, data=(), ndisplay=2) -> None:
        self._ndisplay = ndisplay
        self.bounding_boxes = []
        self._displayed = []
        self._slice_key = []
        self.displayed_vertices = []
        self.displayed_index = []
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)

        self._mesh = Mesh(ndisplay=self.ndisplay)

        self._edge_color = np.empty((0, 4))
        self._face_color = np.empty((0, 4))

        for d in data:
            self.add(d)

    @property
    def data(self):
        """list of (M, D) array: data arrays for each bounding box."""
        return [bb.data for bb in self.bounding_boxes]

    @property
    def ndisplay(self):
        """int: Number of displayed dimensions."""
        return self._ndisplay

    @ndisplay.setter
    def ndisplay(self, ndisplay):
        if self.ndisplay == ndisplay:
            return

        self._ndisplay = ndisplay
        self._mesh.ndisplay = self.ndisplay
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        for index in range(len(self.bounding_boxes)):
            bounding_box = self.bounding_boxes[index]
            bounding_box.ndisplay = self.ndisplay
            self.remove(index, renumber=False)
            self.add(bounding_box, bounding_box_index=index)
        self._update_z_order()

    @property
    def slice_keys(self):
        """(N, 2, P) array: slice key for each bounding box."""
        return np.array([bb.slice_key for bb in self.bounding_boxes])

    @property
    def edge_color(self):
        """(N x 4) np.ndarray: Array of RGBA edge colors for each bounding box"""
        return self._edge_color

    @edge_color.setter
    def edge_color(self, edge_color):
        self._set_color(edge_color, 'edge')

    @property
    def face_color(self):
        """(N x 4) np.ndarray: Array of RGBA face colors for each bounding box"""
        return self._face_color

    @face_color.setter
    def face_color(self, face_color):
        self._set_color(face_color, 'face')

    def _set_color(self, colors, attribute):
        """Set the face_color or edge_color property

        Parameters
        ----------
        colors : (N, 4) np.ndarray
            The value for setting edge or face_color. There must
            be one color for each bounding box
        attribute : str in {'edge', 'face'}
            The name of the attribute to set the color of.
            Should be 'edge' for edge_color or 'face' for face_color.
        """
        n_bounding_boxes = len(self.data)
        if not np.all(colors.shape == (n_bounding_boxes, 4)):
            raise ValueError(
                trans._(
                    '{attribute}_color must have shape ({n_bounding_boxes}, 4)',
                    deferred=True,
                    attribute=attribute,
                    n_bounding_boxes=n_bounding_boxes,
                )
            )

        update_method = getattr(self, f'update_{attribute}_colors')
        indices = np.arange(len(colors))
        update_method(indices, colors, update=False)
        self._update_displayed()

    @property
    def edge_widths(self):
        """list of float: edge width for each bounding box."""
        return [bb.edge_width for bb in self.bounding_boxes]

    @property
    def z_indices(self):
        """list of int: z-index for each bounding box."""
        return [bb.z_index for bb in self.bounding_boxes]

    @property
    def slice_key(self):
        """list: slice key for slicing n-dimensional bounding boxes."""
        return self._slice_key

    @slice_key.setter
    def slice_key(self, slice_key):
        slice_key = list(slice_key)
        if not np.all(self._slice_key == slice_key):
            self._slice_key = slice_key
            self._update_displayed()

    def _update_displayed(self):
        """Update the displayed data based on the slice key."""
        # The list slice key is repeated to check against both the min and
        # max values stored in the shapes slice key.
        slice_key = np.array(self.slice_key)
        if len(slice_key) != self.slice_keys.shape[-1]:
            return
        # Slice key must exactly match mins and maxs of shape as then the
        # shape is entirely contained within the current slice.
        if len(self.bounding_boxes) > 0:
            self._displayed = np.all(
                np.logical_and(self.slice_keys[:, 0, :] <= slice_key, slice_key <= self.slice_keys[:, 1, :]), axis=1)
        else:
            self._displayed = []
        disp_indices = np.where(self._displayed)[0]

        z_order = self._mesh.triangles_z_order
        disp_tri = np.isin(
            self._mesh.triangles_index[z_order, 0], disp_indices
        )
        self._mesh.displayed_triangles = self._mesh.triangles[z_order][
            disp_tri
        ]
        self._mesh.displayed_triangles_index = self._mesh.triangles_index[
            z_order
        ][disp_tri]
        self._mesh.displayed_triangles_colors = self._mesh.triangles_colors[
            z_order
        ][disp_tri]
        disp_vert = np.isin(self._index, disp_indices)
        self.displayed_vertices = self._vertices[disp_vert]
        self.displayed_index = self._index[disp_vert]

    def add(
        self,
        bounding_box: Union[BoundingBox, Sequence[BoundingBox]],
        face_color=None,
        edge_color=None,
        bounding_box_index=None,
        z_refresh=True,
    ):
        """Adds a single BoundingBox object

        Parameters
        ----------
        bounding_box : BoundingBox
            The bounding box to add
        face_color : color (or iterable of colors of same length as shape)
        edge_color : color (or iterable of colors of same length as shape)
        bounding_box_index : None | int
            If int then edits the bounding box date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new bounding box to end of bounding boxes list
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When bounding_box_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of bounding boxes, set to false  and then call
            BoundingBoxList._update_z_order() once at the end.
        """
        # single shape mode
        if issubclass(type(bounding_box), BoundingBox):
            self._add_single_bounding_box(
                bounding_box=bounding_box,
                face_color=face_color,
                edge_color=edge_color,
                bounding_box_index=bounding_box_index,
                z_refresh=z_refresh,
            )
        # multiple shape mode
        elif isinstance(bounding_box, Iterable):
            if bounding_box_index is not None:
                raise ValueError(
                    trans._(
                        'bounding_box_index must be None when adding multiple bounding boxes',
                        deferred=True,
                    )
                )
            self._add_multiple_bounding_boxes(
                bounding_boxes=bounding_box,
                face_colors=face_color,
                edge_colors=edge_color,
                z_refresh=z_refresh,
            )
        else:
            raise TypeError(
                trans._(
                    'Cannot add single nor multiple bounding box',
                    deferred=True,
                )
            )

    def _add_single_bounding_box(
        self,
        bounding_box,
        face_color=None,
        edge_color=None,
        bounding_box_index=None,
        z_refresh=True,
    ):
        """Adds a single BoundingBox object

        Parameters
        ----------
        bounding_box : subclass BoundingBox
            Must be a BoundingBox
        bounding_box_index : None | int
            If int then edits the bounding box date at current index. To be used in
            conjunction with `remove` when renumber is `False`. If None, then
            appends a new bounding box to end of bounding boxes list
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When bounding_box_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of bounding boxes, set to false  and then call
            BoundingBoxList._update_z_order() once at the end.
        """
        if not issubclass(type(bounding_box), BoundingBox):
            raise TypeError(
                trans._(
                    'bounding_box must be subclass of BoundingBox',
                    deferred=True,
                )
            )

        if bounding_box_index is None:
            bounding_box_index = len(self.bounding_boxes)
            self.bounding_boxes.append(bounding_box)
            self._z_index = np.append(self._z_index, bounding_box.z_index)

            if face_color is None:
                face_color = np.array([1, 1, 1, 1])
            self._face_color = np.vstack([self._face_color, face_color])
            if edge_color is None:
                edge_color = np.array([0, 0, 0, 1])
            self._edge_color = np.vstack([self._edge_color, edge_color])
        else:
            z_refresh = False
            self.bounding_boxes[bounding_box_index] = bounding_box
            self._z_index[bounding_box_index] = bounding_box.z_index

            if face_color is None:
                face_color = self._face_color[bounding_box_index]
            else:
                self._face_color[bounding_box_index, :] = face_color
            if edge_color is None:
                edge_color = self._edge_color[bounding_box_index]
            else:
                self._edge_color[bounding_box_index, :] = edge_color

        self._vertices = np.append(
            self._vertices, bounding_box.data_displayed, axis=0
        )
        index = np.repeat(bounding_box_index, len(bounding_box.data_displayed)) # TODO check
        self._index = np.append(self._index, index, axis=0)

        # Add faces to mesh
        m = len(self._mesh.vertices)
        vertices = bounding_box._face_vertices
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = bounding_box._face_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = np.zeros(bounding_box._face_vertices.shape)
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[bounding_box_index, 0]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = bounding_box._face_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[bounding_box_index, 0]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([face_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        # Add edges to mesh
        m = len(self._mesh.vertices)
        vertices = (
                bounding_box._edge_vertices + bounding_box.edge_width * bounding_box._edge_offsets
        )
        self._mesh.vertices = np.append(self._mesh.vertices, vertices, axis=0)
        vertices = bounding_box._edge_vertices
        self._mesh.vertices_centers = np.append(
            self._mesh.vertices_centers, vertices, axis=0
        )
        vertices = bounding_box._edge_offsets
        self._mesh.vertices_offsets = np.append(
            self._mesh.vertices_offsets, vertices, axis=0
        )
        index = np.repeat([[bounding_box_index, 1]], len(vertices), axis=0)
        self._mesh.vertices_index = np.append(
            self._mesh.vertices_index, index, axis=0
        )

        triangles = bounding_box._edge_triangles + m
        self._mesh.triangles = np.append(
            self._mesh.triangles, triangles, axis=0
        )
        index = np.repeat([[bounding_box_index, 1]], len(triangles), axis=0)
        self._mesh.triangles_index = np.append(
            self._mesh.triangles_index, index, axis=0
        )
        color_array = np.repeat([edge_color], len(triangles), axis=0)
        self._mesh.triangles_colors = np.append(
            self._mesh.triangles_colors, color_array, axis=0
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()

    def _add_multiple_bounding_boxes(
        self,
        bounding_boxes,
        face_colors=None,
        edge_colors=None,
        z_refresh=True,
    ):
        """Add multiple bounding boxes at once (faster than adding them one by one)

        Parameters
        ----------
        bounding_boxes : iterable of BoundingBox
        face_colors : iterable of face_color
        edge_colors : iterable of edge_color
        z_refresh : bool
            If set to true, the mesh elements are reindexed with the new z order.
            When bounding_box_index is provided, z_refresh will be overwritten to false,
            as the z indices will not change.
            When adding a batch of bounding boxes, set to false  and then call
            BoundingBoxList._update_z_order() once at the end.

        TODO: Currently shares a lot of code with `add()`, with the
        difference being that `add()` supports inserting bounding boxes at a specific
        `bounding_box_index`, whereas `add_multiple` will append them as a full batch
        """

        def _make_index(length, bounding_box_index, cval=0):
            """Same but faster than `np.repeat([[bounding_box_index, cval]], length, axis=0)`"""
            index = np.empty((length, 2), np.int32)
            index.fill(cval)
            index[:, 0] = bounding_box_index
            return index

        all_z_index = []
        all_vertices = []
        all_index = []
        all_mesh_vertices = []
        all_mesh_vertices_centers = []
        all_mesh_vertices_offsets = []
        all_mesh_vertices_index = []
        all_mesh_triangles = []
        all_mesh_triangles_index = []
        all_mesh_triangles_colors = []

        m_mesh_vertices_count = len(self._mesh.vertices)

        if face_colors is None:
            face_colors = np.tile(np.array([1, 1, 1, 1]), (len(bounding_boxes), 1))
        else:
            face_colors = np.asarray(face_colors)

        if edge_colors is None:
            edge_colors = np.tile(np.array([0, 0, 0, 1]), (len(bounding_boxes), 1))
        else:
            edge_colors = np.asarray(edge_colors)

        if not len(face_colors) == len(edge_colors) == len(bounding_boxes):
            raise ValueError(
                trans._(
                    'bounding_boxes, face_colors, and edge_colors must be the same length',
                    deferred=True,
                )
            )

        if not all(issubclass(type(bounding_box), BoundingBox) for bounding_box in bounding_boxes):
            raise ValueError(
                trans._(
                    'all bounding boxes must be subclass of BoundingBox',
                    deferred=True,
                )
            )

        for bounding_box, face_color, edge_color in zip(
            bounding_boxes, face_colors, edge_colors
        ):
            bounding_box_index = len(self.bounding_boxes)
            self.bounding_boxes.append(bounding_box)
            all_z_index.append(bounding_box.z_index)
            all_vertices.append(bounding_box.data_displayed)
            all_index.append([bounding_box_index] * len(bounding_box.data_displayed))

            # Add faces to mesh
            m_tmp = m_mesh_vertices_count
            all_mesh_vertices.append(bounding_box._face_vertices)
            m_mesh_vertices_count += len(bounding_box._face_vertices)
            all_mesh_vertices_centers.append(bounding_box._face_vertices)
            vertices = np.zeros(bounding_box._face_vertices.shape)
            all_mesh_vertices_offsets.append(vertices)
            all_mesh_vertices_index.append(
                _make_index(len(vertices), bounding_box_index, cval=0)
            )

            triangles = bounding_box._face_triangles + m_tmp
            all_mesh_triangles.append(triangles)
            all_mesh_triangles_index.append(
                _make_index(len(triangles), bounding_box_index, cval=0)
            )

            color_array = np.repeat([face_color], len(triangles), axis=0)
            all_mesh_triangles_colors.append(color_array)

            # Add edges to mesh
            m_tmp = m_mesh_vertices_count

            vertices = (
                bounding_box._edge_vertices + bounding_box.edge_width * bounding_box._edge_offsets
            )
            all_mesh_vertices.append(vertices)
            m_mesh_vertices_count += len(vertices)

            all_mesh_vertices_centers.append(bounding_box._edge_vertices)

            all_mesh_vertices_offsets.append(bounding_box._edge_offsets)

            all_mesh_vertices_index.append(
                _make_index(len(bounding_box._edge_offsets), bounding_box_index, cval=1)
            )

            triangles = bounding_box._edge_triangles + m_tmp
            all_mesh_triangles.append(triangles)
            all_mesh_triangles_index.append(
                _make_index(len(triangles), bounding_box_index, cval=1)
            )

            color_array = np.repeat([edge_color], len(triangles), axis=0)
            all_mesh_triangles_colors.append(color_array)

        # assemble properties
        self._z_index = np.append(self._z_index, np.array(all_z_index), axis=0)
        self._face_color = np.vstack((self._face_color, face_colors))
        self._edge_color = np.vstack((self._edge_color, edge_colors))
        self._vertices = np.vstack((self._vertices, np.vstack(all_vertices)))
        self._index = np.append(self._index, np.concatenate(all_index), axis=0)

        self._mesh.vertices = np.vstack(
            (self._mesh.vertices, np.vstack(all_mesh_vertices))
        )
        self._mesh.vertices_centers = np.vstack(
            (self._mesh.vertices_centers, np.vstack(all_mesh_vertices_centers))
        )
        self._mesh.vertices_offsets = np.vstack(
            (self._mesh.vertices_offsets, np.vstack(all_mesh_vertices_offsets))
        )
        self._mesh.vertices_index = np.vstack(
            (self._mesh.vertices_index, np.vstack(all_mesh_vertices_index))
        )

        self._mesh.triangles = np.vstack(
            (self._mesh.triangles, np.vstack(all_mesh_triangles))
        )
        self._mesh.triangles_index = np.vstack(
            (self._mesh.triangles_index, np.vstack(all_mesh_triangles_index))
        )
        self._mesh.triangles_colors = np.vstack(
            (self._mesh.triangles_colors, np.vstack(all_mesh_triangles_colors))
        )

        if z_refresh:
            # Set z_order
            self._update_z_order()

    def remove_all(self):
        """Removes all bounding boxes"""
        self.bounding_boxes = []
        self._vertices = np.empty((0, self.ndisplay))
        self._index = np.empty((0), dtype=int)
        self._z_index = np.empty((0), dtype=int)
        self._z_order = np.empty((0), dtype=int)
        self._mesh.clear()
        self._update_displayed()

    def remove(self, index, renumber=True):
        """Removes a single bounding box located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be removed.
        renumber : bool
            Bool to indicate whether to renumber all bounding boxes or not. If not the
            expectation is that this bounding box is being immediately added back to the
            list using `add`.
        """
        indices = self._index != index
        self._vertices = self._vertices[indices]
        self._index = self._index[indices]

        # Remove triangles
        indices = self._mesh.triangles_index[:, 0] != index
        self._mesh.triangles = self._mesh.triangles[indices]
        self._mesh.triangles_colors = self._mesh.triangles_colors[indices]
        self._mesh.triangles_index = self._mesh.triangles_index[indices]

        # Remove vertices
        indices = self._mesh.vertices_index[:, 0] != index
        self._mesh.vertices = self._mesh.vertices[indices]
        self._mesh.vertices_centers = self._mesh.vertices_centers[indices]
        self._mesh.vertices_offsets = self._mesh.vertices_offsets[indices]
        self._mesh.vertices_index = self._mesh.vertices_index[indices]
        indices = np.where(np.invert(indices))[0]
        num_indices = len(indices)
        if num_indices > 0:
            indices = self._mesh.triangles > indices[0]
            self._mesh.triangles[indices] = (
                self._mesh.triangles[indices] - num_indices
            )

        if renumber:
            del self.bounding_boxes[index]
            indices = self._index > index
            self._index[indices] = self._index[indices] - 1
            self._z_index = np.delete(self._z_index, index)
            indices = self._mesh.triangles_index[:, 0] > index
            self._mesh.triangles_index[indices, 0] = (
                self._mesh.triangles_index[indices, 0] - 1
            )
            indices = self._mesh.vertices_index[:, 0] > index
            self._mesh.vertices_index[indices, 0] = (
                self._mesh.vertices_index[indices, 0] - 1
            )
            self._update_z_order()

    def _update_mesh_vertices(self, index, edge=False, face=False):
        """Updates the mesh vertex data and vertex data for a single bounding box
        located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        edge : bool
            Bool to indicate whether to update mesh vertices corresponding to
            edges
        face : bool
            Bool to indicate whether to update mesh vertices corresponding to
            faces and to update the underlying bounding box vertices
        """
        bounding_box = self.bounding_boxes[index]
        if edge:
            indices = np.all(self._mesh.vertices_index == [index, 1], axis=1)
            self._mesh.vertices[indices] = (
                bounding_box._edge_vertices + bounding_box.edge_width * bounding_box._edge_offsets
            )
            self._mesh.vertices_centers[indices] = bounding_box._edge_vertices
            self._mesh.vertices_offsets[indices] = bounding_box._edge_offsets
            self._update_displayed()

        if face:
            indices = np.all(self._mesh.vertices_index == [index, 0], axis=1)
            self._mesh.vertices[indices] = bounding_box._face_vertices
            self._mesh.vertices_centers[indices] = bounding_box._face_vertices
            indices = self._index == index
            self._vertices[indices] = bounding_box.data_displayed
            self._update_displayed()

    def _update_z_order(self):
        """Updates the z order of the triangles given the z_index list"""
        self._z_order = np.argsort(self._z_index)
        if len(self._z_order) == 0:
            self._mesh.triangles_z_order = np.empty((0), dtype=int)
        else:
            _, idx, counts = np.unique(
                self._mesh.triangles_index[:, 0],
                return_index=True,
                return_counts=True,
            )
            triangles_z_order = [
                np.arange(idx[z], idx[z] + counts[z]) for z in self._z_order
            ]
            self._mesh.triangles_z_order = np.concatenate(triangles_z_order)
        self._update_displayed()

    def edit(
        self, index, data, face_color=None, edge_color=None
    ):
        """Updates the data of a single bounding box located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        data : np.ndarray
            NxD array of vertices.
        """

        bounding_box = self.bounding_boxes[index]
        bounding_box.data = data

        if face_color is not None:
            self._face_color[index] = face_color
        if edge_color is not None:
            self._edge_color[index] = edge_color

        self.remove(index, renumber=False)
        self.add(bounding_box, bounding_box_index=index)
        self._update_z_order()

    def update_edge_width(self, index, edge_width):
        """Updates the edge width of a single bounding box located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        edge_width : float
            thickness of lines and edges.
        """
        self.bounding_boxes[index].edge_width = edge_width
        self._update_mesh_vertices(index, edge=True)

    def update_edge_color(self, index, edge_color, update=True):
        """Updates the edge color of a single bounding box located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        edge_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        update : bool
            If True, update the mesh with the new color property. Set to False to avoid
            repeated updates when modifying multiple bounding boxes. Default is True.
        """
        self._edge_color[index] = edge_color
        indices = np.all(self._mesh.triangles_index == [index, 1], axis=1)
        self._mesh.triangles_colors[indices] = self._edge_color[index]
        if update:
            self._update_displayed()

    def update_edge_colors(self, indices, edge_colors, update=True):
        """same as update_edge_color() but for multiple indices/edgecolors at once"""
        self._edge_color[indices] = edge_colors
        all_indices = np.bitwise_and(
            np.isin(self._mesh.triangles_index[:, 0], indices),
            self._mesh.triangles_index[:, 1] == 1,
        )
        self._mesh.triangles_colors[all_indices] = self._edge_color[
            self._mesh.triangles_index[all_indices, 0]
        ]
        if update:
            self._update_displayed()

    def update_face_color(self, index, face_color, update=True):
        """Updates the face color of a single bounding box located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        face_color : str | tuple
            If string can be any color name recognized by vispy or hex value if
            starting with `#`. If array-like must be 1-dimensional array with 3
            or 4 elements.
        update : bool
            If True, update the mesh with the new color property. Set to False to avoid
            repeated updates when modifying multiple bounding boxes. Default is True.
        """
        self._face_color[index] = face_color
        indices = np.all(self._mesh.triangles_index == [index, 0], axis=1)
        self._mesh.triangles_colors[indices] = self._face_color[index]
        if update:
            self._update_displayed()

    def update_face_colors(self, indices, face_colors, update=True):
        """same as update_face_color() but for multiple indices/facecolors at once"""
        self._face_color[indices] = face_colors
        all_indices = np.bitwise_and(
            np.isin(self._mesh.triangles_index[:, 0], indices),
            self._mesh.triangles_index[:, 1] == 0,
        )
        self._mesh.triangles_colors[all_indices] = self._face_color[
            self._mesh.triangles_index[all_indices, 0]
        ]
        if update:
            self._update_displayed()

    def update_dims_order(self, dims_order):
        """Updates dimensions order for all bounding boxes.

        Parameters
        ----------
        dims_order : (D,) list
            Order that the dimensions are rendered in.
        """
        for index in range(len(self.bounding_boxes)):
            if not self.bounding_boxes[index].dims_order == dims_order:
                bounding_box = self.bounding_boxes[index]
                bounding_box.dims_order = dims_order
                self.remove(index, renumber=False)
                self.add(bounding_box, bounding_box_index=index)
        self._update_z_order()

    def update_z_index(self, index, z_index):
        """Updates the z order of a single bounding box located at index.

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        z_index : int
            Specifier of z order priority. Bounding boxes with higher z order are
            displayed ontop of others.
        """
        self.bounding_boxes[index].z_index = z_index
        self._z_index[index] = z_index
        self._update_z_order()

    def shift(self, index, shift):
        """Performs a 2D shift on a single bounding box located at index

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        shift : np.ndarray
            length 2 array specifying shift of bounding boxes.
        """
        self.bounding_boxes[index].shift(shift)
        self._update_mesh_vertices(index, edge=True, face=True)

    def scale(self, index, scale, center=None):
        """Performs a scaling on a single bounding box located at index

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        scale : float, list
            scalar or list specifying rescaling of bounding box.
        center : list
            length 2 list specifying coordinate of center of scaling.
        """
        self.bounding_boxes[index].scale(scale, center=center)
        bounding_box = self.bounding_boxes[index]
        self.remove(index, renumber=False)
        self.add(bounding_box, bounding_box_index=index)
        self._update_z_order()

    def flip(self, index, axis, center=None):
        """Performs an vertical flip on a single bounding box located at index

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        axis : int
            integer specifying axis of flip. `0` flips horizontal, `1` flips
            vertical.
        center : list
            length 2 list specifying coordinate of center of flip axes.
        """
        self.bounding_boxes[index].flip(axis, center=center)
        self._update_mesh_vertices(index, edge=True, face=True)

    def transform(self, index, transform):
        """Performs a linear transform on a single bounding box located at index

        Parameters
        ----------
        index : int
            Location in list of the bounding box to be changed.
        transform : np.ndarray
            2x2 array specifying linear transform.
        """
        self.bounding_boxes[index].transform(transform)
        bounding_box = self.bounding_boxes[index]
        self.remove(index, renumber=False)
        self.add(bounding_box, bounding_box_index=index)
        self._update_z_order()

    def outline(self, indices):
        """Finds outlines of bounding boxes listed in indices

        Parameters
        ----------
        indices : int | list
            Location in list of the bounding boxes to be outline. If list must be a
            list of int

        Returns
        -------
        centers : np.ndarray
            Nx2 array of centers of outline
        offsets : np.ndarray
            Nx2 array of offsets of outline
        triangles : np.ndarray
            Mx3 array of any indices of vertices for triangles of outline
        """
        if type(indices) is list:
            meshes = self._mesh.triangles_index
            triangle_indices = [
                i
                for i, x in enumerate(meshes)
                if x[0] in indices and x[1] == 1
            ]
            meshes = self._mesh.vertices_index
            vertices_indices = [
                i
                for i, x in enumerate(meshes)
                if x[0] in indices and x[1] == 1
            ]
        else:
            triangle_indices = np.all(
                self._mesh.triangles_index == [indices, 1], axis=1
            )
            triangle_indices = np.where(triangle_indices)[0]
            vertices_indices = np.all(
                self._mesh.vertices_index == [indices, 1], axis=1
            )
            vertices_indices = np.where(vertices_indices)[0]

        offsets = self._mesh.vertices_offsets[vertices_indices]
        centers = self._mesh.vertices_centers[vertices_indices]
        triangles = self._mesh.triangles[triangle_indices]

        if type(indices) is list:
            t_ind = self._mesh.triangles_index[triangle_indices][:, 0]
            inds = self._mesh.vertices_index[vertices_indices][:, 0]
            starts = np.unique(inds, return_index=True)[1]
            for i, ind in enumerate(indices):
                inds = t_ind == ind
                adjust_index = starts[i] - vertices_indices[starts[i]]
                triangles[inds] = triangles[inds] + adjust_index
        else:
            triangles = triangles - vertices_indices[0]

        return centers, offsets, triangles

    def bounding_boxes_in_box(self, corners):
        """Determines which bounding boxes, if any, are inside an axis aligned box.

        Looks only at displayed bounding boxes

        Parameters
        ----------
        corners : np.ndarray
            2x2 array of two corners that will be used to create an axis
            aligned box.

        Returns
        -------
        bounding_boxes : list
            List of bounding boxes that are inside the box.
        """
        if len(self._mesh.vertices) == 0:
            return []
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        intersects = triangles_intersect_box(triangles, corners)
        bounding_boxes = self._mesh.displayed_triangles_index[intersects, 0]
        bounding_boxes = np.unique(bounding_boxes).tolist()

        return bounding_boxes

    def inside(self, coord):
        """Determines if any bounding box at given coord by looking inside triangle
        meshes. Looks only at displayed bounding boxes

        Parameters
        ----------
        coord : sequence of float
            Image coordinates to check if any bounding boxes are at.

        Returns
        -------
        bounding_box : int | None
            Index of bounding box if any that is at the coordinates. Returns `None`
            if no bounding box is found.
        """
        if len(self._mesh.vertices) == 0:
            return None
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        indices = inside_triangles(triangles - coord)
        bounding_boxes = self._mesh.displayed_triangles_index[indices, 0]

        if len(bounding_boxes) == 0:
            return None

        z_list = self._z_order.tolist()
        order_indices = np.array([z_list.index(m) for m in bounding_boxes])
        ordered_bounding_boxes = bounding_boxes[np.argsort(order_indices)]
        return ordered_bounding_boxes[0]

    def _inside_3d(self, ray_position: np.ndarray, ray_direction: np.ndarray):
        """Determines if any bounding box is intersected by a ray by looking inside triangle
        meshes. Looks only at displayed bounding boxes.

        Parameters
        ----------
        ray_position : np.ndarray
            (3,) array containing the location that was clicked. This
            should be in the same coordinate system as the vertices.
        ray_direction : np.ndarray
            (3,) array describing the direction camera is pointing in
            the scene. This should be in the same coordinate system as
            the vertices.

        Returns
        -------
        bounding_box : int | None
            Index of bounding_box if any that is at the coordinates. Returns `None`
            if no bounding_box is found.
        intersection_point : Optional[np.ndarray]
            The point where the ray intersects the mesh face. If there was
            no intersection, returns None.
        """
        if len(self._mesh.vertices) == 0:
            return None, None
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        inside = line_in_triangles_3d(
            line_point=ray_position,
            line_direction=ray_direction,
            triangles=triangles,
        )
        intersected_bounding_boxes = self._mesh.displayed_triangles_index[inside, 0]
        if len(intersected_bounding_boxes) == 0:
            return None, None

        intersection_points = self._triangle_intersection(
            triangle_indices=inside,
            ray_position=ray_position,
            ray_direction=ray_direction,
        )
        start_to_intersection = intersection_points - ray_position
        distances = np.linalg.norm(start_to_intersection, axis=1)
        closest_bounding_box_index = np.argmin(distances)
        bounding_box = intersected_bounding_boxes[closest_bounding_box_index]
        intersection = intersection_points[closest_bounding_box_index]
        return bounding_box, intersection

    def _triangle_intersection(
        self,
        triangle_indices: np.ndarray,
        ray_position: np.ndarray,
        ray_direction: np.ndarray,
    ):
        """Find the intersection of a ray with specified triangles.

        Parameters
        ----------
        triangle_indices : np.ndarray
            (n,) array of bounding box indices to find the intersection with the ray. The indices should
            correspond with self._mesh.displayed_triangles.
        ray_position : np.ndarray
            (3,) array with the coordinate of the starting point of the ray in layer coordinates.
            Only provide the 3 displayed dimensions.
        ray_direction : np.ndarray
            (3,) array of the normal direction of the ray in layer coordinates.
            Only provide the 3 displayed dimensions.

        Returns
        -------
        intersection_points : np.ndarray
            (n x 3) array of the intersection of the ray with each of the specified bounding boxes in layer coordinates.
            Only the 3 displayed dimensions are provided.
        """
        if len(self._mesh.vertices) == 0:
            return np.empty((0, 3))
        triangles = self._mesh.vertices[self._mesh.displayed_triangles]
        intersected_triangles = triangles[triangle_indices]
        intersection_points = intersect_line_with_triangles(
            line_point=ray_position,
            line_direction=ray_direction,
            triangles=intersected_triangles,
        )
        return intersection_points

    def to_masks(self, mask_shape=None, zoom_factor=1, offset=(0, 0)):
        """Returns N binary masks, one for each bounding_box, embedded in an array of
        shape `mask_shape`.

        Parameters
        ----------
        mask_shape : np.ndarray | tuple | None
            2-tuple defining shape of mask to be generated. If non specified,
            takes the max of all the vertices
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        masks : (N, M, P) np.ndarray
            Array where there is one binary mask of shape MxP for each of
            N shapes
        """
        if mask_shape is None:
            mask_shape = self.displayed_vertices.max(axis=0).astype('int')

        masks = np.array(
            [
                s.to_mask(mask_shape, zoom_factor=zoom_factor, offset=offset)
                for s in self.shapes
            ]
        )

        return masks

    def to_labels(self, labels_shape=None, zoom_factor=1, offset=(0, 0)):
        """Returns a integer labels image, where each shape is embedded in an
        array of shape labels_shape with the value of the index + 1
        corresponding to it, and 0 for background. For overlapping shapes
        z-ordering will be respected.

        Parameters
        ----------
        labels_shape : np.ndarray | tuple | None
            2-tuple defining shape of labels image to be generated. If non
            specified, takes the max of all the vertices
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.

        Returns
        -------
        labels : np.ndarray
            MxP integer array where each value is either 0 for background or an
            integer up to N for points inside the corresponding shape.
        """
        if labels_shape is None:
            labels_shape = self.displayed_vertices.max(axis=0).astype(int)

        labels = np.zeros(labels_shape, dtype=int)

        for ind in self._z_order[::-1]:
            mask = self.shapes[ind].to_mask(
                labels_shape, zoom_factor=zoom_factor, offset=offset
            )
            labels[mask] = ind + 1

        return labels

    def to_colors(
        self, colors_shape=None, zoom_factor=1, offset=[0, 0], max_bounding_boxes=None
    ):
        # TODO: check if works
        """Rasterize bounding boxes to an RGBA image array.

        Each bounding box is embedded in an array of shape `colors_shape` with the
        RGBA value of the bounding box, and 0 for background. For overlapping bounding boxes
        z-ordering will be respected.

        Parameters
        ----------
        colors_shape : np.ndarray | tuple | None
            2-tuple defining shape of colors image to be generated. If non
            specified, takes the max of all the vertiecs
        zoom_factor : float
            Premultiplier applied to coordinates before generating mask. Used
            for generating as downsampled mask.
        offset : 2-tuple
            Offset subtracted from coordinates before multiplying by the
            zoom_factor. Used for putting negative coordinates into the mask.
        max_bounding_boxes : None | int
            If provided, this is the maximum number of bounding boxes that will be rasterized.
            If the number of bounding boxes in view exceeds max_bounding_boxes, max_bounding_boxes bounding boxes
            will be randomly selected from the in view bounding boxes. If set to None, no
            maximum is applied. The default value is None.

        Returns
        -------
        colors : (N, M, 4) array
            rgba array where each value is either 0 for background or the rgba
            value of the bounding box for points inside the corresponding bounding box.
        """
        if colors_shape is None:
            colors_shape = self.displayed_vertices.max(axis=0).astype(int)

        colors = np.zeros((*colors_shape, 4), dtype=float)
        colors[..., 3] = 1

        z_order = self._z_order[::-1]
        bounding_boxes_in_view = np.argwhere(self._displayed)
        z_order_in_view_mask = np.isin(z_order, bounding_boxes_in_view)
        z_order_in_view = z_order[z_order_in_view_mask]

        # If there are too many bounding boxes to render responsively, just render
        # the top max_bounding_boxes bounding boxes
        if max_bounding_boxes is not None and len(z_order_in_view) > max_bounding_boxes:
            z_order_in_view = z_order_in_view[0:max_bounding_boxes]

        for ind in z_order_in_view:
            mask = self.bounding_boxes[ind].to_mask(
                colors_shape, zoom_factor=zoom_factor, offset=offset
            )
            col = self._face_color[ind]
            colors[mask, :] = col

        return colors
