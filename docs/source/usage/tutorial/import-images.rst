Importing images
----------------

.. image:: ../../_static/empty-project.png

What you see on the left is all the protocols that will populate our workflow. We have:

 * imports protocols, used to imports stuff inside the software.
 * picking protocols, to detect particles in the images.
 * reconstruciton protocols, to reconstruct a model.
 * tools, a diverse set of algorithms.

Double-click on the protocol ``import fluoimages``. It should open a window.

.. image:: ../../_static/import-files.png

In the *Files directory* field, select the ``isim`` folder that was given with **spfluo-app**. This folder contains the data you will be working with:

.. code-block:: text

    spfluo-app
    └── examples
        └── uEXM
            ├── FOV_1_MMStack_Pos0.ome.tiff
            ├── FOV_2_MMStack_Pos0.ome.tiff
            ├── FOV_3_MMStack_Pos0.ome.tiff
            ├── ...
            └── psf.tiff

The images to import are named ``FOV_XX_MMStack_Pos0.ome.tiff``. To match them, we fill in the *pattern* field with ``FOV_*.tiff``. This will match all the files that starts with ``FOV_`` and ends with ``.tiff``.

Choose ``Automatic`` for the reader, this will work with our images.

The images have a pixel of size 56nm x 56nm x 150nm. We fill in the acquisition infos in micrometers accordingly.

.. note::

    The wizard button can be used to retrieve the infos from the metadata of the images. However, it will not always work! (it doesn't work for our images)

    While OME-TIFF format is well supported by **spfluo-app**, other TIFF formats might not be well read: for instance, the Z dimension may miss, or may be replaced by a temporal dimension.
    The field ``transpose T<>Z axes when possible?`` enables autocorrect of the dimensions. 

.. image:: ../../_static/import-files-filled.png

Then, click on *Execute*.

After some time, the import should be done and the protocol box should turn green. On the bottom of the screen, a panel is summarizing the protocol. In the *Output* section, a ``SetOfFluoImages`` object is displayed. This object represents the images you imported. You can right-click on it to see the available viewers.

.. image:: ../../_static/output-right-click-napariviewer.png

.. note::
    
    Almost all the protocols you will use output objects in the *Output* section. You can visualise them with any of the available viewers.

Visualise the data you imported with napari in the next section.