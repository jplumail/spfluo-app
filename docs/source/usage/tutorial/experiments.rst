Conducting experiments
----------------------

When experimenting with the algorithms, you may want to change parameters or add particles to your picking.

This is possible with Scipion! There are 2 ways :
 - If you don't care about the previous results, you can restart a protocol with new parameters / inputs with the ``Run mode`` set to Restart.

   .. image:: ../../_static/scipion-runmode.png
   
   This will overwrite your previous run and the data will be lost.

 - If you want to keep the previous results, create another protocol. You can have as many protocols as you want in your project.
   
   .. image:: ../../_static/scipion-twoexp.png

   Renaming your protocols depending on the experiment you're doing might be a good idea. In the protocol windows, simply change the ``Run name``.

.. note::
    The ``manual picking`` protocol is *interactive* which means you can add particles without going into the restart mode.
    A new set of particles is saved each time you enter the manual picking.