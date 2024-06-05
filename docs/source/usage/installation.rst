Installation
============

The **spfluo-app** is available on :ref:`Windows <windows-section>` and :ref:`Linux <linux-section>`. A GPU is recommended but not necessary.

.. _windows-section:

On Windows
----------

1. Install Python
    a. Go to the `Python download page <https://www.python.org/downloads/>`_.
    b. Download Python 3.11.
    c. Run the executable.
    d. **Important**: tick the box `Add Python to PATH`.
    e. Click `Install Now`.

2. Install **spfluo-app**
    a. Unzip `spfluo-app.zip`
    b. Open the command line interpreter. To do so:
        i. Press the keys windows + R
        ii. Enter "cmd".
    c. Once the interpreter is opened, enter the following command::

        $ cd C:\path\to\spfluo-app\app
    
    The :command:`cd` command will move you to the desired folder.

    .. note::
        Please note that the dollar sign $ is indicative of being within the interpreter,
        and you should not include it when typing the command.
    
    d. Enter the :command:`dir` command. Make sure the following files are present:
        
        .. code-block:: text

            app/
            ├── requirements.txt
            └── spfluo-app.pyw
    
    e. Run the installation with the following commands:
        - If you have a GPU::

            $ py -m pip install -r requirements-gpu.txt
            $ py -m pip install --force-reinstall -r requirements-gpu-windows.txt
        - Otherwise::

            $ py -m pip install -r requirements-cpu.txt

.. 
    TODO: find the level of access needed!

3. Enable the `developer mode <https://learn.microsoft.com/windows/apps/get-started/enable-your-device-for-development#activate-developer-mode>`_:
    a. Press the ``Windows`` key and search for *developer*.
       From the *For developers settings* dialog, choose the level of access that you need. 
    
    b. Read the disclaimer for the setting you choose. Click Yes to accept the change.

The installation is now complete! You should now be able to double-click on the file ``spfluo-app.pyw``. It should open the Scipion launcher :numref:`[%s] <scipion-launcher>`.

You can create a shortcut:
    1. Right click on the file
    2. More options
    3. Send to the desktop


.. _linux-section:

On Linux
--------

1. Install Python
    Python should already be availble on your distribution. If you have Ubuntu for instance, the command `python` should be available.

2. Install **spfluo-app**
    a. Unzip `spfluo-app.zip`
    b. Open the terminal.
    c. Once the terminal is opened, enter the following command::

        $ cd /path/to/spfluo-app/app
    
    The :command:`cd` command will move you to the desired folder.
    
    d. Enter the :command:`ls` command. Make sure the following files are present:

        .. code-block:: text

            app/
            ├── requirements.txt
            └── spfluo-app.py


    e. Run the installation with the following commands:
        - If you have a GPU::

            $ python3 -m pip install -r requirements-gpu.txt
        - Otherwise::

            $ python3 -m pip install -r requirements-cpu.txt
            $ python3 -m pip install -r requirements-cpu-linux.txt

The installation is now complete! You should now be able to run ``python3 spfluo-app.pyw``. It should open the Scipion launcher :numref:`[%s] <scipion-launcher>`.

.. _scipion-launcher:

.. figure:: ../_static/scipion-launcher-empty.png
   :alt: scipion launcher empty
   :figwidth: 500px
   :figclass: align-center

   The Scipion launcher

Go to :doc:`tutorial/index` to get started with using **spfluo-app**.


Upgrade
-------
To upgrade your installation::

    $ python3 -m pip install --upgrade --upgrade-strategy eager -r requirements.txt