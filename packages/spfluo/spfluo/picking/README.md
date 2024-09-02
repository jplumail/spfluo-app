<!--
+----------------------------------------------------------------------------------------------+
|                                                                                              |
|                                               MAIN TITLE                                     |
|                                                                                              |
+----------------------------------------------------------------------------------------------+
-->

# Stage Picking Fluo    



<!--
+----------------------------------------------------------------------------------------------+
|                                                                                              |
|                                          TABLE OF CONTENTS                                   |
|                                                                                              |
+----------------------------------------------------------------------------------------------+
 -->

# Sommaire

- [Source code scheme](#source-code-scheme)
- [Setup](#setup-an-environment-to-use-this-code)



# Source code scheme

![](./assets/codebase_scheme.png)



# Setup an environment to use this code

1. install dependencies
```
pip install -r requirements.txt
```

2. Data generation relies on a C++ library that needs to be set up:
    1. run:
    ```
    python modules/pretraining/generate/setup.py --action setup
    ```
    2. close and re-open terminal or run 
    ```
    source ~/.bashrc
    ```
    3. check setup went well:
    ```
    python modules/pretraining/generate/setup.py --action check
    ```


# Pipeline 

This code is organized in 6 stages, grouped into 3 submodules, as bellow.
<br />
It is organized with the following design in mind:
    - It launches a pipeline, consisting of any combination of stages.
    - Every stages are inside modules and should never be called directely. 
    - The only code to eventually be executed should be `main.py`: it will call any required submodules.

![](./assets/pipeline.png)


# Data generation

The data generation is illustrated in the available notebooks.
