# AtmoSwing #

AtmoSwing stands for Analog Technique Model for Statistical weather forecastING. The software allows for real-time precipitation forecasting based on a downscaling method, the analogue technique. It identifies analogue days, in terms of atmospheric circulation and humidity variables, in a long archive of past situations and then uses the corresponding measured precipitation to establish an empirical conditional distribution considered as the probabilistic forecast for the target day. This method is used in different institutions for hydro-meteorological forecasting in the framework of real-time flood management or electricity production.

The model is standalone and automatically handles the download of the GFS global numerical weather prediction forecasts on which the analogy is processed. The development aimed at creating a very modular object oriented tool that can be used to parameterize any known version of the analogue method. There is no limitation on the number of analogy steps, neither on the amount of atmospheric variables used as input.

The software is written in C++, is cross-platform and open source (under the Common Development and Distribution License Version 1.0 (CDDL-1.0), which can be found in the accompanying license.txt file).

## What's in there ##

This repository contains 2 separate tools:
- The Forecaster: automatically processes the forecast (build files in "build/forecaster")
- The Viewer: displays the resulting files in a GIS environment (build files in "build/viewer")

Additionally, multiple unit tests are available in the "test" directory. It is highly recommended to run these tests before using AtmoSwing operationally.

## How to build AtmoSwing ##

The wiki (https://bitbucket.org/atmoswing/atmoswing/wiki/Home) explains how to compile the required libraries and the source code of AtmoSwing. It is highly recommended to run the unit tests before using AtmoSwing operationally.

## How to contribute ##

If you want to contribute to the software development, you can fork this repository and then suggest your improvements by sending pull requests. We would be glad to see a community growing around this project.

Additionally, you can report issues or suggestions in the issues tracker (https://bitbucket.org/atmoswing/atmoswing/issues).

## Credits ##

Copyright (C) 2007-2013, University of Lausanne, Switzerland.
Developed by Pascal Horton.
Financed by the cantons of Valais and Vaud, Switzerland.
Resulting from the MINERVE project (2011).
See both license.txt and notice.txt files for details about the license and its enforcement.

## Contact ##

You can contact the main developer by email: pascal.horton@unil.ch.
