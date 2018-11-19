[![AtmoSwing](https://raw.githubusercontent.com/atmoswing/atmoswing/master/art/logo/logo.png)](http://www.atmoswing.org)

[![DOI](https://zenodo.org/badge/95885904.svg)](https://zenodo.org/badge/latestdoi/95885904) 
[![Build Status](https://travis-ci.org/atmoswing/atmoswing.svg?branch=master)](https://travis-ci.org/atmoswing/atmoswing)
[![Build status](https://ci.appveyor.com/api/projects/status/1otp6de8c8go0nxm?svg=true)](https://ci.appveyor.com/project/pascalhorton/atmoswing)
[![codecov](https://codecov.io/gh/atmoswing/atmoswing/branch/master/graph/badge.svg)](https://codecov.io/gh/atmoswing/atmoswing)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/13133/badge.svg)](https://scan.coverity.com/projects/atmoswing-atmoswing)

AtmoSwing stands for Analog Technique Model for Statistical weather forecastING. The software allows for real-time precipitation forecasting based on a downscaling method, the analogue technique. It identifies analogue days, in terms of atmospheric circulation and humidity variables, in a long archive of past situations and then uses the corresponding measured precipitation to establish an empirical conditional distribution considered as the probabilistic forecast for the target day. This method is used in different institutions for hydro-meteorological forecasting in the framework of real-time flood management or electricity production.

The model is standalone and automatically handles the download of the GFS global numerical weather prediction forecasts on which the analogy is processed. The development aimed at creating a very modular object-oriented tool that can be used to parameterize any known version of the analogue method. There is no limitation on the number of analogy steps, neither on the number of atmospheric variables used as input.

The software is written in C++, is cross-platform and open source (under the Common Development and Distribution License Version 1.0 (CDDL-1.0), which can be found in the accompanying license.txt file).

## What's in there ##

This repository contains 3 separate tools:

* The Forecaster: automatically processes the forecast
* The Viewer: displays the resulting files in a GIS environment
* The Optimizer: optimizes the method for a given precipitation timeseries
* The Downscaler: downscaling for climate impact studies

Additionally, multiple unit tests are available and are built along with the software. It is highly recommended to run these tests before using AtmoSwing operationally.

## Download AtmoSwing ##

You can download the releases under: https://github.com/atmoswing/atmoswing/releases

Nightly (experimental) automatic builds are available for:

* Linux (Ubuntu) & osx: https://console.cloud.google.com/storage/browser/atmoswing-deploy
* Windows: https://ci.appveyor.com/project/pascalhorton/atmoswing

## How to build AtmoSwing ##

The wiki (https://github.com/atmoswing/atmoswing/wiki) explains how to compile the required libraries and the source code of AtmoSwing. In order to get Atmoswing compiled, follow these steps:

1. [Get the required **libraries**](https://github.com/atmoswing/atmoswing/wiki/Libraries)
3. [**Configure / build** with CMake](https://github.com/atmoswing/atmoswing/wiki/Build)
4. [**Compile**](https://github.com/atmoswing/atmoswing/wiki/Compile)
5. [**Test**](https://github.com/atmoswing/atmoswing/wiki/Test)
6. [**Install**](https://github.com/atmoswing/atmoswing/wiki/Install)

## How to contribute ##

If you want to contribute to the software development, you can fork this repository (keep it public !) and then suggest your improvements by sending pull requests. We would be glad to see a community growing around this project.

Additionally, you can report issues or suggestions in the issues tracker (https://github.com/atmoswing/atmoswing/issues).

## Credits ##

[![University of Lausanne](https://raw.githubusercontent.com/atmoswing/atmoswing/master/art/misc/logo-Unil.png)](http://unil.ch/iste) 
&nbsp;&nbsp;&nbsp;&nbsp;
[![Terranum](https://raw.githubusercontent.com/atmoswing/atmoswing/master/art/misc/logo-Terranum.png)](http://terranum.ch) 
&nbsp;&nbsp;&nbsp;&nbsp;
[![University of Bern](https://raw.githubusercontent.com/atmoswing/atmoswing/master/art/misc/logo-Unibe.png)](http://www.geography.unibe.ch/) 

Copyright (C) 2007-2013, [University of Lausanne](http://unil.ch/iste), Switzerland.

Copyright (C) 2013-2015, [Terranum](http://terranum.ch), Switzerland.

Copyright (C) 2016-2017, [University of Bern](http://www.geography.unibe.ch/), Switzerland.

Contributions:

* Developed by Pascal Horton
* Under the supervision of Charles Obled and Michel Jaboyedoff
* With inputs from Lucien Schreiber, Richard Metzger and Renaud Marty

Financial contributions:

* 2008-2011 Cantons of Valais and Vaud (Switzerland): basis of the software from the MINERVE project.
* 2011-2013 University of Lausanne (Switzerland): reorganization of the source code, improvement of the build system, documentation.
* 2014 Direction régionale de l’environnement, de l’aménagement et du logement (France): addition of new forecast skill scores (reliability of the CRPS and rank histogram).
* 2015 Cantons of Valais (Switzerland): addition of synthetic xml export and the aggregation of parametrizations in the viewer.

See both license.txt and notice.txt files for details about the license and its enforcement.
