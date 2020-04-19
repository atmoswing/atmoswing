[![AtmoSwing](https://raw.githubusercontent.com/atmoswing/atmoswing/master/art/logo/logo.png)](http://www.atmoswing.org)

[![DOI](https://zenodo.org/badge/95885904.svg)](https://zenodo.org/badge/latestdoi/95885904) 
[![Build Status](https://travis-ci.org/atmoswing/atmoswing.svg?branch=master)](https://travis-ci.org/atmoswing/atmoswing)
[![Build status](https://ci.appveyor.com/api/projects/status/1otp6de8c8go0nxm?svg=true)](https://ci.appveyor.com/project/pascalhorton/atmoswing)
[![codecov](https://codecov.io/gh/atmoswing/atmoswing/branch/master/graph/badge.svg)](https://codecov.io/gh/atmoswing/atmoswing)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/13133/badge.svg)](https://scan.coverity.com/projects/atmoswing-atmoswing)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/1107/badge)](https://bestpractices.coreinfrastructure.org/projects/1107)
[![Documentation Status](https://readthedocs.org/projects/atmoswing/badge/?version=latest)](https://atmoswing.readthedocs.io/en/latest/?badge=latest)


Analog methods (AMs) allow predicting local meteorological variables of interest (predictand), such as the daily precipitation, based on synoptic variables (predictors). They rely on the hypothesis that similar atmospheric conditions are likely to result in similar local effects. The statistical relationship is first defined (e.g. which predictors, and how many subsampling steps) and calibrated (e.g. which spatial domain, and how many analogues) before being applied to the target period, may it be for operational forecasting or for climate impact studies. A benefit of AMs is that they are lightweight and can provide valuable results for a negligible cost.

AtmoSwing is an open source (CDDL-1.0) software that implements different AM variants in a very flexible way, so that they can be easily configured by means of XML files. It is written in C++, is object-oriented and multi-platform. AtmoSwing provides four tools: the Optimizer to establish the relationship between the predictand and predictors, the Downscaler to apply the method for climate impact studies, the Forecaster to perform operational forecasts, and the Viewer to display the results. 

The Optimizer provides a semi-automatic sequential approach, as well as Monte-Carlo analyses, and a global optimization technique by means of Genetic Algorithms. It calibrates the statistical relationship that can be later applied in a forecasting or climatic context.

The Downscaler takes as input the outputs of climate models, either GCMs or RCMs in order to provide a downscaled time series of the predictand of interest at a local scale.

The Forecaster automatically downloads and reads operational NWP outputs to provide operational forecasting of the predictand of interest. The processing of a forecast is extremely lightweight in terms of computing resources; it can indeed run on almost any computer.

The Viewer displays the forecasts in an interactive GIS environment. It contains several layers of syntheses and details in order to provide a quick overview of the potential critical situations in the coming days, as well as the possibility for the user to go into the details of the forecasted predictand distribution.

## What's in there ##

This repository contains 3 separate tools:

* The Forecaster: automatically processes the forecast
* The Viewer: displays the resulting files in a GIS environment
* The Optimizer: optimizes the method for a given precipitation timeseries
* The Downscaler: downscaling for climate impact studies

Additionally, multiple unit tests are available and are built along with the software. It is highly recommended to run these tests before using AtmoSwing operationally.

## Documentation ##

AtmoSwing documentation can be found here: https://atmoswing.readthedocs.io/en/latest/

The repository of the documentation is https://github.com/atmoswing/user-manual

## Docker images ##

AtmoSwing Forecaster image: https://hub.docker.com/repository/docker/atmoswing/forecaster

AtmoSwing Optimizer image: https://hub.docker.com/repository/docker/atmoswing/optimizer

AtmoSwing Downscaler image: https://hub.docker.com/repository/docker/atmoswing/downscaler

## Download AtmoSwing ##

You can download the releases under: https://github.com/atmoswing/atmoswing/releases

Nightly (experimental) automatic builds are available for:

* Linux (Ubuntu) & osx: https://console.cloud.google.com/storage/browser/atmoswing-deploy
* Windows: https://ci.appveyor.com/project/pascalhorton/atmoswing

## How to build AtmoSwing ##

The wiki (https://github.com/atmoswing/atmoswing/wiki) explains how to compile the required libraries and the source code of AtmoSwing. In order to get AtmoSwing compiled, follow these steps:

1. [Get the required **libraries**](https://github.com/atmoswing/atmoswing/wiki/Libraries)
3. [**Configure / build** with CMake](https://github.com/atmoswing/atmoswing/wiki/Build)

## How to contribute ##

If you want to contribute to the software development, you can fork this repository (keep it public !) and then suggest your improvements by sending pull requests. We would be glad to see a community growing around this project.

When adding a new feature, please write a test along with it.

Additionally, you can report issues or suggestions in the issues tracker (https://github.com/atmoswing/atmoswing/issues).

AtmoSwing will follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) (not the case so far) with a few differences (mainly based on [wxWidgets Coding Guidelines](https://www.wxwidgets.org/develop/coding-guidelines)):
* Use ``CamelCase`` for types (classes, structs, enums, unions), methods and functions 
* Use ``camelCase`` for the variables.
* Use ``m_`` prefix for member variables.
* Global variables shouldn’t normally be used at all, but if they are, should have ``g_`` prefix.
* Use Set/Get prefixes for accessors

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
