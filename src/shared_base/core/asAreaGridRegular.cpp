/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asAreaGridRegular.h"
#include "asIncludes.h"

#include <cmath>

asAreaGridRegular::asAreaGridRegular(const Coo& cornerUL, const Coo& cornerUR, const Coo& cornerLL, const Coo& cornerLR,
                                     double xStep, double yStep, int flatAllowed, bool isLatLon)
    : asAreaGrid(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed, isLatLon),
      _xStep(xStep),
      _yStep(yStep) {
    _isRegular = true;
    _xStepData = 0;
    _yStepData = 0;
}

asAreaGridRegular::asAreaGridRegular(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                                     int flatAllowed, bool isLatLon)
    : asAreaGrid(xMin, xWidth, yMin, yWidth, flatAllowed, isLatLon),
      _xStep(xStep),
      _yStep(yStep) {
    _isRegular = true;
    _xStepData = 0;
    _yStepData = 0;
}

asAreaGridRegular::asAreaGridRegular(double xMin, int xPtsNb, double yMin, int yPtsNb, int flatAllowed, bool isLatLon)
    : asAreaGrid(xMin, 0, yMin, 0, flatAllowed, isLatLon),
      _xStep(0),
      _yStep(0) {
    _isRegular = true;
    _xPtsNb = xPtsNb;
    _yPtsNb = yPtsNb;
    _xStepData = 0;
    _yStepData = 0;
}

bool asAreaGridRegular::GridsOverlay(asAreaGrid* otherArea) const {
    if (!otherArea->IsRegular()) return false;

    auto otherAreaRegular(dynamic_cast<asAreaGridRegular*>(otherArea));

    if (!otherAreaRegular) return false;

    if (std::fabs(GetXstep() - otherAreaRegular->GetXstep()) > 1e-10) return false;

    if (std::fabs(GetYstep() - otherAreaRegular->GetYstep()) > 1e-10) return false;

    return true;
}

bool asAreaGridRegular::InitializeAxes(const a1d& lons, const a1d& lats, bool strideAllowed, bool getLarger) {
    wxASSERT(lons.size() > 1);
    wxASSERT(lats.size() > 1);

    _xStepData = std::abs(lons[1] - lons[0]);
    _yStepData = std::abs(lats[1] - lats[0]);

    if (_xStep == 0) {
        _xStep = _xStepData;
    }
    if (_yStep == 0) {
        _yStep = _yStepData;
    }

    bool needsLarger = false;

    if (_xStep != _xStepData || _yStep != _yStepData) {
        if (std::fmod(_xStep, _xStepData) != 0 || std::fmod(_yStep, _yStepData) != 0) {
            needsLarger = true;
        }
    }

    if (!asAreaGrid::InitializeAxes(lons, lats, true, needsLarger)) {
        return false;
    }

    if (!strideAllowed) {
        return true;
    }

    if (_xStep != _xStepData) {
        if (std::fmod(_xStep, _xStepData) == 0) {
            auto xFactor = (int)(_xStep / _xStepData);

            a1d lonsC0 = _xAxis;
            auto sizeC0new = (int)((lonsC0.rows() - 1) / xFactor) + 1;
            _xAxis.resize(sizeC0new);

            for (int i = 0; i < sizeC0new; ++i) {
                _xAxis[i] = lonsC0[i * xFactor];
            }

        } else {
            double xMin = GetXmin();
            double xMax = GetXmax();
            auto xSize = (int)std::floor((xMax - xMin) / _xStep) + 1;
            _xAxis.resize(xSize);

            wxASSERT(xMin < xMax);
            wxASSERT(_xStep > 0);
            wxASSERT(xSize > 0);

            for (int i = 0; i < xSize; ++i) {
                _xAxis[i] = xMin + i * _xStep;
            }
        }
    }

    if (_yStep != _yStepData) {
        if (std::fmod(_yStep, _yStepData) == 0) {
            auto yFactor = (int)(_yStep / _yStepData);

            a1d latsC0 = _yAxis;
            auto sizeC0new = (int)((latsC0.rows() - 1) / yFactor) + 1;
            _yAxis.resize(sizeC0new);

            for (int i = 0; i < sizeC0new; ++i) {
                _yAxis[i] = latsC0[i * yFactor];
            }

        } else {
            double yMin = GetYmin();
            double yMax = GetYmax();
            auto ySize = (int)std::floor((yMax - yMin) / _yStep) + 1;
            _yAxis.resize(ySize);

            wxASSERT(yMin < yMax);
            wxASSERT(_yStep > 0);
            wxASSERT(ySize > 0);

            for (int i = 0; i < ySize; ++i) {
                _yAxis[i] = yMin + i * _yStep;
            }
        }
    }

    return true;
}
