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

#ifndef AS_AREA_GRID_REGULAR_H
#define AS_AREA_GRID_REGULAR_H

#include "asAreaGrid.h"
#include "asIncludes.h"

class asAreaGridRegular : public asAreaGrid {
  public:
    asAreaGridRegular(const Coo& cornerUL, const Coo& cornerUR, const Coo& cornerLL, const Coo& cornerLR, double xStep,
                      double yStep, int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaGridRegular(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                      int flatAllowed = asFLAT_FORBIDDEN, bool isLatLon = true);

    asAreaGridRegular(double xMin, int xPtsNb, double yMin, int yPtsNb, int flatAllowed = asFLAT_FORBIDDEN,
                      bool isLatLon = true);

    ~asAreaGridRegular() override = default;

    bool GridsOverlay(asAreaGrid* otherArea) const override;

    bool InitializeAxes(const a1d& lons, const a1d& lats, bool strideAllowed = true, bool getLarger = false) override;

    double GetXstep() const override {
        return _xStep;
    }

    double GetYstep() const override {
        return _yStep;
    }

    double GetYstepData() const {
        return _yStepData;
    }

    int GetXstepStride() const {
        wxASSERT(_xStep > 0);
        wxASSERT(fmod(_xStep, _xStepData) == 0);
        return int(_xStep / _xStepData);
    }

    int GetYstepStride() const {
        wxASSERT(_yStep > 0);
        wxASSERT(fmod(_yStep, _yStepData) == 0);
        return int(_yStep / _yStepData);
    }

    void SetSameStepAsData() {
        _xStep = _xStepData;
        _yStep = _yStepData;
    }

  protected:
  private:
    double _xStep;
    double _yStep;
    double _xStepData;
    double _yStepData;
};

#endif
