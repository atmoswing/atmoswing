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

#include "asAreaGrid.h"

#include <iostream>

#include "asAreaGenGrid.h"
#include "asAreaRegGrid.h"
#include "asParameters.h"
#include "asPredictor.h"
#include "asTypeDefs.h"

asAreaGrid* asAreaGrid::GetInstance(const asParameters* params, int iStep, int iPtor) {
    wxString gridType = params->GetPredictorGridType(iStep, iPtor);
    double xMin = params->GetPredictorXmin(iStep, iPtor);
    int xPtsNb = params->GetPredictorXptsnb(iStep, iPtor);
    double xStep = params->GetPredictorXstep(iStep, iPtor);
    double yMin = params->GetPredictorYmin(iStep, iPtor);
    int yPtsNb = params->GetPredictorYptsnb(iStep, iPtor);
    double yStep = params->GetPredictorYstep(iStep, iPtor);
    int flatAllowed = params->GetPredictorFlatAllowed(iStep, iPtor);
    bool isLatLon = asPredictor::IsLatLon(params->GetPredictorDatasetId(iStep, iPtor));

    return GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep, flatAllowed, isLatLon);
}

asAreaGrid* asAreaGrid::GetInstance(const wxString& type, double xMin, int xPtsNb, double xStep, double yMin,
                                    int yPtsNb, double yStep, int flatAllowed, bool isLatLon) {
    if (type.IsSameAs("Regular", false)) {
        if (xStep > 0 && yStep > 0) {
            double xWidth = (double)(xPtsNb - 1) * xStep;
            double yWidth = (double)(yPtsNb - 1) * yStep;
            return new asAreaRegGrid(xMin, xWidth, xStep, yMin, yWidth, yStep, flatAllowed, isLatLon);
        } else {
            return new asAreaRegGrid(xMin, xPtsNb, yMin, yPtsNb, flatAllowed, isLatLon);
        }
    } else if (type.IsEmpty() || type.IsSameAs("Generic", false)) {
        return new asAreaGenGrid(xMin, xPtsNb, yMin, yPtsNb, asFLAT_ALLOWED, isLatLon);
    } else {
        wxLogError(_("Given grid type: %s"), type);
        asThrowException(_("The given grid type doesn't correspond to any existing option."));
    }
}

asAreaGrid* asAreaGrid::GetInstance(double xMin, int xPtsNb, double xStep, double yMin, int yPtsNb, double yStep,
                                    int flatAllowed, bool isLatLon) {
    return GetInstance("Generic", xMin, xPtsNb, xStep, yMin, yPtsNb, yStep, flatAllowed, isLatLon);
}

asAreaGrid::asAreaGrid(const Coo& cornerUL, const Coo& cornerUR, const Coo& cornerLL, const Coo& cornerLR,
                       int flatAllowed, bool isLatLon)
    : asArea(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed, isLatLon),
      m_isRegular(false),
      m_isInitialized(false),
      m_allowResizeFromData(false),
      m_xPtsNb(0),
      m_yPtsNb(0) {}

asAreaGrid::asAreaGrid(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed, bool isLatLon)
    : asArea(xMin, xWidth, yMin, yWidth, flatAllowed, isLatLon),
      m_isRegular(false),
      m_isInitialized(false),
      m_allowResizeFromData(false),
      m_xPtsNb(0),
      m_yPtsNb(0) {}

asAreaGrid::asAreaGrid()
    : asArea(),
      m_isRegular(false),
      m_isInitialized(false),
      m_allowResizeFromData(false),
      m_xPtsNb(0),
      m_yPtsNb(0) {}

bool asAreaGrid::InitializeAxes(const a1d& lons, const a1d& lats, bool strideAllowed, bool getLarger) {
    m_isInitialized = true;

    if (AreaDefinedByPointsNb(lons, lats)) return HandleAreaDefinedByPointsNb(lons, lats);

    return CreateAxes(lons, lats, getLarger);
}

void asAreaGrid::CorrectCornersWithAxes() {
    if (GetXmin() != m_xAxis[0]) {
        Coo cornerLL = GetCornerLL();
        cornerLL.x = m_xAxis[0];
        SetCornerLL(cornerLL, true);
        Coo cornerUL = GetCornerUL();
        cornerUL.x = m_xAxis[0];
        SetCornerUL(cornerUL, true);
    }

    if (GetXmax() != m_xAxis.tail(1)[0]) {
        Coo cornerLR = GetCornerLR();
        cornerLR.x = m_xAxis.tail(1)[0];
        SetCornerLR(cornerLR, true);
        Coo cornerUR = GetCornerUR();
        cornerUR.x = m_xAxis.tail(1)[0];
        SetCornerUR(cornerUR, true);
    }

    if (GetCornerLL().x > GetCornerLR().x || GetCornerUL().x > GetCornerUR().x) {
        if (m_isLatLon) {
            Coo cornerLR = GetCornerLR();
            cornerLR.x += 360;
            SetCornerLR(cornerLR, true);
            Coo cornerUR = GetCornerUR();
            cornerUR.x += 360;
            SetCornerUR(cornerUR, true);
        } else {
            asThrowException(_("Inconsistent x coordinates on a non lat / lon axis."));
        }
    }

    if (GetYmin() != m_yAxis[0]) {
        Coo cornerLL = GetCornerLL();
        cornerLL.y = m_yAxis[0];
        SetCornerLL(cornerLL, true);
        Coo cornerLR = GetCornerLR();
        cornerLR.y = m_yAxis[0];
        SetCornerLR(cornerLR, true);
    }

    if (GetYmax() != m_yAxis.tail(1)[0]) {
        Coo cornerUL = GetCornerUL();
        cornerUL.y = m_yAxis.tail(1)[0];
        SetCornerUL(cornerUL, true);
        Coo cornerUR = GetCornerUR();
        cornerUR.y = m_yAxis.tail(1)[0];
        SetCornerUR(cornerUR, true);
    }
}

bool asAreaGrid::CreateAxes(const a1d& lons, const a1d& lats, bool getLarger) {
    int indexXmin, indexXmax;
    auto nlons = int(lons.size() - 1);

    if (getLarger) {  // Get larger when interpolation is needed

        indexXmin = asFindFloor(&lons[0], &lons[nlons], GetXmin(), asHIDE_WARNINGS);
        if (indexXmin == asOUT_OF_RANGE) {
            indexXmin = asFindFloor(&lons[0], &lons[nlons], GetXmin() + 360.0, asHIDE_WARNINGS);
        }
        if (indexXmin == asOUT_OF_RANGE) {
            indexXmin = asFindFloor(&lons[0], &lons[nlons], GetXmin() - 360.0, asHIDE_WARNINGS);
        }

        indexXmax = asFindCeil(&lons[0], &lons[nlons], GetXmax(), asHIDE_WARNINGS);
        if (indexXmax == asOUT_OF_RANGE) {
            indexXmax = asFindCeil(&lons[0], &lons[nlons], GetXmax() + 360.0, asHIDE_WARNINGS);
        }
        if (indexXmax == asOUT_OF_RANGE) {
            wxASSERT(lons.size() > 1);
            double dataStep = lons[1] - lons[0];
            indexXmax = asFindCeil(&lons[0], &lons[nlons], GetXmax() - dataStep, asHIDE_WARNINGS);
        }
        if (indexXmax == asOUT_OF_RANGE) {
            indexXmax = asFindCeil(&lons[0], &lons[nlons], GetXmax() - 360.0, asHIDE_WARNINGS);
        }

    } else {
        indexXmin = asFindClosest(&lons[0], &lons[nlons], GetXmin(), asHIDE_WARNINGS);
        if (indexXmin == asOUT_OF_RANGE) {
            indexXmin = asFindClosest(&lons[0], &lons[nlons], GetXmin() + 360.0, asHIDE_WARNINGS);
        }
        if (indexXmin == asOUT_OF_RANGE) {
            indexXmin = asFindClosest(&lons[0], &lons[nlons], GetXmin() - 360.0, asHIDE_WARNINGS);
        }

        indexXmax = asFindClosest(&lons[0], &lons[nlons], GetXmax(), asHIDE_WARNINGS);
        if (indexXmax == asOUT_OF_RANGE) {
            indexXmax = asFindClosest(&lons[0], &lons[nlons], GetXmax() + 360.0, asHIDE_WARNINGS);
        }
        if (indexXmax == asOUT_OF_RANGE) {
            wxASSERT(lons.size() > 1);
            double dataStep = lons[1] - lons[0];
            indexXmax = asFindClosest(&lons[0], &lons[nlons], GetXmax() - dataStep, asHIDE_WARNINGS);
        }
        if (indexXmax == asOUT_OF_RANGE) {
            indexXmax = asFindClosest(&lons[0], &lons[nlons], GetXmax() - 360.0, asHIDE_WARNINGS);
        }
    }

    if (m_allowResizeFromData) {
        if (indexXmin != asOUT_OF_RANGE && indexXmax == asOUT_OF_RANGE) {
            indexXmax = nlons;
        }
    }

    if (indexXmax == asOUT_OF_RANGE) {
        wxLogError(_("Cannot find the corresponding value (%g) on the longitude axis."), GetXmax());
        return false;
    }

    wxASSERT(indexXmin >= 0);
    wxASSERT(indexXmax >= 0);
    wxASSERT(indexXmin <= indexXmax);

    if (indexXmin > indexXmax) {
        wxLogError(_("The index (%d) of the longitude min is larger than the one for the longitude max (%d)."),
                   indexXmin, indexXmax);
        return false;
    }

    int indexYmin, indexYmax;
    int nlats = int(lats.size() - 1);

    if (getLarger) {
        indexYmin = asFindFloor(&lats[0], &lats[nlats], GetYmin(), asHIDE_WARNINGS);
        indexYmax = asFindCeil(&lats[0], &lats[nlats], GetYmax(), asHIDE_WARNINGS);

        if (m_allowResizeFromData) {
            if (indexYmin != asOUT_OF_RANGE && indexYmax == asOUT_OF_RANGE) {
                if (lats[nlats] > lats[0]) {
                    indexYmax = nlats;
                } else {
                    indexYmax = 0;
                }
            }
        }

        if (indexYmin > indexYmax) {
            indexYmin = asFindCeil(&lats[0], &lats[nlats], GetYmax(), asHIDE_WARNINGS);
            indexYmax = asFindFloor(&lats[0], &lats[nlats], GetYmin(), asHIDE_WARNINGS);

            if (m_allowResizeFromData) {
                if (indexYmin == asOUT_OF_RANGE && indexYmax != asOUT_OF_RANGE) {
                    if (lats[nlats] > lats[0]) {
                        indexYmax = nlats;
                    } else {
                        indexYmax = 0;
                    }
                }
            }
        }

    } else {
        indexYmin = asFindClosest(&lats[0], &lats[nlats], GetYmin(), asHIDE_WARNINGS);
        indexYmax = asFindClosest(&lats[0], &lats[nlats], GetYmax(), asHIDE_WARNINGS);

        if (m_allowResizeFromData) {
            if (indexYmin != asOUT_OF_RANGE && indexYmax == asOUT_OF_RANGE) {
                if (lats[nlats] > lats[0]) {
                    indexYmax = nlats;
                } else {
                    indexYmax = 0;
                }
            }
            if (indexYmin == asOUT_OF_RANGE && indexYmax != asOUT_OF_RANGE) {
                if (lats[nlats] > lats[0]) {
                    indexYmax = nlats;
                } else {
                    indexYmax = 0;
                }
            }
        }

        if (indexYmin > indexYmax) {
            int tmp = indexYmax;
            if (IsRegular()) {
                asAreaRegGrid areaReg = dynamic_cast<asAreaRegGrid&>(*this);
                if (areaReg.GetYstep() > areaReg.GetYstepData()) {
                    auto newIndex = (float)indexYmin;
                    float stepIndY = areaReg.GetYstep() / areaReg.GetYstepData();
                    indexYmax = indexYmin;
                    indexYmin = tmp;
                    while (true) {
                        newIndex -= stepIndY;
                        if (newIndex >= tmp) {
                            indexYmin = (int)newIndex;
                        } else {
                            break;
                        }
                    }
                } else {
                    indexYmax = indexYmin;
                    indexYmin = tmp;
                }
            } else {
                indexYmax = indexYmin;
                indexYmin = tmp;
            }
        }
    }

    if (indexYmin < 0 || indexYmax < 0) {
        wxLogError(_("Cannot find one the corresponding value (%g or %g) on the latitude axis."), GetYmin(), GetYmax());
        return false;
    }

    wxASSERT(indexYmin >= 0);
    wxASSERT(indexYmax >= 0);
    wxASSERT(indexYmin <= indexYmax);

    m_xAxis = lons.segment(indexXmin, indexXmax - indexXmin + 1);
    m_yAxis = lats.segment(indexYmin, indexYmax - indexYmin + 1);

    return true;
}

bool asAreaGrid::AreaDefinedByPointsNb(const a1d& lons, const a1d& lats) {
    return m_xPtsNb > 0 && m_yPtsNb > 0 && GetXmin() == GetXmax() && GetYmin() == GetYmax();
}

bool asAreaGrid::HandleAreaDefinedByPointsNb(const a1d& lons, const a1d& lats) {
    int indexYmin = asFindClosest(&lats[0], &lats[lats.size() - 1], GetYmin());
    int indexYmax = 0;
    a1d latsAxis;

    wxASSERT(indexYmin >= 0);
    wxASSERT(lats.size() > 1);

    if (lats[1] > lats[0]) {  // Increasing
        if (indexYmin + m_yPtsNb > lats.size()) {
            wxLogMessage(_("The number of points on the latitude axis was reduced to fit the data."));
            m_yPtsNb = (int)lats.size() - indexYmin;
        }
        indexYmax = indexYmin + m_yPtsNb - 1;
        latsAxis = lats.segment(indexYmin, m_yPtsNb);
    } else {  // Decreasing
        if (indexYmin - m_yPtsNb + 1 < 0) {
            wxLogMessage(_("The number of points on the latitude axis was reduced to fit the data."));
            m_yPtsNb = indexYmin + 1;
        }
        indexYmax = indexYmin - m_yPtsNb + 1;
        latsAxis = lats.segment(indexYmax, m_yPtsNb);
    }

    int indexXmin = asFindClosest(&lons[0], &lons[lons.size() - 1], GetXmin(), asHIDE_WARNINGS);
    if (indexXmin == asOUT_OF_RANGE) {
        indexXmin = asFindClosest(&lons[0], &lons[lons.size() - 1], GetXmin() + 360.0, asHIDE_WARNINGS);
    }
    if (indexXmin == asOUT_OF_RANGE) {
        indexXmin = asFindClosest(&lons[0], &lons[lons.size() - 1], GetXmin() - 360.0, asHIDE_WARNINGS);
    }

    wxASSERT(indexXmin >= 0);
    if (indexXmin < 0) {
        wxLogError(_("A negative index was found during the area generation."));
        return false;
    }

    if (indexXmin + m_xPtsNb > lons.size()) {
        wxLogMessage(_("The number of points on the longitude axis was reduced to fit the data."));
        m_xPtsNb = (int)lons.size() - indexXmin;
    }

    m_xAxis = lons.segment(indexXmin, m_xPtsNb);
    m_yAxis = latsAxis;

    m_cornerLL = {lons[indexXmin], m_cornerLR.y};
    m_cornerUL = {lons[indexXmin], lats[indexYmax]};
    m_cornerUR = {lons[indexXmin + m_xPtsNb - 1], lats[indexYmax]};
    m_cornerLR = {lons[indexXmin + m_xPtsNb - 1], m_cornerLR.y};

    return true;
}

a1d asAreaGrid::GetXaxis() {
    wxASSERT(m_isInitialized);
    return m_xAxis;
}

a1d asAreaGrid::GetYaxis() {
    wxASSERT(m_isInitialized);
    return m_yAxis;
}

int asAreaGrid::GetXaxisPtsnb() {
    wxASSERT(m_isInitialized);
    return (int)m_xAxis.size();
}

int asAreaGrid::GetYaxisPtsnb() {
    wxASSERT(m_isInitialized);
    return (int)m_yAxis.size();
}

double asAreaGrid::GetXaxisStart() const {
    wxASSERT(m_isInitialized);
    return m_xAxis[0];
}

double asAreaGrid::GetYaxisStart() const {
    wxASSERT(m_isInitialized);
    return m_yAxis[0];
}

double asAreaGrid::GetXaxisEnd() const {
    wxASSERT(m_isInitialized);
    return m_xAxis[m_xAxis.size() - 1];
}

double asAreaGrid::GetYaxisEnd() const {
    wxASSERT(m_isInitialized);
    return m_yAxis[m_yAxis.size() - 1];
}

int asAreaGrid::GetXptsNb() {
    return GetXaxisPtsnb();
}

int asAreaGrid::GetYptsNb() {
    return GetYaxisPtsnb();
}