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

#include "asAreaCompGrid.h"
#include "asAreaCompRegGrid.h"
#include "asAreaCompGenGrid.h"
#include "asParameters.h"
#include "asPredictor.h"
#include "asTypeDefs.h"
#include <iostream>


asAreaCompGrid * asAreaCompGrid::GetInstance(const asParameters *params, int iStep, int iPtor)
{
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

asAreaCompGrid *asAreaCompGrid::GetInstance(const wxString &type, double xMin, int xPtsNb, double xStep, double yMin,
                                            int yPtsNb, double yStep, int flatAllowed, bool isLatLon)
{
    if (type.IsSameAs("Regular", false)) {
        if (xStep > 0 && yStep > 0) {
            double xWidth = (double) (xPtsNb - 1) * xStep;
            double yWidth = (double) (yPtsNb - 1) * yStep;
            return new asAreaCompRegGrid(xMin, xWidth, xStep, yMin, yWidth, yStep, flatAllowed, isLatLon);
        } else {
            return new asAreaCompRegGrid(xMin, xPtsNb, yMin, yPtsNb, flatAllowed, isLatLon);
        }
    } else if (type.IsEmpty() || type.IsSameAs("Generic", false)) {
        return new asAreaCompGenGrid(xMin, xPtsNb, yMin, yPtsNb, asFLAT_ALLOWED, isLatLon);
    } else {
        wxLogError(_("Given grid type: %s"), type);
        asThrowException("The given grid type doesn't correspond to any existing option.");
    }
}

asAreaCompGrid *asAreaCompGrid::GetInstance(double xMin, int xPtsNb, double xStep, double yMin, int yPtsNb,
                                            double yStep, int flatAllowed, bool isLatLon)
{
    return GetInstance("Generic", xMin, xPtsNb, xStep, yMin, yPtsNb, yStep, flatAllowed, isLatLon);
}

asAreaCompGrid::asAreaCompGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                               int flatAllowed, bool isLatLon)
        : asAreaComp(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed, isLatLon),
          m_isRegular(false),
          m_isInitialized(false),
          m_allowResizeFromData(false),
          m_xPtsNb(0),
          m_yPtsNb(0)
{
}

asAreaCompGrid::asAreaCompGrid(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed, bool isLatLon)
        : asAreaComp(xMin, xWidth, yMin, yWidth, flatAllowed, isLatLon),
          m_isRegular(false),
          m_isInitialized(false),
          m_allowResizeFromData(false),
          m_xPtsNb(0),
          m_yPtsNb(0)
{
}

asAreaCompGrid::asAreaCompGrid()
        : asAreaComp(),
          m_isRegular(false),
          m_isInitialized(false),
          m_allowResizeFromData(false),
          m_xPtsNb(0),
          m_yPtsNb(0)
{
}

bool asAreaCompGrid::InitializeAxes(const a1d &lons, const a1d &lats, bool strideAllowed, bool getLarger)
{
    m_isInitialized = true;

    if (AreaDefinedByPointsNb(lons, lats))
        return HandleAreaDefinedByPointsNb(lons, lats);

    if (m_isLatLon) {
        HandleNegativeLongitudes(lons);
        HandleLongitudesAbove360(lons);
        HandleLongitudesSplitAt180(lons);
        HandleMissing360(lons);
        HandleMissing180(lons);
    }

    return CreateCompositeAxes(lons, lats, getLarger);
}

void asAreaCompGrid::CorrectCornersWithAxes()
{
    if (m_composites[0].GetXmin() != m_compositeXaxes[0][0]) {
        Coo cornerLL = m_composites[0].GetCornerLL();
        cornerLL.x = m_compositeXaxes[0][0];
        m_composites[0].SetCornerLL(cornerLL, true);
        Coo cornerUL = m_composites[0].GetCornerUL();
        cornerUL.x = m_compositeXaxes[0][0];
        m_composites[0].SetCornerUL(cornerUL, true);
    }

    if (m_composites[0].GetXmax() != m_compositeXaxes[0].tail(1)[0]) {
        Coo cornerLR = m_composites[0].GetCornerLR();
        cornerLR.x = m_compositeXaxes[0].tail(1)[0];
        m_composites[0].SetCornerLR(cornerLR, true);
        Coo cornerUR = m_composites[0].GetCornerUR();
        cornerUR.x = m_compositeXaxes[0].tail(1)[0];
        m_composites[0].SetCornerUR(cornerUR, true);
    }

    if (m_composites[0].GetCornerLL().x > m_composites[0].GetCornerLR().x ||
        m_composites[0].GetCornerUL().x > m_composites[0].GetCornerUR().x) {
        if (m_isLatLon) {
            Coo cornerLR = m_composites[0].GetCornerLR();
            cornerLR.x += 360;
            m_composites[0].SetCornerLR(cornerLR, true);
            Coo cornerUR = m_composites[0].GetCornerUR();
            cornerUR.x += 360;
            m_composites[0].SetCornerUR(cornerUR, true);
        } else {
            asThrowException(_("Inconsistent x coordinates on a non lat / lon axis."));
        }
    }

    if (m_composites[0].GetYmin() != m_compositeYaxes[0][0]) {
        Coo cornerLL = m_composites[0].GetCornerLL();
        cornerLL.y = m_compositeYaxes[0][0];
        m_composites[0].SetCornerLL(cornerLL, true);
        Coo cornerLR = m_composites[0].GetCornerLR();
        cornerLR.y = m_compositeYaxes[0][0];
        m_composites[0].SetCornerLR(cornerLR, true);
    }

    if (m_composites[0].GetYmax() != m_compositeYaxes[0].tail(1)[0]) {
        Coo cornerUL = m_composites[0].GetCornerUL();
        cornerUL.y = m_compositeYaxes[0].tail(1)[0];
        m_composites[0].SetCornerUL(cornerUL, true);
        Coo cornerUR = m_composites[0].GetCornerUR();
        cornerUR.y = m_compositeYaxes[0].tail(1)[0];
        m_composites[0].SetCornerUR(cornerUR, true);
    }

    if (GetNbComposites() > 1) {
        if (m_composites[1].GetXmin() != m_compositeXaxes[1][0]) {
            Coo cornerLL = m_composites[1].GetCornerLL();
            cornerLL.x = m_compositeXaxes[1][0];
            m_composites[1].SetCornerLL(cornerLL, true);
            Coo cornerUL = m_composites[1].GetCornerUL();
            cornerUL.x = m_compositeXaxes[1][0];
            m_composites[1].SetCornerUL(cornerUL, true);
        }

        if (m_composites[1].GetXmax() != m_compositeXaxes[1].tail(1)[0]) {
            Coo cornerLR = m_composites[1].GetCornerLR();
            cornerLR.x = m_compositeXaxes[1].tail(1)[0];
            m_composites[1].SetCornerLR(cornerLR, true);
            Coo cornerUR = m_composites[1].GetCornerUR();
            cornerUR.x = m_compositeXaxes[1].tail(1)[0];
            m_composites[1].SetCornerUR(cornerUR, true);
        }

        if (m_composites[1].GetYmin() != m_compositeYaxes[1][0]) {
            Coo cornerLL = m_composites[1].GetCornerLL();
            cornerLL.y = m_compositeYaxes[1][0];
            m_composites[1].SetCornerLL(cornerLL, true);
            Coo cornerLR = m_composites[1].GetCornerLR();
            cornerLR.y = m_compositeYaxes[1][0];
            m_composites[1].SetCornerLR(cornerLR, true);
        }

        if (m_composites[1].GetYmax() != m_compositeYaxes[1].tail(1)[0]) {
            Coo cornerUL = m_composites[1].GetCornerUL();
            cornerUL.y = m_compositeYaxes[1].tail(1)[0];
            m_composites[1].SetCornerUL(cornerUL, true);
            Coo cornerUR = m_composites[1].GetCornerUR();
            cornerUR.y = m_compositeYaxes[1].tail(1)[0];
            m_composites[1].SetCornerUR(cornerUR, true);
        }
    }
}

bool asAreaCompGrid::CreateCompositeAxes(const a1d &lons, const a1d &lats, bool getLarger)
{
    m_compositeXaxes.resize((size_t) GetNbComposites());
    m_compositeYaxes.resize((size_t) GetNbComposites());

    for (int i = 0; i < GetNbComposites(); ++i) {

        int indexXmin, indexXmax;
        auto nlons = static_cast<int>(lons.size() - 1);

        if (getLarger) { // Get larger when interpolation is needed interpolation

            indexXmin = asFindFloor(&lons[0], &lons[nlons], m_composites[i].GetXmin(), asHIDE_WARNINGS);
            if (indexXmin == asOUT_OF_RANGE) {
                indexXmin = asFindFloor(&lons[0], &lons[nlons], m_composites[i].GetXmin() + 360.0, asHIDE_WARNINGS);
            }
            if (indexXmin == asOUT_OF_RANGE) {
                indexXmin = asFindFloor(&lons[0], &lons[nlons], m_composites[i].GetXmin() - 360.0, asHIDE_WARNINGS);
            }

            indexXmax = asFindCeil(&lons[0], &lons[nlons], m_composites[i].GetXmax(), asHIDE_WARNINGS);
            if (indexXmax == asOUT_OF_RANGE) {
                indexXmax = asFindCeil(&lons[0], &lons[nlons], m_composites[i].GetXmax() + 360.0, asHIDE_WARNINGS);
            }
            if (indexXmax == asOUT_OF_RANGE) {
                wxASSERT(lons.size() > 1);
                double dataStep = lons[1] - lons[0];
                indexXmax = asFindCeil(&lons[0], &lons[nlons], m_composites[i].GetXmax() - dataStep, asHIDE_WARNINGS);
            }
            if (indexXmax == asOUT_OF_RANGE) {
                indexXmax = asFindCeil(&lons[0], &lons[nlons], m_composites[i].GetXmax() - 360.0, asHIDE_WARNINGS);
            }

        } else {

            indexXmin = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmin(), asHIDE_WARNINGS);
            if (indexXmin == asOUT_OF_RANGE) {
                indexXmin = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmin() + 360.0, asHIDE_WARNINGS);
            }
            if (indexXmin == asOUT_OF_RANGE) {
                indexXmin = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmin() - 360.0, asHIDE_WARNINGS);
            }

            indexXmax = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmax(), asHIDE_WARNINGS);
            if (indexXmax == asOUT_OF_RANGE) {
                indexXmax = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmax() + 360.0, asHIDE_WARNINGS);
            }
            if (indexXmax == asOUT_OF_RANGE) {
                wxASSERT(lons.size() > 1);
                double dataStep = lons[1] - lons[0];
                indexXmax = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmax() - dataStep, asHIDE_WARNINGS);
            }
            if (indexXmax == asOUT_OF_RANGE) {
                indexXmax = asFindClosest(&lons[0], &lons[nlons], m_composites[i].GetXmax() - 360.0, asHIDE_WARNINGS);
            }
        }

        if (m_allowResizeFromData) {
            if (indexXmin != asOUT_OF_RANGE && indexXmax == asOUT_OF_RANGE) {
                indexXmax = nlons;
            }
        }

        if (indexXmax == asOUT_OF_RANGE) {
            std::cout << lons;
            wxLogError(_("Cannot find the corresponding value (%g) on the longitude axis."), m_composites[i].GetXmax());
            return false;
        }

        wxASSERT(indexXmin >= 0);
        wxASSERT(indexXmax >= 0);
        wxASSERT(indexXmin <= indexXmax);

        int indexYmin, indexYmax;
        int nlats = static_cast<int>(lats.size() - 1);

        if (getLarger) {

            indexYmin = asFindFloor(&lats[0], &lats[nlats], m_composites[i].GetYmin(), asHIDE_WARNINGS);
            indexYmax = asFindCeil(&lats[0], &lats[nlats], m_composites[i].GetYmax(), asHIDE_WARNINGS);

            if (m_allowResizeFromData) {
                if (indexYmin != asOUT_OF_RANGE && indexYmax == asOUT_OF_RANGE) {
                    indexYmax = nlats;
                }
            }

            if (indexYmin > indexYmax) {
                indexYmin = asFindCeil(&lats[0], &lats[nlats], m_composites[i].GetYmax(), asHIDE_WARNINGS);
                indexYmax = asFindFloor(&lats[0], &lats[nlats], m_composites[i].GetYmin(), asHIDE_WARNINGS);

                if (m_allowResizeFromData) {
                    if (indexYmin == asOUT_OF_RANGE && indexYmax != asOUT_OF_RANGE) {
                        indexYmin = nlats;
                    }
                }
            }

        } else {

            indexYmin = asFindClosest(&lats[0], &lats[nlats], m_composites[i].GetYmin(), asHIDE_WARNINGS);
            indexYmax = asFindClosest(&lats[0], &lats[nlats], m_composites[i].GetYmax(), asHIDE_WARNINGS);

            if (m_allowResizeFromData) {
                if (indexYmin != asOUT_OF_RANGE && indexYmax == asOUT_OF_RANGE) {
                    indexYmax = nlats;
                }
                if (indexYmin == asOUT_OF_RANGE && indexYmax != asOUT_OF_RANGE) {
                    indexYmin = nlats;
                }
            }

            if (indexYmin > indexYmax) {
                int tmp = indexYmax;
                indexYmax = indexYmin;
                indexYmin = tmp;
            }
        }

        wxASSERT(indexYmin >= 0);
        wxASSERT(indexYmax >= 0);
        wxASSERT(indexYmin <= indexYmax);

        m_compositeXaxes[i] = lons.segment(indexXmin, indexXmax - indexXmin + 1);
        m_compositeYaxes[i] = lats.segment(indexYmin, indexYmax - indexYmin + 1);
    }

    return true;
}

bool asAreaCompGrid::AreaDefinedByPointsNb(const a1d &lons, const a1d &lats)
{
    return m_xPtsNb > 0 && m_yPtsNb > 0 && GetNbComposites() == 1 &&
           m_composites[0].GetXmin() == m_composites[0].GetXmax() &&
           m_composites[0].GetYmin() == m_composites[0].GetYmax();
}

bool asAreaCompGrid::HandleAreaDefinedByPointsNb(const a1d &lons, const a1d &lats)
{
    int indexYmin = asFindClosest(&lats[0], &lats[lats.size()-1], m_composites[0].GetYmin());
    int indexYmax = 0;
    a1d latsAxis;

    wxASSERT(indexYmin >= 0);
    wxASSERT(lats.size() > 1);

    if (lats[1] > lats[0]) { // Increasing
        if (indexYmin + m_yPtsNb > lats.size()) {
            wxLogMessage(_("The number of points on the latitude axis was reduced to fit the data."));
            m_yPtsNb = (int)lats.size() - indexYmin;
        }
        indexYmax = indexYmin + m_yPtsNb - 1;
        latsAxis = lats.segment(indexYmin, m_yPtsNb);
    } else { // Decreasing
        if (indexYmin - m_yPtsNb + 1 < 0) {
            wxLogMessage(_("The number of points on the latitude axis was reduced to fit the data."));
            m_yPtsNb = indexYmin + 1;
        }
        indexYmax = indexYmin - m_yPtsNb + 1;
        latsAxis = lats.segment(indexYmax, m_yPtsNb);
    }

    int indexXmin = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[0].GetXmin(), asHIDE_WARNINGS);
    if (indexXmin == asOUT_OF_RANGE) {
        indexXmin = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[0].GetXmin() + 360.0, asHIDE_WARNINGS);
    }
    if (indexXmin == asOUT_OF_RANGE) {
        indexXmin = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[0].GetXmin() - 360.0, asHIDE_WARNINGS);
    }
    wxASSERT(indexXmin >= 0);

    if (indexXmin + m_xPtsNb > lons.size()) {
        // Handle the case where the longitude axis covers the whole globe
        wxASSERT(lons.size() > 1);
        if (std::abs(lons[0] + 360 - lons[lons.size() - 1]) < 2 * lons[1] - lons[0]) {
            m_compositeXaxes.resize(2);
            m_compositeYaxes.resize(2);

            m_compositeXaxes[0] = lons.segment(indexXmin, lons.size() - indexXmin);
            m_compositeYaxes[0] = latsAxis;
            m_compositeXaxes[1] = lons.segment(0, m_xPtsNb - m_compositeXaxes[1].size());
            m_compositeYaxes[1] = latsAxis;

            m_cornerUL = {m_cornerUL.x, lats[indexYmax]};
            m_cornerUR = {m_compositeXaxes[1][m_compositeXaxes[1].size() - 1], lats[indexYmax]};
            m_cornerLR = {m_compositeXaxes[1][m_compositeXaxes[1].size() - 1], m_cornerLR.y};

            m_composites[0].SetCornerUL(m_cornerUL, true);
            m_composites[0].SetCornerUR({m_compositeXaxes[0][m_compositeXaxes[0].size() - 1], m_cornerUR.y}, true);
            m_composites[0].SetCornerLR({m_compositeXaxes[0][m_compositeXaxes[0].size() - 1], m_cornerLR.y}, true);

            asArea area2(m_composites[0]);
            area2.SetCornerLL({m_compositeXaxes[1][0], area2.GetCornerLL().y}, true);
            area2.SetCornerUL({m_compositeXaxes[1][0], area2.GetCornerUL().y}, true);
            area2.SetCornerLR({m_compositeXaxes[1][m_compositeXaxes[1].size() - 1], area2.GetCornerLR().y}, true);
            area2.SetCornerUR({m_compositeXaxes[1][m_compositeXaxes[1].size() - 1], area2.GetCornerUR().y}, true);
            m_composites.push_back(area2);

            return true;
        } else {
            wxLogMessage(_("The number of points on the longitude axis was reduced to fit the data."));
            m_xPtsNb = (int)lons.size() - indexXmin;
        }
    }

    m_compositeXaxes.resize(1);
    m_compositeYaxes.resize(1);
    m_compositeXaxes[0] = lons.segment(indexXmin, m_xPtsNb);
    m_compositeYaxes[0] = latsAxis;

    m_cornerLL = {lons[indexXmin], m_cornerLR.y};
    m_cornerUL = {lons[indexXmin], lats[indexYmax]};
    m_cornerUR = {lons[indexXmin + m_xPtsNb - 1], lats[indexYmax]};
    m_cornerLR = {lons[indexXmin + m_xPtsNb - 1], m_cornerLR.y};

    m_composites[0].SetCornerLL(m_cornerLL, true);
    m_composites[0].SetCornerUL(m_cornerUL, true);
    m_composites[0].SetCornerUR(m_cornerUR, true);
    m_composites[0].SetCornerLR(m_cornerLR, true);

    return true;
}

void asAreaCompGrid::HandleLongitudesAbove360(const a1d &lons)
{
    // If longitudes axis has values above 360, allow a single area instead of composites.
    if (GetNbComposites() == 2 && lons[lons.size() - 1] > 360) {
        m_cornerUR.x += 360;
        m_cornerLR.x += 360;

        m_composites[0].SetCornerUR({m_composites[1].GetCornerUR().x + 360, m_composites[1].GetCornerUR().y}, true);
        m_composites[0].SetCornerLR({m_composites[1].GetCornerLR().x + 360, m_composites[1].GetCornerLR().y}, true);

        m_composites.pop_back();
    }
}

void asAreaCompGrid::HandleLongitudesSplitAt180(const a1d &lons)
{
    // If longitudes axis is between -180° and +180°, adapt areas if on the edge.
    if (GetNbComposites() == 1 && lons[0] == -180 && m_cornerUR.x > 180) {
        if (m_cornerUL.x < 180) {

            // 2 areas needed
            m_cornerUR.x -= 360;
            m_cornerLR.x -= 360;

            asArea area2(m_composites[0]);
            area2.SetCornerLL({-180, area2.GetCornerLL().y}, true);
            area2.SetCornerUL({-180, area2.GetCornerUL().y}, true);
            area2.SetCornerUR({area2.GetCornerUR().x - 360, area2.GetCornerUR().y}, true);
            area2.SetCornerLR({area2.GetCornerLR().x - 360, area2.GetCornerLR().y}, true);
            m_composites.push_back(area2);

            m_composites[0].SetCornerUR({180, m_composites[0].GetCornerUR().y}, true);
            m_composites[0].SetCornerLR({180, m_composites[0].GetCornerLR().y}, true);

        } else {

            // 1 area fits
            m_cornerUR.x -= 360;
            m_cornerLR.x -= 360;
            m_cornerUL.x -= 360;
            m_cornerLL.x -= 360;

            m_composites[0].SetCornerUR(m_cornerUR, true);
            m_composites[0].SetCornerLR(m_cornerLR, true);
            m_composites[0].SetCornerUL(m_cornerUL, true);
            m_composites[0].SetCornerLL(m_cornerLL, true);

        }
    }
}

void asAreaCompGrid::HandleNegativeLongitudes(const a1d &lons)
{
    // If longitudes axis has negative values, allow a single area instead of composites.
    if (GetNbComposites() == 2 && lons[0] <= m_composites[0].GetXmin() - 360) {
        m_cornerUL.x -= 360;
        m_cornerLL.x -= 360;

        m_composites[1].SetCornerUL({m_composites[0].GetCornerUL().x - 360, m_composites[0].GetCornerUL().y}, true);
        m_composites[1].SetCornerLL({m_composites[0].GetCornerLL().x - 360, m_composites[0].GetCornerLL().y}, true);

        m_composites.erase(m_composites.begin());
    }
}

void asAreaCompGrid::HandleMissing360(const a1d &lons)
{
    // The last longitudes column might not be present in the file (360° case)
    if (GetNbComposites() == 1 && m_composites[0].GetXmax() == 360 && lons[lons.size() - 1] < 360 && lons[0] >= 0) {
        wxASSERT(lons.size() > 1);
        double dataStep = lons[1] - lons[0];
        if (std::abs(360 - lons[lons.size() - 1] - dataStep) > 0.01) {
            asThrowException("Cannot find the desired value on the longitude axis.");
        }

        asArea area2(m_composites[0]);
        area2.SetCornerLL({360, area2.GetCornerLL().y}, true);
        area2.SetCornerUL({360, area2.GetCornerUL().y}, true);
        m_composites.push_back(area2);

        m_composites[0].SetCornerUR({m_composites[0].GetCornerUR().x - dataStep, m_composites[0].GetCornerUR().y}, true);
        m_composites[0].SetCornerLR({m_composites[0].GetCornerLR().x - dataStep, m_composites[0].GetCornerLR().y}, true);
    }
}

void asAreaCompGrid::HandleMissing180(const a1d &lons)
{
    // The last longitudes column might not be present in the file (360° case)
    if (GetNbComposites() == 1 && m_composites[0].GetXmax() == 180 && lons[lons.size() - 1] < 180 && lons[0] >= -180) {
        wxASSERT(lons.size() > 1);
        double dataStep = lons[1] - lons[0];
        if (std::abs(180 - lons[lons.size() - 1] - dataStep) > 0.01) {
            asThrowException("Cannot find the desired value on the longitude axis.");
        }

        asArea area2(m_composites[0]);
        area2.SetCornerLL({180, area2.GetCornerLL().y}, true);
        area2.SetCornerUL({180, area2.GetCornerUL().y}, true);
        m_composites.push_back(area2);

        m_composites[0].SetCornerUR({m_composites[0].GetCornerUR().x - dataStep, m_composites[0].GetCornerUR().y}, true);
        m_composites[0].SetCornerLR({m_composites[0].GetCornerLR().x - dataStep, m_composites[0].GetCornerLR().y}, true);
    }
}

a1d asAreaCompGrid::GetXaxis()
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeXaxes.empty());

    if (GetNbComposites() == 1) {
        return m_compositeXaxes[0];
    }

    a1d newAxis(GetXaxisCompositePtsnb(0) + GetXaxisCompositePtsnb(1));
    newAxis.head(GetXaxisCompositePtsnb(0)) = m_compositeXaxes[0];
    a1d rightPart = m_compositeXaxes[1];
    if (m_compositeXaxes[0][m_compositeXaxes[0].size() - 1] > m_compositeXaxes[1][0]) {
        rightPart = rightPart + 360;
    }
    newAxis.tail(GetXaxisCompositePtsnb(1)) = rightPart;

    return newAxis;
}

a1d asAreaCompGrid::GetYaxis()
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeYaxes.empty());

    return m_compositeYaxes[0];
}

a1d asAreaCompGrid::GetXaxisComposite(int compositeNb)
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeXaxes.empty());

    return m_compositeXaxes[compositeNb];
}

a1d asAreaCompGrid::GetYaxisComposite(int compositeNb)
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeYaxes.empty());

    return m_compositeYaxes[compositeNb];
}

int asAreaCompGrid::GetXaxisCompositePtsnb(int compositeNb)
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeXaxes.empty());

    return (int)m_compositeXaxes[compositeNb].size();
}

int asAreaCompGrid::GetYaxisCompositePtsnb(int compositeNb)
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeYaxes.empty());

    return (int) m_compositeYaxes[compositeNb].size();
}


double asAreaCompGrid::GetXaxisCompositeStart(int compositeNb) const
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeXaxes.empty());

    return m_compositeXaxes[compositeNb][0];
}

double asAreaCompGrid::GetYaxisCompositeStart(int compositeNb) const
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeYaxes.empty());

    return m_compositeYaxes[compositeNb][0];
}

double asAreaCompGrid::GetXaxisCompositeEnd(int compositeNb) const
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeXaxes.empty());

    return m_compositeXaxes[compositeNb][m_compositeXaxes[compositeNb].size() - 1];
}

double asAreaCompGrid::GetYaxisCompositeEnd(int compositeNb) const
{
    wxASSERT(m_isInitialized);
    wxASSERT(!m_compositeYaxes.empty());

    return m_compositeYaxes[compositeNb][m_compositeYaxes[compositeNb].size() - 1];
}

int asAreaCompGrid::GetXptsNb()
{
    if (GetNbComposites() == 1) {
        return GetXaxisCompositePtsnb(0);

    } else {
        return GetXaxisCompositePtsnb(0) + GetXaxisCompositePtsnb(1);
    }
}

int asAreaCompGrid::GetYptsNb()
{
    return GetYaxisCompositePtsnb(0);
}

double asAreaCompGrid::GetXmin() const
{
    return m_composites[0].GetXmin();
}

double asAreaCompGrid::GetXmax() const
{
    if (GetNbComposites() == 1) {
        return m_composites[0].GetXmax();
    }

    return m_composites[1].GetXmax();
}

double asAreaCompGrid::GetYmin() const
{
    return m_composites[0].GetYmin();
}

double asAreaCompGrid::GetYmax() const
{
    return m_composites[0].GetYmax();
}