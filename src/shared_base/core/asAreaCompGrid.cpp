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

    return GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep, flatAllowed);
}

asAreaCompGrid *asAreaCompGrid::GetInstance(const wxString &type, double xMin, int xPtsNb, double xStep, double yMin,
                                            int yPtsNb, double yStep, int flatAllowed)
{
    if (type.IsSameAs("Regular", false)) {
        if (xStep > 0 && yStep > 0) {
            double xWidth = (double) (xPtsNb - 1) * xStep;
            double yWidth = (double) (yPtsNb - 1) * yStep;
            return new asAreaCompRegGrid(xMin, xWidth, xStep, yMin, yWidth, yStep, flatAllowed);
        } else {
            return new asAreaCompRegGrid(xMin, xPtsNb, yMin, yPtsNb, flatAllowed);
        }
    } else if (type.IsEmpty() || type.IsSameAs("Generic", false)) {
        return new asAreaCompGenGrid(xMin, xPtsNb, yMin, yPtsNb, flatAllowed);
    } else {
        wxLogError(_("Given grid type: %s"), type);
        asThrowException("The given grid type doesn't correspond to any existing option.");
    }
}

asAreaCompGrid *asAreaCompGrid::GetInstance(double xMin, int xPtsNb, double xStep, double yMin, int yPtsNb,
                                            double yStep, int flatAllowed)
{
    return GetInstance("Generic", xMin, xPtsNb, xStep, yMin, yPtsNb, yStep, flatAllowed);
}

asAreaCompGrid::asAreaCompGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                               int flatAllowed)
        : asAreaComp(cornerUL, cornerUR, cornerLL, cornerLR, flatAllowed),
          m_isRegular(false),
          m_isInitialized(false),
          m_xPtsNb(0),
          m_yPtsNb(0)
{
}

asAreaCompGrid::asAreaCompGrid(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed)
        : asAreaComp(xMin, xWidth, yMin, yWidth, flatAllowed),
          m_isRegular(false),
          m_isInitialized(false),
          m_xPtsNb(0),
          m_yPtsNb(0)
{
}

asAreaCompGrid::asAreaCompGrid()
        : asAreaComp(),
          m_isRegular(false),
          m_isInitialized(false),
          m_xPtsNb(0),
          m_yPtsNb(0)
{
}

wxString asAreaCompGrid::GetGridTypeString() const
{
    if (m_isRegular) {
        return "Regular";
    }

    return "Generic";
}

bool asAreaCompGrid::InitializeAxes(const a1d &lons, const a1d &lats)
{
    int compositeNb = GetNbComposites();
    m_compositeXaxes.resize((size_t) compositeNb);
    m_compositeYaxes.resize((size_t) compositeNb);

    for (int i = 0; i < compositeNb; ++i) {

        int indexXmin = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmin(), asHIDE_WARNINGS);
        if (indexXmin == asOUT_OF_RANGE) {
            indexXmin = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmin() + 360.0, asHIDE_WARNINGS);
        }
        if (indexXmin == asOUT_OF_RANGE) {
            indexXmin = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmin() - 360.0, asHIDE_WARNINGS);
        }

        int indexXmax = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmax(), asHIDE_WARNINGS);
        if (indexXmax == asOUT_OF_RANGE) {
            indexXmax = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmax() + 360.0, asHIDE_WARNINGS);
        }
        if (indexXmax == asOUT_OF_RANGE) {
            indexXmax = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmax() - 360.0, asHIDE_WARNINGS);
        }
        if (indexXmax == asOUT_OF_RANGE) {
            wxASSERT(lons.size() > 1);
            double dataStep = lons[1] - lons[0];
            indexXmax = asFindClosest(&lons[0], &lons[lons.size()-1], m_composites[i].GetXmax() - dataStep, asHIDE_WARNINGS);
            if (indexXmax == asOUT_OF_RANGE) {
                std::cout << lons;
                wxLogError(_("Cannot find the corresponding value (%d) on the longitude axis."), m_composites[i].GetXmax());
                return false;
            }
        }

        wxASSERT(indexXmin >= 0);
        wxASSERT(indexXmax >= 0);
        wxASSERT(indexXmin <= indexXmax);

        int indexYmin = asFindClosest(&lats[0], &lats[lats.size()-1], m_composites[i].GetYmin());
        int indexYmax = asFindClosest(&lats[0], &lats[lats.size()-1], m_composites[i].GetYmax());
        if (indexYmin > indexYmax) {
            int tmp = indexYmax;
            indexYmax = indexYmin;
            indexYmin = tmp;
        }
        wxASSERT(indexYmin >= 0);
        wxASSERT(indexYmax >= 0);
        wxASSERT(indexYmin <= indexYmax);

        m_compositeXaxes[i] = lons.segment(indexXmin, indexXmax - indexXmin + 1);
        m_compositeYaxes[i] = lats.segment(indexYmin, indexYmax - indexYmin + 1);
    }

    m_isInitialized = true;

    return true;
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
