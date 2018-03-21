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

#include "asGeoAreaCompositeGaussianGrid.h"

asGeoAreaCompositeGaussianGrid::asGeoAreaCompositeGaussianGrid(const Coo &cornerUL, const Coo &cornerUR,
                                                               const Coo &cornerLL, const Coo &cornerLR,
                                                               asGeo::GridType type, float level, float height,
                                                               int flatAllowed)
        : asGeoAreaCompositeGrid(cornerUL, cornerUR, cornerLL, cornerLR, level, height, flatAllowed)
{
    m_gridType = type;

    asGeoAreaGaussianGrid::BuildLonAxis(m_fullAxisX, type);
    asGeoAreaGaussianGrid::BuildLatAxis(m_fullAxisY, type);

    if (!IsOnGrid(cornerUL))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(cornerUR))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(cornerLL))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(cornerLR))
        asThrowException(_("The given area does not match a gaussian grid."));
}

asGeoAreaCompositeGaussianGrid::asGeoAreaCompositeGaussianGrid(double xMin, int xPtsNb, double yMin, int yPtsNb,
                                                               asGeo::GridType type, float level, float height,
                                                               int flatAllowed)
        : asGeoAreaCompositeGrid(level, height)
{
    m_gridType = type;

    asGeoAreaGaussianGrid::BuildLonAxis(m_fullAxisX, type);
    asGeoAreaGaussianGrid::BuildLatAxis(m_fullAxisY, type);

    // Check input
    if (!IsOnGrid(xMin, yMin))
        asThrowException(wxString::Format(_("The given area does not match a gaussian grid (xMin = %g, yMin = %g)."),
                                          xMin, yMin));

    // Get real size to generate parent member variables
    int indexXmin = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], xMin, 0.01);
    int indexYmin = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], yMin, 0.01);
    wxASSERT(indexXmin >= 0);
    wxASSERT(indexYmin >= 0);
    wxASSERT(m_fullAxisX.size() > indexXmin + xPtsNb - 1);
    wxASSERT(m_fullAxisY.size() > indexYmin + yPtsNb - 1);
    wxASSERT(xPtsNb >= 0);
    wxASSERT(yPtsNb >= 0);
    if (m_fullAxisX.size() <= indexXmin + xPtsNb - 1)
        asThrowException(_("The given width exceeds the grid size of the Gaussian grid."));
    if (m_fullAxisY.size() <= indexYmin + yPtsNb - 1)
        asThrowException(_("The given height exceeds the grid size of the Gaussian grid."));
    if (xPtsNb < 0)
        asThrowException(wxString::Format(_("The given width (points number) is not consistent in the Gaussian grid: %d"),
                                          xPtsNb));
    if (yPtsNb < 0)
        asThrowException(wxString::Format(_("The given height (points number) is not consistent in the Gaussian grid: %d"),
                                          yPtsNb));
    if (indexXmin < 0 || indexYmin < 0)
        asThrowException(_("Negative indices found when building the Gaussian grid."));
    double xWidth = m_fullAxisX[indexXmin + xPtsNb - 1] - m_fullAxisX[indexXmin];
    double yWidth = m_fullAxisY[indexYmin + yPtsNb - 1] - m_fullAxisY[indexYmin];

    // Regenerate with correct sizes
    Generate(xMin, xWidth, yMin, yWidth, flatAllowed);
}

bool asGeoAreaCompositeGaussianGrid::GridsOverlay(asGeoAreaCompositeGrid *otherarea) const
{
    if (otherarea->GetGridType() != GetGridType())
        return false;
    auto *otherareaGaussian(dynamic_cast<asGeoAreaCompositeGaussianGrid *>(otherarea));
    if (!otherareaGaussian)
        return false;

    return otherareaGaussian->GetGridType() == GetGridType();
}

a1d asGeoAreaCompositeGaussianGrid::GetXaxisComposite(int compositeNb)
{
    double xMin = GetComposite(compositeNb).GetXmin();
    double xMax = GetComposite(compositeNb).GetXmax();

    int xMinIndex = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], xMin, 0.01);
    int xMaxIndex = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], xMax, 0.01);

    wxASSERT(xMinIndex >= 0);
    wxASSERT(xMaxIndex >= 0);
    wxASSERT(xMaxIndex >= xMinIndex);

    return m_fullAxisX.segment(xMinIndex, xMaxIndex - xMinIndex + 1);
}

a1d asGeoAreaCompositeGaussianGrid::GetYaxisComposite(int compositeNb)
{
    double yMin = GetComposite(compositeNb).GetYmin();
    double yMax = GetComposite(compositeNb).GetYmax();

    int yMinIndex = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], yMin, 0.01);
    int yMaxIndex = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], yMax, 0.01);

    wxASSERT(yMinIndex >= 0);
    wxASSERT(yMaxIndex >= 0);
    wxASSERT(yMaxIndex >= yMinIndex);

    return m_fullAxisY.segment(yMinIndex, yMaxIndex - yMinIndex + 1);
}

int asGeoAreaCompositeGaussianGrid::GetXaxisCompositePtsnb(int compositeNb)
{
    double xMin = GetComposite(compositeNb).GetXmin();
    double xMax = GetComposite(compositeNb).GetXmax();

    int xMinIndex = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], xMin, 0.01);
    int xMaxIndex = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], xMax, 0.01);

    wxASSERT(xMinIndex >= 0);
    wxASSERT(xMaxIndex >= 0);

    int ptsnb = xMaxIndex - xMinIndex;

    if (compositeNb == 0) // from 0
    {
        ptsnb += 1;
    } else if (compositeNb == 1) // to 360
    {
        ptsnb += 1;
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }

    return ptsnb;
}

int asGeoAreaCompositeGaussianGrid::GetYaxisCompositePtsnb(int compositeNb)
{
    double yMin = GetComposite(compositeNb).GetYmin();
    double yMax = GetComposite(compositeNb).GetYmax();

    int yMinIndex = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], yMin, 0.01);
    int yMaxIndex = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], yMax, 0.01);

    wxASSERT(yMinIndex >= 0);
    wxASSERT(yMaxIndex >= 0);

    int ptsnb = yMaxIndex - yMinIndex;
    ptsnb += 1;

    return ptsnb;
}

double asGeoAreaCompositeGaussianGrid::GetXaxisCompositeWidth(int compositeNb) const
{
    return std::abs(GetComposite(compositeNb).GetXmax() - GetComposite(compositeNb).GetXmin());
}

double asGeoAreaCompositeGaussianGrid::GetYaxisCompositeWidth(int compositeNb) const
{
    return std::abs(GetComposite(compositeNb).GetYmax() - GetComposite(compositeNb).GetYmin());
}

double asGeoAreaCompositeGaussianGrid::GetXaxisCompositeStart(int compositeNb) const
{
    // If only one composite
    if (GetNbComposites() == 1) {
        return GetComposite(compositeNb).GetXmin();
    }

    // If multiple composites
    if (compositeNb == 0) // from 0
    {
        return GetComposite(compositeNb).GetXmin();
    } else if (compositeNb == 1) // to 360
    {
        return GetComposite(compositeNb).GetXmin();
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeGaussianGrid::GetYaxisCompositeStart(int compositeNb) const
{
    // If only one composite
    if (GetNbComposites() == 1) {
        return GetComposite(compositeNb).GetYmin();
    }

    // If multiple composites
    if (compositeNb == 0) // from 0
    {
        return GetComposite(compositeNb).GetYmin();
    } else if (compositeNb == 1) // to 360
    {
        return GetComposite(compositeNb).GetYmin();
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeGaussianGrid::GetXaxisCompositeEnd(int compositeNb) const
{
    // If only one composite
    if (GetNbComposites() == 1) {
        return GetComposite(compositeNb).GetXmax();
    }

    // If multiple composites
    if (compositeNb == 1) // to 360
    {
        return GetComposite(compositeNb).GetXmax();
    } else if (compositeNb == 0) // from 0
    {
        return GetComposite(compositeNb).GetXmax();
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asGeoAreaCompositeGaussianGrid::GetYaxisCompositeEnd(int compositeNb) const
{
    // If only one composite
    if (GetNbComposites() == 1) {
        return GetComposite(compositeNb).GetYmax();
    }

    // If multiple composites
    if (compositeNb == 1) // to 360
    {
        return GetComposite(compositeNb).GetYmax();
    } else if (compositeNb == 0) // from 0
    {
        return GetComposite(compositeNb).GetYmax();
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

bool asGeoAreaCompositeGaussianGrid::IsOnGrid(const Coo &point) const
{
    if (!IsRectangle())
        return false;

    int foundX = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], point.x, 0.01);
    if ((foundX == asNOT_FOUND) || (foundX == asOUT_OF_RANGE))
        return false;

    int foundY = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], point.y, 0.01);
    if ((foundY == asNOT_FOUND) || (foundY == asOUT_OF_RANGE))
        return false;

    return true;
}

bool asGeoAreaCompositeGaussianGrid::IsOnGrid(double xCoord, double yCoord) const
{
    int foundX = asFind(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], xCoord, 0.01);
    if ((foundX == asNOT_FOUND) || (foundX == asOUT_OF_RANGE))
        return false;

    int foundY = asFind(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], yCoord, 0.01);
    if ((foundY == asNOT_FOUND) || (foundY == asOUT_OF_RANGE))
        return false;

    return true;
}
