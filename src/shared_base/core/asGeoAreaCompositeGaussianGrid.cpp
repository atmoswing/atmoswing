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

asGeoAreaCompositeGaussianGrid::asGeoAreaCompositeGaussianGrid(const Coo &CornerUL, const Coo &CornerUR,
                                                               const Coo &CornerLL, const Coo &CornerLR,
                                                               asGeo::GridType type, float Level, float Height,
                                                               int flatAllowed)
        : asGeoAreaCompositeGrid(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    m_gridType = type;

    asGeoAreaGaussianGrid::BuildLonAxis(m_fullAxisX, type);
    asGeoAreaGaussianGrid::BuildLatAxis(m_fullAxisY, type);

    if (!IsOnGrid(CornerUL))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(CornerUR))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(CornerLL))
        asThrowException(_("The given area does not match a gaussian grid."));
    if (!IsOnGrid(CornerLR))
        asThrowException(_("The given area does not match a gaussian grid."));
}

asGeoAreaCompositeGaussianGrid::asGeoAreaCompositeGaussianGrid(double Xmin, int Xptsnb, double Ymin, int Yptsnb,
                                                               asGeo::GridType type, float Level, float Height,
                                                               int flatAllowed)
        : asGeoAreaCompositeGrid(Level, Height)
{
    m_gridType = type;

    asGeoAreaGaussianGrid::BuildLonAxis(m_fullAxisX, type);
    asGeoAreaGaussianGrid::BuildLatAxis(m_fullAxisY, type);

    // Check input
    if (!IsOnGrid(Xmin, Ymin))
        asThrowException(wxString::Format(_("The given area does not match a gaussian grid (Xmin = %g, Ymin = %g)."),
                                          Xmin, Ymin));

    // Get real size to generate parent member variables
    int indexXmin = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmin, 0.01);
    int indexYmin = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymin, 0.01);
    wxASSERT(indexXmin >= 0);
    wxASSERT(indexYmin >= 0);
    wxASSERT(m_fullAxisX.size() > indexXmin + Xptsnb - 1);
    wxASSERT(m_fullAxisY.size() > indexYmin + Yptsnb - 1);
    wxASSERT(Xptsnb >= 0);
    wxASSERT(Yptsnb >= 0);
    if (m_fullAxisX.size() <= indexXmin + Xptsnb - 1)
        asThrowException(_("The given width exceeds the grid size of the gaussian grid."));
    if (m_fullAxisY.size() <= indexYmin + Yptsnb - 1)
        asThrowException(_("The given height exceeds the grid size of the gaussian grid."));
    if (Xptsnb < 0)
        asThrowException(wxString::Format(_("The given width (points number) is not consistent in the gaussian grid: %d"),
                                          Xptsnb));
    if (Yptsnb < 0)
        asThrowException(wxString::Format(_("The given height (points number) is not consistent in the gaussian grid: %d"),
                                          Yptsnb));
    double Xwidth = m_fullAxisX[indexXmin + Xptsnb - 1] - m_fullAxisX[indexXmin];
    double Ywidth = m_fullAxisY[indexYmin + Yptsnb - 1] - m_fullAxisY[indexYmin];

    // Regenerate with correct sizes
    Generate(Xmin, Xwidth, Ymin, Ywidth, flatAllowed);
}

asGeoAreaCompositeGaussianGrid::~asGeoAreaCompositeGaussianGrid()
{
    //dtor
}

bool asGeoAreaCompositeGaussianGrid::GridsOverlay(asGeoAreaCompositeGrid *otherarea) const
{
    if (otherarea->GetGridType() != GetGridType())
        return false;
    asGeoAreaCompositeGaussianGrid *otherareaGaussian(dynamic_cast<asGeoAreaCompositeGaussianGrid *>(otherarea));

    return otherareaGaussian->GetGridType() == GetGridType();
}

Array1DDouble asGeoAreaCompositeGaussianGrid::GetXaxisComposite(int compositeNb)
{
    double Xmin = GetComposite(compositeNb).GetXmin();
    double Xmax = GetComposite(compositeNb).GetXmax();

    int XminIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmin, 0.01);
    int XmaxIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmax, 0.01);

    wxASSERT(XminIndex >= 0);
    wxASSERT(XmaxIndex >= 0);
    wxASSERT(XmaxIndex >= XminIndex);

    return m_fullAxisX.segment(XminIndex, XmaxIndex - XminIndex + 1);
}

Array1DDouble asGeoAreaCompositeGaussianGrid::GetYaxisComposite(int compositeNb)
{
    double Ymin = GetComposite(compositeNb).GetYmin();
    double Ymax = GetComposite(compositeNb).GetYmax();

    int YminIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymin, 0.01);
    int YmaxIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymax, 0.01);

    wxASSERT(YminIndex >= 0);
    wxASSERT(YmaxIndex >= 0);
    wxASSERT(YmaxIndex >= YminIndex);

    return m_fullAxisY.segment(YminIndex, YmaxIndex - YminIndex + 1);
}

int asGeoAreaCompositeGaussianGrid::GetXaxisCompositePtsnb(int compositeNb)
{
    double Xmin = GetComposite(compositeNb).GetXmin();
    double Xmax = GetComposite(compositeNb).GetXmax();

    int XminIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmin, 0.01);
    int XmaxIndex = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xmax, 0.01);

    wxASSERT(XminIndex >= 0);
    wxASSERT(XmaxIndex >= 0);

    int ptsnb = XmaxIndex - XminIndex;

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
    double Ymin = GetComposite(compositeNb).GetYmin();
    double Ymax = GetComposite(compositeNb).GetYmax();

    int YminIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymin, 0.01);
    int YmaxIndex = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ymax, 0.01);

    wxASSERT(YminIndex >= 0);
    wxASSERT(YmaxIndex >= 0);

    int ptsnb = YmaxIndex - YminIndex;
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

    int foundX = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], point.x, 0.01);
    if ((foundX == asNOT_FOUND) || (foundX == asOUT_OF_RANGE))
        return false;

    int foundY = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], point.y, 0.01);
    if ((foundY == asNOT_FOUND) || (foundY == asOUT_OF_RANGE))
        return false;

    return true;
}

bool asGeoAreaCompositeGaussianGrid::IsOnGrid(double Xcoord, double Ycoord) const
{
    int foundX = asTools::SortedArraySearch(&m_fullAxisX[0], &m_fullAxisX[m_fullAxisX.size() - 1], Xcoord, 0.01);
    if ((foundX == asNOT_FOUND) || (foundX == asOUT_OF_RANGE))
        return false;

    int foundY = asTools::SortedArraySearch(&m_fullAxisY[0], &m_fullAxisY[m_fullAxisY.size() - 1], Ycoord, 0.01);
    if ((foundY == asNOT_FOUND) || (foundY == asOUT_OF_RANGE))
        return false;

    return true;
}
