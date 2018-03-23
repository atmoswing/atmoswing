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

#include "asAreaRegGrid.h"

asAreaRegGrid::asAreaRegGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                             double xStep, double yStep, float level, int flatAllowed)
        : asArea(cornerUL, cornerUR, cornerLL, cornerLR, level, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    if (!IsOnGrid(m_xStep, m_yStep))
        asThrowException(_("The given area does not match a grid."));
}

asAreaRegGrid::asAreaRegGrid(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                             float level, int flatAllowed)
        : asArea(xMin, xWidth, yMin, yWidth, level, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    if (!IsOnGrid(m_xStep, m_yStep))
        asThrowException(_("The given area does not match a grid."));
}

int asAreaRegGrid::GetXaxisPtsnb() const
{
    // Get axis size
    return asRound(std::abs((GetXmax() - GetXmin()) / m_xStep) + 1.0);
}

int asAreaRegGrid::GetYaxisPtsnb() const
{
    // Get axis size
    return asRound(std::abs((GetYmax() - GetYmin()) / m_xStep) + 1.0);
}

a1d asAreaRegGrid::GetXaxis() const
{
    // Get axis size
    int ptsnb = GetXaxisPtsnb();
    a1d xAxis(ptsnb);

    // Build array
    double xMin = GetXmin();
    for (int i = 0; i < ptsnb; i++) {
        xAxis(i) = xMin + i * m_xStep;
    }
    wxASSERT(xAxis(ptsnb - 1) == GetXmax());

    return xAxis;
}

a1d asAreaRegGrid::GetYaxis() const
{
    // Get axis size
    int ptsnb = GetYaxisPtsnb();
    a1d yAxis(ptsnb);

    // Build array
    double vmin = GetYmin();
    for (int i = 0; i < ptsnb; i++) {
        yAxis(i) = vmin + i * m_yStep;
    }
    wxASSERT(yAxis(ptsnb - 1) == GetYmax());

    return yAxis;
}

bool asAreaRegGrid::IsOnGrid(double step) const
{
    if (!IsRectangle())
        return false;

    if (std::abs(std::fmod(m_cornerUL.x - m_cornerUR.x, step)) > 0.0000001)
        return false;
    if (std::abs(std::fmod(m_cornerUL.y - m_cornerLL.y, step)) > 0.0000001)
        return false;

    return true;
}

bool asAreaRegGrid::IsOnGrid(double stepX, double stepY) const
{
    if (!IsRectangle())
        return false;

    if (std::abs(std::fmod(m_cornerUL.x - m_cornerUR.x, stepX)) > 0.0000001)
        return false;
    if (std::abs(std::fmod(m_cornerUL.y - m_cornerLL.y, stepY)) > 0.0000001)
        return false;

    return true;
}
