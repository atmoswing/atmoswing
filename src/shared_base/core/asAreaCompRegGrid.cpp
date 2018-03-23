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

#include "asAreaCompRegGrid.h"

asAreaCompRegGrid::asAreaCompRegGrid(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                                     double xStep, double yStep, float level, int flatAllowed)
        : asAreaCompGrid(cornerUL, cornerUR, cornerLL, cornerLR, level, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    m_gridType = Regular;

    if (!asIsNaN(xStep) && !asIsNaN(yStep) && xStep > 0 && yStep > 0) {
        if (!IsOnGrid(xStep, yStep))
            asThrowException(_("The given area does not match a grid."));
        m_axesInitialized = true;
    }
}

asAreaCompRegGrid::asAreaCompRegGrid(double xMin, double xWidth, double xStep, double yMin, double yWidth, double yStep,
                                     float level, int flatAllowed)
        : asAreaCompGrid(xMin, xWidth, yMin, yWidth, level, flatAllowed),
          m_xStep(xStep),
          m_yStep(yStep)
{
    m_gridType = Regular;

    if (!asIsNaN(xStep) && !asIsNaN(yStep) && xStep > 0 && yStep > 0) {
        if (!IsOnGrid(xStep, yStep))
            asThrowException(_("The given area does not match a grid."));
        m_axesInitialized = true;
    }
}

bool asAreaCompRegGrid::GridsOverlay(asAreaCompGrid *otherarea) const
{
    if (otherarea->GetGridType() != Regular)
        return false;

    auto *otherareaRegular(dynamic_cast<asAreaCompRegGrid *>(otherarea));

    if (!otherareaRegular)
        return false;

    if (GetXstep() != otherareaRegular->GetXstep())
        return false;

    if (GetYstep() != otherareaRegular->GetYstep())
        return false;

    return true;
}

a1d asAreaCompRegGrid::GetXaxisComposite(int compositeNb)
{
    // Get axis size
    int size = GetXaxisCompositePtsnb(compositeNb);
    a1d xAxis(size);

    // Build array
    double xMin = GetComposite(compositeNb).GetXmin();
    if (compositeNb == 0) // Left border
    {
        double xMax = GetComposite(compositeNb).GetXmax();
        double restovers = xMax - xMin - m_xStep * (size - 1);
        xMin += restovers;
    }

    for (int i = 0; i < size; i++) {
        xAxis(i) = xMin + (double) i * m_xStep;
    }

    return xAxis;
}

a1d asAreaCompRegGrid::GetYaxisComposite(int compositeNb)
{
    // Get axis size
    int size = GetYaxisCompositePtsnb(compositeNb);
    a1d yAxis(size);

    // Build array
    double yMin = GetComposite(compositeNb).GetYmin();
    // FIXME (Pascal#3#): Check the compositeNb==0 in this case
    if (compositeNb == 0) // Not sure...
    {
        double yMax = GetComposite(compositeNb).GetYmax();
        double restovers = yMax - yMin - m_yStep * (size - 1);
        yMin += restovers;
    }

    for (int i = 0; i < size; i++) {
        yAxis(i) = yMin + i * m_yStep;
    }
    //wxASSERT(Yaxis(size-1)==GetComposite(compositeNb).GetYmax()); // Not always true

    return yAxis;
}

int asAreaCompRegGrid::GetXaxisCompositePtsnb(int compositeNb)
{
    double diff = std::abs((GetComposite(compositeNb).GetXmax() - GetComposite(compositeNb).GetXmin())) / m_xStep;
    double size;
    double rest = modf(diff, &size);

    if (compositeNb == 0) // from 0
    {
        size += 1;
    } else if (compositeNb == 1) // to 360
    {
        size += 1;
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }

    if (rest < 0.0000001 || rest > 0.999999) //Precision issue
    {
        return size + asRound(rest);
    } else {
        return size;
    }
}

int asAreaCompRegGrid::GetYaxisCompositePtsnb(int compositeNb)
{
    double diff = std::abs((GetComposite(compositeNb).GetYmax() - GetComposite(compositeNb).GetYmin())) / m_yStep;
    double size;
    double rest = modf(diff, &size);
    size += 1;

    if (rest < 0.0000001 || rest > 0.999999) //Precision issue
    {
        return size + asRound(rest);
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asAreaCompRegGrid::GetXaxisCompositeWidth(int compositeNb) const
{
    return std::abs(GetComposite(compositeNb).GetXmax() - GetComposite(compositeNb).GetXmin());
}

double asAreaCompRegGrid::GetYaxisCompositeWidth(int compositeNb) const
{
    return std::abs(GetComposite(compositeNb).GetYmax() - GetComposite(compositeNb).GetYmin());
}

double asAreaCompRegGrid::GetXaxisCompositeStart(int compositeNb) const
{
    // If only one composite
    if (GetNbComposites() == 1) {
        return GetComposite(compositeNb).GetXmin();
    }

    // If multiple composites
    if (compositeNb == 0) // from 0
    {
        // Composites are not forced on the grid. So we may need to adjust the split of the longitudes axis.
        double dX = std::abs(GetComposite(1).GetXmax() - GetComposite(1).GetXmin());

        if (std::fmod(dX, m_xStep) < 0.000001) {
            return GetComposite(compositeNb).GetXmin();
        } else {
            double rest = std::fmod(dX, m_xStep);
            return m_xStep - rest;
        }
    } else if (compositeNb == 1) // to 360
    {
        return GetComposite(compositeNb).GetXmin();
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asAreaCompRegGrid::GetYaxisCompositeStart(int compositeNb) const
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

double asAreaCompRegGrid::GetXaxisCompositeEnd(int compositeNb) const
{
    // If only one composite
    if (GetNbComposites() == 1) {
        return GetComposite(compositeNb).GetXmax();
    }

    // If multiple composites
    if (compositeNb == 1) // to 360
    {
        // Composites are not forced on the grid. So we may need to adjust the split of the longitudes axis.
        double dX = std::abs(GetComposite(1).GetXmax() - GetComposite(1).GetXmin());
        double rest = std::fmod(dX, m_xStep);
        if (rest < 0.000001) {
            return GetComposite(compositeNb).GetXmax();
        } else {
            return GetComposite(compositeNb).GetXmax() - rest;
        }
    } else if (compositeNb == 0) // from 0
    {
        return GetComposite(compositeNb).GetXmax();
    } else {
        asThrowException(_("The latitude split is not implemented yet."));
    }
}

double asAreaCompRegGrid::GetYaxisCompositeEnd(int compositeNb) const
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

bool asAreaCompRegGrid::IsOnGrid(double step) const
{
    if (!IsRectangle())
        return false;

    if (std::abs(std::fmod(GetXaxisWidth(), step)) > 0.0000001)
        return false;
    if (std::abs(std::fmod(GetYaxisWidth(), step)) > 0.0000001)
        return false;

    return true;
}

bool asAreaCompRegGrid::IsOnGrid(double stepX, double stepY) const
{
    if (!IsRectangle())
        return false;

    if (std::abs(std::fmod(GetXaxisWidth(), stepX)) > 0.0000001)
        return false;
    if (std::abs(std::fmod(GetYaxisWidth(), stepY)) > 0.0000001)
        return false;

    return true;
}
