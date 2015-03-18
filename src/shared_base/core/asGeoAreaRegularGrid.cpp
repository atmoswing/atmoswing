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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include "asGeoAreaRegularGrid.h"

asGeoAreaRegularGrid::asGeoAreaRegularGrid(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, double Xstep, double Ystep, float Level, float Height, int flatAllowed)
:
asGeoArea(CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    if(!IsOnGrid(Xstep, Ystep)) asThrowException(_("The given area does not match a grid."));

    m_xstep = Xstep;
    m_ystep = Ystep;
}

asGeoAreaRegularGrid::asGeoAreaRegularGrid(double Xmin, double Xwidth, double Xstep, double Ymin, double Ywidth, double Ystep, float Level, float Height, int flatAllowed)
:
asGeoArea(Xmin, Xwidth, Ymin, Ywidth, Level, Height, flatAllowed)
{
    if(!IsOnGrid(Xstep, Ystep)) asThrowException(_("The given area does not match a grid."));

    m_xstep = Xstep;
    m_ystep = Ystep;
}

asGeoAreaRegularGrid::~asGeoAreaRegularGrid()
{
    //dtor
}

int asGeoAreaRegularGrid::GetXaxisPtsnb()
{
    // Get axis size
    return asTools::Round(abs((GetXmax()-GetXmin())/m_xstep)+1.0);
}

int asGeoAreaRegularGrid::GetYaxisPtsnb()
{
    // Get axis size
    return asTools::Round(abs((GetYmax()-GetYmin())/m_xstep)+1.0);
}

Array1DDouble asGeoAreaRegularGrid::GetXaxis()
{
    // Get axis size
    int ptsnb = GetXaxisPtsnb();
    Array1DDouble Xaxis = Array1DDouble(ptsnb);

    // Build array
    double Xmin = GetXmin();
    for (int i=0; i<ptsnb; i++)
    {
        Xaxis(i) = Xmin+i*m_xstep;
    }
    wxASSERT(Xaxis(ptsnb-1)==GetXmax());

    return Xaxis;
}

Array1DDouble asGeoAreaRegularGrid::GetYaxis()
{
    // Get axis size
    int ptsnb = GetYaxisPtsnb();
    Array1DDouble Yaxis = Array1DDouble(ptsnb);

    // Build array
    double vmin = GetYmin();
    for (int i=0; i<ptsnb; i++)
    {
        Yaxis(i) = vmin+i*m_ystep;
    }
    wxASSERT(Yaxis(ptsnb-1)==GetYmax());

    return Yaxis;
}

bool asGeoAreaRegularGrid::IsOnGrid(double step)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(m_cornerUL.x-m_cornerUR.x,step))>0.0000001) return false;
    if (abs(fmod(m_cornerUL.y-m_cornerLL.y,step))>0.0000001) return false;

    return true;
}

bool asGeoAreaRegularGrid::IsOnGrid(double stepX, double stepY)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(m_cornerUL.x-m_cornerUR.x,stepX))>0.0000001) return false;
    if (abs(fmod(m_cornerUL.y-m_cornerLL.y,stepY))>0.0000001) return false;

    return true;
}
