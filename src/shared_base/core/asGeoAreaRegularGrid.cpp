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

asGeoAreaRegularGrid::asGeoAreaRegularGrid(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, double Ustep, double Vstep, float Level, float Height, int flatAllowed)
:
asGeoArea(coosys, CornerUL, CornerUR, CornerLL, CornerLR, Level, Height, flatAllowed)
{
    if(!IsOnGrid(Ustep, Vstep)) asThrowException(_("The given area does not match a grid."));

    m_Ustep = Ustep;
    m_Vstep = Vstep;
}

asGeoAreaRegularGrid::asGeoAreaRegularGrid(CoordSys coosys, double Umin, double Uwidth, double Ustep, double Vmin, double Vwidth, double Vstep, float Level, float Height, int flatAllowed)
:
asGeoArea(coosys, Umin, Uwidth, Vmin, Vwidth, Level, Height, flatAllowed)
{
    if(!IsOnGrid(Ustep, Vstep)) asThrowException(_("The given area does not match a grid."));

    m_Ustep = Ustep;
    m_Vstep = Vstep;
}

asGeoAreaRegularGrid::~asGeoAreaRegularGrid()
{
    //dtor
}

int asGeoAreaRegularGrid::GetUaxisPtsnb()
{
    // Get axis size
    return asTools::Round(abs((GetUmax()-GetUmin())/m_Ustep)+1);
}

int asGeoAreaRegularGrid::GetVaxisPtsnb()
{
    // Get axis size
    return asTools::Round(abs((GetVmax()-GetVmin())/m_Ustep)+1);
}

Array1DDouble asGeoAreaRegularGrid::GetUaxis()
{
    // Get axis size
    int ptsnb = GetUaxisPtsnb();
    Array1DDouble Uaxis = Array1DDouble(ptsnb);

    // Build array
    double umin = GetUmin();
    for (int i=0; i<ptsnb; i++)
    {
        Uaxis(i) = umin+i*m_Ustep;
    }
    wxASSERT(Uaxis(ptsnb-1)==GetUmax());

    return Uaxis;
}

Array1DDouble asGeoAreaRegularGrid::GetVaxis()
{
    // Get axis size
    int ptsnb = GetVaxisPtsnb();
    Array1DDouble Vaxis = Array1DDouble(ptsnb);

    // Build array
    double vmin = GetVmin();
    for (int i=0; i<ptsnb; i++)
    {
        Vaxis(i) = vmin+i*m_Vstep;
    }
    wxASSERT(Vaxis(ptsnb-1)==GetVmax());

    return Vaxis;
}

bool asGeoAreaRegularGrid::IsOnGrid(double step)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(m_CornerUL.u-m_CornerUR.u,step))>0.0000001) return false;
    if (abs(fmod(m_CornerUL.v-m_CornerLL.v,step))>0.0000001) return false;

    return true;
}

bool asGeoAreaRegularGrid::IsOnGrid(double stepU, double stepV)
{
    if (!IsRectangle()) return false;

    if (abs(fmod(m_CornerUL.u-m_CornerUR.u,stepU))>0.0000001) return false;
    if (abs(fmod(m_CornerUL.v-m_CornerLL.v,stepV))>0.0000001) return false;

    return true;
}
