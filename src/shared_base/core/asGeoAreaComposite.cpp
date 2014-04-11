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
 
#include "asGeoAreaComposite.h"

asGeoAreaComposite::asGeoAreaComposite(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level, float Height, int flatAllowed)
:
asGeo(coosys)
{
    // Set the members
    m_CoordSys = coosys;
    m_CornerUL = CornerUL;
    m_CornerUR = CornerUR;
    m_CornerLL = CornerLL;
    m_CornerLR = CornerLR;
    m_Level = Level;
    m_Height = Height;
    m_FlatAllowed = flatAllowed;
    m_AbsoluteUmin = m_CornerUL.u;
    m_AbsoluteUmax = m_CornerUR.u;
    m_AbsoluteVmin = m_CornerLL.v;
    m_AbsoluteVmax = m_CornerUL.v;

    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully created."));
}

asGeoAreaComposite::asGeoAreaComposite(CoordSys coosys, double Umin, double Uwidth, double Vmin, double Vwidth, float Level, float Height, int flatAllowed)
:
asGeo(coosys)
{
    // Set the members
    m_CoordSys = coosys;
    m_CornerUL.u = Umin;
    m_CornerUL.v = Vmin+Vwidth;
    m_CornerUR.u = Umin+Uwidth;
    m_CornerUR.v = Vmin+Vwidth;
    m_CornerLL.u = Umin;
    m_CornerLL.v = Vmin;
    m_CornerLR.u = Umin+Uwidth;
    m_CornerLR.v = Vmin;
    m_Level = Level;
    m_Height = Height;
    m_FlatAllowed = flatAllowed;
    m_AbsoluteUmin = m_CornerUL.u;
    m_AbsoluteUmax = m_CornerUR.u;
    m_AbsoluteVmin = m_CornerLL.v;
    m_AbsoluteVmax = m_CornerUL.v;

    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully created."));
}

asGeoAreaComposite::asGeoAreaComposite(CoordSys coosys, float Level, float Height)
:
asGeo(coosys)
{
    // Set the members
    m_CoordSys = coosys;
    m_Level = Level;
    m_Height = Height;

    m_CornerUL.u = 0;
    m_CornerUL.v = 0;
    m_CornerUR.u = 0;
    m_CornerUR.v = 0;
    m_CornerLL.u = 0;
    m_CornerLL.v = 0;
    m_CornerLR.u = 0;
    m_CornerLR.v = 0;
    m_FlatAllowed = asFLAT_ALLOWED;
    m_AbsoluteUmin = 0;
    m_AbsoluteUmax = 0;
    m_AbsoluteVmin = 0;
    m_AbsoluteVmax = 0;
}

asGeoAreaComposite::~asGeoAreaComposite()
{
    //dtor
}

void asGeoAreaComposite::Generate(double Umin, double Uwidth, double Vmin, double Vwidth, int flatAllowed)
{
    // Set the members
    m_CornerUL.u = Umin;
    m_CornerUL.v = Vmin+Vwidth;
    m_CornerUR.u = Umin+Uwidth;
    m_CornerUR.v = Vmin+Vwidth;
    m_CornerLL.u = Umin;
    m_CornerLL.v = Vmin;
    m_CornerLR.u = Umin+Uwidth;
    m_CornerLR.v = Vmin;
    m_FlatAllowed = flatAllowed;
    m_AbsoluteUmin = m_CornerUL.u;
    m_AbsoluteUmax = m_CornerUR.u;
    m_AbsoluteVmin = m_CornerLL.v;
    m_AbsoluteVmax = m_CornerUL.v;

    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully generated."));
}

void asGeoAreaComposite::Init()
{
    InitBounds();
    DoCheckPoints();
    if (!CheckConsistency()) asThrowException(_("Unable to build a consistent area with the given coordinates."));
}

bool asGeoAreaComposite::DoCheckPoints()
{
    // Check the points and proceed to changes if necessary
    CheckPoint(m_CornerUL, asEDIT_ALLOWED);
    CheckPoint(m_CornerUR, asEDIT_ALLOWED);
    CheckPoint(m_CornerLL, asEDIT_ALLOWED);
    CheckPoint(m_CornerLR, asEDIT_ALLOWED);
    return true;
}

bool asGeoAreaComposite::CheckConsistency()
{
    // Area is a single point
    if (m_FlatAllowed == asFLAT_FORBIDDEN)
    {
        if ((m_CornerUL.u == m_CornerUR.u) || (m_CornerLL.u == m_CornerLR.u) || (m_CornerLL.v == m_CornerUL.v) || (m_CornerLR.v == m_CornerUR.v))
        {
            return false;
        }
    }

    // Lon min is on the edge and should be corrected
    if ((m_CornerUL.u > m_CornerUR.u) && (m_CornerUL.u == m_AxisUmax))
    {
        m_CornerUL.u -= m_AxisUmax;
    }
    if ((m_CornerLL.u > m_CornerLR.u) && (m_CornerLL.u == m_AxisUmax))
    {
        m_CornerLL.u -= m_AxisUmax;
    }

    // Coordinates order vary
    if ((m_CornerUL.u > m_CornerUR.u) || (m_CornerLL.u > m_CornerLR.u) || (m_CornerLL.v > m_CornerUL.v) || (m_CornerLR.v > m_CornerUR.v))
    {
        // Do not proceed to change
        wxLogVerbose(_("The given coordinates are not increasing. This is a normal behavior if the area is on the coordinates edge."));
    }

    return true;
}

double asGeoAreaComposite::GetUmin()
{
    double RealUmin = InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealUmin = wxMin(RealUmin, m_Composites[i_area].GetUmin());
    }
    return RealUmin;
}

double asGeoAreaComposite::GetUmax()
{
    double RealUmax = -InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealUmax = wxMax(RealUmax, m_Composites[i_area].GetUmax());
    }
    return RealUmax;
}

double asGeoAreaComposite::GetVmin()
{
    double RealVmin = InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealVmin = wxMin(RealVmin, m_Composites[i_area].GetVmin());
    }
    return RealVmin;
}

double asGeoAreaComposite::GetVmax()
{
    double RealVmax = -InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealVmax = wxMax(RealVmax, m_Composites[i_area].GetVmax());
    }
    return RealVmax;
}

Coo asGeoAreaComposite::GetCenter()
{
    Coo center;

    if((m_CornerUL.u<m_CornerUR.u) & (m_CornerLL.u<m_CornerLR.u) & (m_CornerLL.v<m_CornerUL.v) & (m_CornerLR.v<m_CornerUR.v))
    {
        center = m_Composites[0].GetCenter();
    }
    else if((m_CornerUL.u>m_CornerUR.u) & (m_CornerLL.u>m_CornerLR.u) & (m_CornerLL.v<m_CornerUL.v) & (m_CornerLR.v<m_CornerUR.v))
    {
        double CornerUR = 360+m_CornerUR.u;
        double CornerLR = 360+m_CornerLR.u;
        double Umin = wxMin(m_CornerUL.u, m_CornerLL.u);
        double Umax = wxMin(CornerUR, CornerLR);
        center.u = Umin + (Umax-Umin)/2;
        center.v = GetVmin() + (GetVmax()-GetVmin())/2;
        return center;
    }
    else
    {
// TODO (phorton#1#): Implement me !
        asThrowException(_("This case is not managed yet."));
    }


    center.u = GetUmin() + (GetUmax()-GetUmin())/2;
    center.v = GetVmin() + (GetVmax()-GetVmin())/2;
    return center;
}

bool asGeoAreaComposite::IsRectangle()
{
    // Check that the area is a square
    if ((m_CornerUL.u != m_CornerLL.u) | (m_CornerUL.v != m_CornerUR.v) | (m_CornerUR.u != m_CornerLR.u) | (m_CornerLL.v != m_CornerLR.v))
    {
        return false;
    }
    return true;
}

void asGeoAreaComposite::ProjConvert(CoordSys newcoordsys)
{
    m_CornerUL = ProjTransform(newcoordsys, m_CornerUL);
    m_CornerUR = ProjTransform(newcoordsys, m_CornerUR);
    m_CornerLL = ProjTransform(newcoordsys, m_CornerLL);
    m_CornerLR = ProjTransform(newcoordsys, m_CornerLR);
    m_CoordSys = newcoordsys;

    // Initialization and check points
    Init();
    CreateComposites();
}

void asGeoAreaComposite::CreateComposites()
{
    m_Composites.clear();
    m_NbComposites = 0;

    if((m_CornerUL.u<=m_CornerUR.u) & (m_CornerLL.u<=m_CornerLR.u) & (m_CornerLL.v<=m_CornerUL.v) & (m_CornerLR.v<=m_CornerUR.v))
    {
        asGeoArea area(m_CoordSys, m_CornerUL, m_CornerUR, m_CornerLL, m_CornerLR, m_Level, m_Height, m_FlatAllowed);
        m_Composites.push_back(area);
        m_NbComposites = 1;
    }
    else if((m_CornerUL.u>=m_CornerUR.u) & (m_CornerLL.u>=m_CornerLR.u) & (m_CornerLL.v<=m_CornerUL.v) & (m_CornerLR.v<=m_CornerUR.v) & (m_CornerLR.u==m_AxisUmin) & (m_CornerUR.u==m_AxisUmin))
    {
        m_CornerLR.u = m_AxisUmax;
        m_CornerUR.u = m_AxisUmax;
        asGeoArea area(m_CoordSys, m_CornerUL, m_CornerUR, m_CornerLL, m_CornerLR, m_Level, m_Height, m_FlatAllowed);
        m_Composites.push_back(area);
        m_NbComposites = 1;
    }
    else if((m_CornerUL.u>=m_CornerUR.u) & (m_CornerLL.u>=m_CornerLR.u) & (m_CornerLL.v<=m_CornerUL.v) & (m_CornerLR.v<=m_CornerUR.v) & (m_CornerLR.u!=m_AxisUmin) & (m_CornerUR.u!=m_AxisUmin))
    {
        Coo a1UL = m_CornerUL, a1UR = m_CornerUR, a1LL = m_CornerLL, a1LR = m_CornerLR;
        Coo a2UL = m_CornerUL, a2UR = m_CornerUR, a2LL = m_CornerLL, a2LR = m_CornerLR;
        a1UL.u = m_AxisUmin;
        a1LL.u = m_AxisUmin;
        a2UR.u = m_AxisUmax;
        a2LR.u = m_AxisUmax;
        asGeoArea area1(m_CoordSys, a1UL, a1UR, a1LL, a1LR, m_Level, m_Height, m_FlatAllowed);
        asGeoArea area2(m_CoordSys, a2UL, a2UR, a2LL, a2LR, m_Level, m_Height, m_FlatAllowed);
        m_Composites.push_back(area1);
        m_Composites.push_back(area2);
        m_NbComposites = 2;
    }
    else
    {
// TODO (phorton#1#): Implement me and check the other functions (GetCenter(), ...)!
        wxString error = "This case is not managed yet (asGeoAreaComposite::CreateComposites):\n ";
        error.Append(wxString::Format( "m_CornerUL.u = %g\n", m_CornerUL.u ));
        error.Append(wxString::Format( "m_CornerUR.u = %g\n", m_CornerUR.u ));
        error.Append(wxString::Format( "m_CornerLL.u = %g\n", m_CornerLL.u ));
        error.Append(wxString::Format( "m_CornerLR.u = %g\n", m_CornerLR.u ));
        error.Append(wxString::Format( "m_CornerLL.v = %g\n", m_CornerLL.v ));
        error.Append(wxString::Format( "m_CornerUL.v = %g\n", m_CornerUL.v ));
        error.Append(wxString::Format( "m_CornerLR.v = %g\n", m_CornerLR.v ));
        error.Append(wxString::Format( "m_CornerUR.v = %g\n", m_CornerUR.v ));
        asLogError(error);
        asThrowException(_("This case is not managed yet."));
    }
}
