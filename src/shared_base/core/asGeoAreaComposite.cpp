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

asGeoAreaComposite::asGeoAreaComposite(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level, float Height, int flatAllowed)
:
asGeo()
{
    // Set the members
    m_CornerUL = CornerUL;
    m_CornerUR = CornerUR;
    m_CornerLL = CornerLL;
    m_CornerLR = CornerLR;
    m_Level = Level;
    m_Height = Height;
    m_FlatAllowed = flatAllowed;
    m_AbsoluteXmin = m_CornerUL.x;
    m_AbsoluteXmax = m_CornerUR.x;
    m_AbsoluteYmin = m_CornerLL.y;
    m_AbsoluteYmax = m_CornerUL.y;

    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully created."));
}

asGeoAreaComposite::asGeoAreaComposite(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level, float Height, int flatAllowed)
:
asGeo()
{
    // Set the members
    m_CornerUL.x = Xmin;
    m_CornerUL.y = Ymin+Ywidth;
    m_CornerUR.x = Xmin+Xwidth;
    m_CornerUR.y = Ymin+Ywidth;
    m_CornerLL.x = Xmin;
    m_CornerLL.y = Ymin;
    m_CornerLR.x = Xmin+Xwidth;
    m_CornerLR.y = Ymin;
    m_Level = Level;
    m_Height = Height;
    m_FlatAllowed = flatAllowed;
    m_AbsoluteXmin = m_CornerUL.x;
    m_AbsoluteXmax = m_CornerUR.x;
    m_AbsoluteYmin = m_CornerLL.y;
    m_AbsoluteYmax = m_CornerUL.y;

    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully created."));
}

asGeoAreaComposite::asGeoAreaComposite(float Level, float Height)
:
asGeo()
{
    // Set the members
    m_Level = Level;
    m_Height = Height;
    m_NbComposites = 0;
    m_CornerUL.x = 0;
    m_CornerUL.y = 0;
    m_CornerUR.x = 0;
    m_CornerUR.y = 0;
    m_CornerLL.x = 0;
    m_CornerLL.y = 0;
    m_CornerLR.x = 0;
    m_CornerLR.y = 0;
    m_FlatAllowed = asFLAT_ALLOWED;
    m_AbsoluteXmin = 0;
    m_AbsoluteXmax = 0;
    m_AbsoluteYmin = 0;
    m_AbsoluteYmax = 0;
}

asGeoAreaComposite::~asGeoAreaComposite()
{
    //dtor
}

void asGeoAreaComposite::Generate(double Xmin, double Xwidth, double Ymin, double Ywidth, int flatAllowed)
{
    // Set the members
    m_CornerUL.x = Xmin;
    m_CornerUL.y = Ymin+Ywidth;
    m_CornerUR.x = Xmin+Xwidth;
    m_CornerUR.y = Ymin+Ywidth;
    m_CornerLL.x = Xmin;
    m_CornerLL.y = Ymin;
    m_CornerLR.x = Xmin+Xwidth;
    m_CornerLR.y = Ymin;
    m_FlatAllowed = flatAllowed;
    m_AbsoluteXmin = m_CornerUL.x;
    m_AbsoluteXmax = m_CornerUR.x;
    m_AbsoluteYmin = m_CornerLL.y;
    m_AbsoluteYmax = m_CornerUL.y;

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
        if ((m_CornerUL.x == m_CornerUR.x) || (m_CornerLL.x == m_CornerLR.x) || (m_CornerLL.y == m_CornerUL.y) || (m_CornerLR.y == m_CornerUR.y))
        {
            return false;
        }
    }

    // Lon min is on the edge and should be corrected
    if ((m_CornerUL.x > m_CornerUR.x) && (m_CornerUL.x == m_AxisXmax))
    {
        m_CornerUL.x -= m_AxisXmax;
    }
    if ((m_CornerLL.x > m_CornerLR.x) && (m_CornerLL.x == m_AxisXmax))
    {
        m_CornerLL.x -= m_AxisXmax;
    }

    // Coordinates order vary
    if ((m_CornerUL.x > m_CornerUR.x) || (m_CornerLL.x > m_CornerLR.x) || (m_CornerLL.y > m_CornerUL.y) || (m_CornerLR.y > m_CornerUR.y))
    {
        // Do not proceed to change
        wxLogVerbose(_("The given coordinates are not increasing. This is a normal behavior if the area is on the coordinates edge."));
    }

    return true;
}

double asGeoAreaComposite::GetXmin()
{
    double RealXmin = InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealXmin = wxMin(RealXmin, m_Composites[i_area].GetXmin());
    }
    return RealXmin;
}

double asGeoAreaComposite::GetXmax()
{
    double RealXmax = -InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealXmax = wxMax(RealXmax, m_Composites[i_area].GetXmax());
    }
    return RealXmax;
}

double asGeoAreaComposite::GetYmin()
{
    double RealYmin = InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealYmin = wxMin(RealYmin, m_Composites[i_area].GetYmin());
    }
    return RealYmin;
}

double asGeoAreaComposite::GetYmax()
{
    double RealYmax = -InfDouble;
    for (int i_area = 0; i_area<m_NbComposites; i_area++)
    {
        RealYmax = wxMax(RealYmax, m_Composites[i_area].GetYmax());
    }
    return RealYmax;
}

Coo asGeoAreaComposite::GetCenter()
{
    Coo center;

    if((m_CornerUL.x<m_CornerUR.x) & (m_CornerLL.x<m_CornerLR.x) & (m_CornerLL.y<m_CornerUL.y) & (m_CornerLR.y<m_CornerUR.y))
    {
        center = m_Composites[0].GetCenter();
    }
    else if((m_CornerUL.x>m_CornerUR.x) & (m_CornerLL.x>m_CornerLR.x) & (m_CornerLL.y<m_CornerUL.y) & (m_CornerLR.y<m_CornerUR.y))
    {
        double CornerUR = 360+m_CornerUR.x;
        double CornerLR = 360+m_CornerLR.x;
        double Xmin = wxMin(m_CornerUL.x, m_CornerLL.x);
        double Xmax = wxMin(CornerUR, CornerLR);
        center.x = Xmin + (Xmax-Xmin)/2;
        center.y = GetYmin() + (GetYmax()-GetYmin())/2;
        return center;
    }
    else
    {
// TODO (phorton#1#): Implement me !
        asThrowException(_("This case is not managed yet."));
    }


    center.x = GetXmin() + (GetXmax()-GetXmin())/2;
    center.y = GetYmin() + (GetYmax()-GetYmin())/2;
    return center;
}

bool asGeoAreaComposite::IsRectangle()
{
    // Check that the area is a square
    if ((m_CornerUL.x != m_CornerLL.x) | (m_CornerUL.y != m_CornerUR.y) | (m_CornerUR.x != m_CornerLR.x) | (m_CornerLL.y != m_CornerLR.y))
    {
        return false;
    }
    return true;
}

void asGeoAreaComposite::CreateComposites()
{
    m_Composites.clear();
    m_NbComposites = 0;

    if((m_CornerUL.x<=m_CornerUR.x) & (m_CornerLL.x<=m_CornerLR.x) & (m_CornerLL.y<=m_CornerUL.y) & (m_CornerLR.y<=m_CornerUR.y))
    {
        asGeoArea area(m_CornerUL, m_CornerUR, m_CornerLL, m_CornerLR, m_Level, m_Height, m_FlatAllowed);
        m_Composites.push_back(area);
        m_NbComposites = 1;
    }
    else if((m_CornerUL.x>=m_CornerUR.x) & (m_CornerLL.x>=m_CornerLR.x) & (m_CornerLL.y<=m_CornerUL.y) & (m_CornerLR.y<=m_CornerUR.y) & (m_CornerLR.x==m_AxisXmin) & (m_CornerUR.x==m_AxisXmin))
    {
        m_CornerLR.x = m_AxisXmax;
        m_CornerUR.x = m_AxisXmax;
        asGeoArea area(m_CornerUL, m_CornerUR, m_CornerLL, m_CornerLR, m_Level, m_Height, m_FlatAllowed);
        m_Composites.push_back(area);
        m_NbComposites = 1;
    }
    else if((m_CornerUL.x>=m_CornerUR.x) & (m_CornerLL.x>=m_CornerLR.x) & (m_CornerLL.y<=m_CornerUL.y) & (m_CornerLR.y<=m_CornerUR.y) & (m_CornerLR.x!=m_AxisXmin) & (m_CornerUR.x!=m_AxisXmin))
    {
        Coo a1UL = m_CornerUL, a1UR = m_CornerUR, a1LL = m_CornerLL, a1LR = m_CornerLR;
        Coo a2UL = m_CornerUL, a2UR = m_CornerUR, a2LL = m_CornerLL, a2LR = m_CornerLR;
        a1UL.x = m_AxisXmin;
        a1LL.x = m_AxisXmin;
        a2UR.x = m_AxisXmax;
        a2LR.x = m_AxisXmax;
        asGeoArea area1(a1UL, a1UR, a1LL, a1LR, m_Level, m_Height, m_FlatAllowed);
        asGeoArea area2(a2UL, a2UR, a2LL, a2LR, m_Level, m_Height, m_FlatAllowed);
        m_Composites.push_back(area1);
        m_Composites.push_back(area2);
        m_NbComposites = 2;
    }
    else
    {
// TODO (phorton#1#): Implement me and check the other functions (GetCenter(), ...)!
        wxString error = "This case is not managed yet (asGeoAreaComposite::CreateComposites):\n ";
        error.Append(wxString::Format( "m_CornerUL.x = %g\n", m_CornerUL.x ));
        error.Append(wxString::Format( "m_CornerUR.x = %g\n", m_CornerUR.x ));
        error.Append(wxString::Format( "m_CornerLL.x = %g\n", m_CornerLL.x ));
        error.Append(wxString::Format( "m_CornerLR.x = %g\n", m_CornerLR.x ));
        error.Append(wxString::Format( "m_CornerLL.y = %g\n", m_CornerLL.y ));
        error.Append(wxString::Format( "m_CornerUL.y = %g\n", m_CornerUL.y ));
        error.Append(wxString::Format( "m_CornerLR.y = %g\n", m_CornerLR.y ));
        error.Append(wxString::Format( "m_CornerUR.y = %g\n", m_CornerUR.y ));
        asLogError(error);
        asThrowException(_("This case is not managed yet."));
    }
}
