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
 
#include "asGeoArea.h"

asGeoArea::asGeoArea(CoordSys coosys, const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level, float Height, int flatAllowed)
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

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

asGeoArea::asGeoArea(CoordSys coosys, double Umin, double Uwidth, double Vmin, double Vwidth, float Level, float Height, int flatAllowed)
:
asGeo(coosys)
{
    if (flatAllowed==asFLAT_ALLOWED)
    {
        Vwidth = wxMax(Vwidth, 0.0);
        Uwidth = wxMax(Uwidth, 0.0);
    }
    else
    {
        wxASSERT(Vwidth>0);
        wxASSERT(Uwidth>0);
    }

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

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

asGeoArea::asGeoArea(CoordSys coosys, float Level, float Height)
:
asGeo(coosys)
{
    // Set the members
    m_CoordSys = coosys;
    m_CornerUL.u = 0;
    m_CornerUL.v = 0;
    m_CornerUR.u = 0;
    m_CornerUR.v = 0;
    m_CornerLL.u = 0;
    m_CornerLL.v = 0;
    m_CornerLR.u = 0;
    m_CornerLR.v = 0;
    m_Level = Level;
    m_Height = Height;
    m_FlatAllowed = asFLAT_ALLOWED;
}

asGeoArea::~asGeoArea()
{
    //dtor
}

void asGeoArea::Generate(double Umin, double Uwidth, double Vmin, double Vwidth, int flatAllowed)
{
    if (flatAllowed==asFLAT_ALLOWED)
    {
        Vwidth = wxMax(Vwidth, 0.0);
        Uwidth = wxMax(Uwidth, 0.0);
    }
    else
    {
        wxASSERT(Vwidth>0);
        wxASSERT(Uwidth>0);
    }

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

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

void asGeoArea::Init()
{
    InitBounds();
    if (!DoCheckPoints()) asThrowException(_("Use asGeoAreaComposite in this case."));
    if (!CheckConsistency()) asThrowException(_("Unable to build a consistent area with the given coordinates."));
}

bool asGeoArea::DoCheckPoints()
{
    if (!CheckPoint(m_CornerUL, asEDIT_FORBIDEN) || !CheckPoint(m_CornerUR, asEDIT_FORBIDEN) || !CheckPoint(m_CornerLL, asEDIT_FORBIDEN) || !CheckPoint(m_CornerLR, asEDIT_FORBIDEN))
    {
        return false;
    }
    return true;
}

bool asGeoArea::CheckConsistency()
{
    Coo cootmp;

    if (m_FlatAllowed == asFLAT_FORBIDDEN)
    {
        if ((m_CornerUL.u == m_CornerUR.u) || (m_CornerLL.u == m_CornerLR.u) || (m_CornerLL.v == m_CornerUL.v) || (m_CornerLR.v == m_CornerUR.v))
        {
            return false;
        }
    }

    if (m_CornerUL.u > m_CornerUR.u)
    {
        cootmp = m_CornerUR;
        m_CornerUR = m_CornerUL;
        m_CornerUL = cootmp;
    }

    if (m_CornerLL.u > m_CornerLR.u)
    {
        cootmp = m_CornerLR;
        m_CornerLR = m_CornerLL;
        m_CornerLL = cootmp;
    }

    if (m_CornerLL.v > m_CornerUL.v)
    {
        cootmp = m_CornerUL;
        m_CornerUL = m_CornerLL;
        m_CornerLL = cootmp;
    }

    if (m_CornerLR.v > m_CornerUR.v)
    {
        cootmp = m_CornerUR;
        m_CornerUR = m_CornerLR;
        m_CornerLR = cootmp;
    }

    return true;
}

double asGeoArea::GetUmin()
{
    return wxMin(wxMin(m_CornerUL.u, m_CornerLL.u), wxMin(m_CornerUR.u, m_CornerLR.u));
}

double asGeoArea::GetUmax()
{
    return wxMax(wxMax(m_CornerUL.u, m_CornerLL.u), wxMax(m_CornerUR.u, m_CornerLR.u));
}

double asGeoArea::GetUwidth()
{
    return abs(m_CornerUR.u-m_CornerUL.u);
}

double asGeoArea::GetVmin()
{
    return wxMin(wxMin(m_CornerUL.v, m_CornerLL.v), wxMin(m_CornerUR.v, m_CornerLR.v));
}

double asGeoArea::GetVmax()
{
    return wxMax(wxMax(m_CornerUL.v, m_CornerLL.v), wxMax(m_CornerUR.v, m_CornerLR.v));
}

double asGeoArea::GetVwidth()
{
    return abs(m_CornerUR.v-m_CornerLR.v);
}

Coo asGeoArea::GetCenter()
{
    Coo center;
    center.u = GetUmin() + (GetUmax()-GetUmin())/2;
    center.v = GetVmin() + (GetVmax()-GetVmin())/2;
    return center;
}

bool asGeoArea::IsRectangle()
{
    // Check that the area is a square
    if ((m_CornerUL.u != m_CornerLL.u) | (m_CornerUL.v != m_CornerUR.v) | (m_CornerUR.u != m_CornerLR.u) | (m_CornerLL.v != m_CornerLR.v))
    {
        return false;
    }
    return true;
}

void asGeoArea::ProjConvert(const CoordSys &newcoordsys)
{
    m_CornerUL = ProjTransform(newcoordsys, m_CornerUL);
    m_CornerUR = ProjTransform(newcoordsys, m_CornerUR);
    m_CornerLL = ProjTransform(newcoordsys, m_CornerLL);
    m_CornerLR = ProjTransform(newcoordsys, m_CornerLR);
    m_CoordSys = newcoordsys;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The coordinate system was successfully converted."));
}
