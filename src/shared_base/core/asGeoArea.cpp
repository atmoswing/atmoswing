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

asGeoArea::asGeoArea(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level, float Height, int flatAllowed)
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

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

asGeoArea::asGeoArea(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level, float Height, int flatAllowed)
:
asGeo()
{
    if (flatAllowed==asFLAT_ALLOWED)
    {
        Ywidth = wxMax(Ywidth, 0.0);
        Xwidth = wxMax(Xwidth, 0.0);
    }
    else
    {
        wxASSERT(Ywidth>0);
        wxASSERT(Xwidth>0);
    }

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

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

asGeoArea::asGeoArea(float Level, float Height)
:
asGeo()
{
    // Set the members
    m_CornerUL.x = 0;
    m_CornerUL.y = 0;
    m_CornerUR.x = 0;
    m_CornerUR.y = 0;
    m_CornerLL.x = 0;
    m_CornerLL.y = 0;
    m_CornerLR.x = 0;
    m_CornerLR.y = 0;
    m_Level = Level;
    m_Height = Height;
    m_FlatAllowed = asFLAT_ALLOWED;
}

asGeoArea::~asGeoArea()
{
    //dtor
}

void asGeoArea::Generate(double Xmin, double Xwidth, double Ymin, double Ywidth, int flatAllowed)
{
    if (flatAllowed==asFLAT_ALLOWED)
    {
        Ywidth = wxMax(Ywidth, 0.0);
        Xwidth = wxMax(Xwidth, 0.0);
    }
    else
    {
        wxASSERT(Ywidth>0);
        wxASSERT(Xwidth>0);
    }

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
        if ((m_CornerUL.x == m_CornerUR.x) || (m_CornerLL.x == m_CornerLR.x) || (m_CornerLL.y == m_CornerUL.y) || (m_CornerLR.y == m_CornerUR.y))
        {
            return false;
        }
    }

    if (m_CornerUL.x > m_CornerUR.x)
    {
        cootmp = m_CornerUR;
        m_CornerUR = m_CornerUL;
        m_CornerUL = cootmp;
    }

    if (m_CornerLL.x > m_CornerLR.x)
    {
        cootmp = m_CornerLR;
        m_CornerLR = m_CornerLL;
        m_CornerLL = cootmp;
    }

    if (m_CornerLL.y > m_CornerUL.y)
    {
        cootmp = m_CornerUL;
        m_CornerUL = m_CornerLL;
        m_CornerLL = cootmp;
    }

    if (m_CornerLR.y > m_CornerUR.y)
    {
        cootmp = m_CornerUR;
        m_CornerUR = m_CornerLR;
        m_CornerLR = cootmp;
    }

    return true;
}

double asGeoArea::GetXmin()
{
    return wxMin(wxMin(m_CornerUL.x, m_CornerLL.x), wxMin(m_CornerUR.x, m_CornerLR.x));
}

double asGeoArea::GetXmax()
{
    return wxMax(wxMax(m_CornerUL.x, m_CornerLL.x), wxMax(m_CornerUR.x, m_CornerLR.x));
}

double asGeoArea::GetXwidth()
{
    return abs(m_CornerUR.x-m_CornerUL.x);
}

double asGeoArea::GetYmin()
{
    return wxMin(wxMin(m_CornerUL.y, m_CornerLL.y), wxMin(m_CornerUR.y, m_CornerLR.y));
}

double asGeoArea::GetYmax()
{
    return wxMax(wxMax(m_CornerUL.y, m_CornerLL.y), wxMax(m_CornerUR.y, m_CornerLR.y));
}

double asGeoArea::GetYwidth()
{
    return abs(m_CornerUR.y-m_CornerLR.y);
}

Coo asGeoArea::GetCenter()
{
    Coo center;
    center.x = GetXmin() + (GetXmax()-GetXmin())/2;
    center.y = GetYmin() + (GetYmax()-GetYmin())/2;
    return center;
}

bool asGeoArea::IsRectangle()
{
    // Check that the area is a square
    if ((m_CornerUL.x != m_CornerLL.x) | (m_CornerUL.y != m_CornerUR.y) | (m_CornerUR.x != m_CornerLR.x) | (m_CornerLL.y != m_CornerLR.y))
    {
        return false;
    }
    return true;
}
