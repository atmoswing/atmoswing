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

#include "asGeoArea.h"

asGeoArea::asGeoArea(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level,
                     float Height, int flatAllowed)
        : asGeo()
{
    // Set the members
    m_cornerUL = CornerUL;
    m_cornerUR = CornerUR;
    m_cornerLL = CornerLL;
    m_cornerLR = CornerLR;
    m_level = Level;
    m_height = Height;
    m_flatAllowed = flatAllowed;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

asGeoArea::asGeoArea(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level, float Height, int flatAllowed)
        : asGeo()
{
    if (flatAllowed == asFLAT_ALLOWED) {
        Ywidth = wxMax(Ywidth, 0.0);
        Xwidth = wxMax(Xwidth, 0.0);
    } else {
        wxASSERT(Ywidth > 0);
        wxASSERT(Xwidth > 0);
    }

    // Set the members
    m_cornerUL.x = Xmin;
    m_cornerUL.y = Ymin + Ywidth;
    m_cornerUR.x = Xmin + Xwidth;
    m_cornerUR.y = Ymin + Ywidth;
    m_cornerLL.x = Xmin;
    m_cornerLL.y = Ymin;
    m_cornerLR.x = Xmin + Xwidth;
    m_cornerLR.y = Ymin;
    m_level = Level;
    m_height = Height;
    m_flatAllowed = flatAllowed;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

asGeoArea::asGeoArea(float Level, float Height)
        : asGeo()
{
    // Set the members
    m_cornerUL.x = 0;
    m_cornerUL.y = 0;
    m_cornerUR.x = 0;
    m_cornerUR.y = 0;
    m_cornerLL.x = 0;
    m_cornerLL.y = 0;
    m_cornerLR.x = 0;
    m_cornerLR.y = 0;
    m_level = Level;
    m_height = Height;
    m_flatAllowed = asFLAT_ALLOWED;
}

asGeoArea::~asGeoArea()
{
    //dtor
}

void asGeoArea::Generate(double Xmin, double Xwidth, double Ymin, double Ywidth, int flatAllowed)
{
    if (flatAllowed == asFLAT_ALLOWED) {
        Ywidth = wxMax(Ywidth, 0.0);
        Xwidth = wxMax(Xwidth, 0.0);
    } else {
        wxASSERT(Ywidth > 0);
        wxASSERT(Xwidth > 0);
    }

    // Set the members
    m_cornerUL.x = Xmin;
    m_cornerUL.y = Ymin + Ywidth;
    m_cornerUR.x = Xmin + Xwidth;
    m_cornerUR.y = Ymin + Ywidth;
    m_cornerLL.x = Xmin;
    m_cornerLL.y = Ymin;
    m_cornerLR.x = Xmin + Xwidth;
    m_cornerLR.y = Ymin;
    m_flatAllowed = flatAllowed;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The area was successfully created."));
}

void asGeoArea::Init()
{
    InitBounds();
    if (!DoCheckPoints())
        asThrowException(_("Use asGeoAreaComposite in this case."));
    if (!CheckConsistency())
        asThrowException(_("Unable to build a consistent area with the given coordinates."));
}

bool asGeoArea::DoCheckPoints()
{
    return !(!CheckPoint(m_cornerUL, asEDIT_FORBIDEN) || !CheckPoint(m_cornerUR, asEDIT_FORBIDEN) ||
             !CheckPoint(m_cornerLL, asEDIT_FORBIDEN) || !CheckPoint(m_cornerLR, asEDIT_FORBIDEN));
}

bool asGeoArea::CheckConsistency()
{
    Coo cootmp;

    if (m_flatAllowed == asFLAT_FORBIDDEN) {
        if ((m_cornerUL.x == m_cornerUR.x) || (m_cornerLL.x == m_cornerLR.x) || (m_cornerLL.y == m_cornerUL.y) ||
            (m_cornerLR.y == m_cornerUR.y)) {
            return false;
        }
    }

    if (m_cornerUL.x > m_cornerUR.x) {
        cootmp = m_cornerUR;
        m_cornerUR = m_cornerUL;
        m_cornerUL = cootmp;
    }

    if (m_cornerLL.x > m_cornerLR.x) {
        cootmp = m_cornerLR;
        m_cornerLR = m_cornerLL;
        m_cornerLL = cootmp;
    }

    if (m_cornerLL.y > m_cornerUL.y) {
        cootmp = m_cornerUL;
        m_cornerUL = m_cornerLL;
        m_cornerLL = cootmp;
    }

    if (m_cornerLR.y > m_cornerUR.y) {
        cootmp = m_cornerUR;
        m_cornerUR = m_cornerLR;
        m_cornerLR = cootmp;
    }

    return true;
}

double asGeoArea::GetXmin() const
{
    return wxMin(wxMin(m_cornerUL.x, m_cornerLL.x), wxMin(m_cornerUR.x, m_cornerLR.x));
}

double asGeoArea::GetXmax() const
{
    return wxMax(wxMax(m_cornerUL.x, m_cornerLL.x), wxMax(m_cornerUR.x, m_cornerLR.x));
}

double asGeoArea::GetXwidth() const
{
    return std::abs(m_cornerUR.x - m_cornerUL.x);
}

double asGeoArea::GetYmin() const
{
    return wxMin(wxMin(m_cornerUL.y, m_cornerLL.y), wxMin(m_cornerUR.y, m_cornerLR.y));
}

double asGeoArea::GetYmax() const
{
    return wxMax(wxMax(m_cornerUL.y, m_cornerLL.y), wxMax(m_cornerUR.y, m_cornerLR.y));
}

double asGeoArea::GetYwidth() const
{
    return std::abs(m_cornerUR.y - m_cornerLR.y);
}

Coo asGeoArea::GetCenter() const
{
    Coo center;
    center.x = GetXmin() + (GetXmax() - GetXmin()) / 2;
    center.y = GetYmin() + (GetYmax() - GetYmin()) / 2;
    return center;
}

bool asGeoArea::IsRectangle() const
{
    // Check that the area is a square
    return !((m_cornerUL.x != m_cornerLL.x) | (m_cornerUL.y != m_cornerUR.y) | (m_cornerUR.x != m_cornerLR.x) |
             (m_cornerLL.y != m_cornerLR.y));
}
