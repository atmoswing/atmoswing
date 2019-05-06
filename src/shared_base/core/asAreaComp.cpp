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

#include "asAreaComp.h"

asAreaComp::asAreaComp(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL, const Coo &cornerLR,
                       int flatAllowed, bool isLatLon)
        : asArea()
{
    // Set the members
    m_cornerUL = cornerUL;
    m_cornerUR = cornerUR;
    m_cornerLL = cornerLL;
    m_cornerLR = cornerLR;
    m_flatAllowed = flatAllowed;
    m_isLatLon = isLatLon;

    // Initialization and check points
    Init();
    CreateComposites();
}

asAreaComp::asAreaComp(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed, bool isLatLon)
        : asArea()
{
    // Set the members
    m_cornerUL = {xMin, yMin + yWidth};
    m_cornerUR = {xMin + xWidth, yMin + yWidth};
    m_cornerLL = {xMin, yMin};
    m_cornerLR = {xMin + xWidth, yMin};
    m_flatAllowed = flatAllowed;
    m_isLatLon = isLatLon;

    // Initialization and check points
    Init();
    CreateComposites();
}

asAreaComp::asAreaComp()
        : asArea()
{

}

void asAreaComp::Init()
{
    DoCheckPoints();
    if (!CheckConsistency())
        asThrowException(_("Unable to build a consistent area with the given coordinates."));
}

bool asAreaComp::DoCheckPoints()
{
    // Check the points and proceed to changes if necessary
    CheckPoint(m_cornerUL, asEDIT_ALLOWED);
    CheckPoint(m_cornerUR, asEDIT_ALLOWED);
    CheckPoint(m_cornerLL, asEDIT_ALLOWED);
    CheckPoint(m_cornerLR, asEDIT_ALLOWED);
    return true;
}

bool asAreaComp::CheckConsistency()
{
    // Area is a single point
    if (m_flatAllowed == asFLAT_FORBIDDEN) {
        if ((m_cornerUL.x == m_cornerUR.x) || (m_cornerLL.x == m_cornerLR.x) || (m_cornerLL.y == m_cornerUL.y) ||
            (m_cornerLR.y == m_cornerUR.y)) {
            return false;
        }
    }

    // Lon min is on the edge and should be corrected
    if ((m_cornerUL.x > m_cornerUR.x) && (m_cornerUL.x == 360)) {
        m_cornerUL.x = 0;
    }
    if ((m_cornerLL.x > m_cornerLR.x) && (m_cornerLL.x == 360)) {
        m_cornerLL.x = 0;
    }

    // Coordinates order vary
    if ((m_cornerUL.x > m_cornerUR.x) || (m_cornerLL.x > m_cornerLR.x) || (m_cornerLL.y > m_cornerUL.y) ||
        (m_cornerLR.y > m_cornerUR.y)) {
        // Do not proceed to change
        wxLogVerbose(_("The given coordinates are not increasing. This is a normal behavior if the area is on the coordinates edge."));
    }

    return true;
}

void asAreaComp::CreateComposites()
{
    m_composites.clear();

    if ((m_cornerUL.x <= m_cornerUR.x) && (m_cornerLL.x <= m_cornerLR.x) && (m_cornerLL.y <= m_cornerUL.y) &&
        (m_cornerLR.y <= m_cornerUR.y)) {
        asArea area(m_cornerUL, m_cornerUR, m_cornerLL, m_cornerLR, m_flatAllowed, m_isLatLon);
        m_composites.push_back(area);
    } else if ((m_cornerUL.x >= m_cornerUR.x) && (m_cornerLL.x >= m_cornerLR.x) && (m_cornerLL.y <= m_cornerUL.y) &&
               (m_cornerLR.y <= m_cornerUR.y) && (m_cornerLR.x == 0) && (m_cornerUR.x == 0)) {
        m_cornerLR.x = 360;
        m_cornerUR.x = 360;
        asArea area(m_cornerUL, m_cornerUR, m_cornerLL, m_cornerLR, m_flatAllowed, m_isLatLon);
        m_composites.push_back(area);
    } else if ((m_cornerUL.x >= m_cornerUR.x) && (m_cornerLL.x >= m_cornerLR.x) && (m_cornerLL.y <= m_cornerUL.y) &&
               (m_cornerLR.y <= m_cornerUR.y) && (m_cornerLR.x != 0) && (m_cornerUR.x != 0)) {
        Coo a1UL = m_cornerUL, a1UR = m_cornerUR, a1LL = m_cornerLL, a1LR = m_cornerLR;
        Coo a2UL = m_cornerUL, a2UR = m_cornerUR, a2LL = m_cornerLL, a2LR = m_cornerLR;
        a1UR.x = 360;
        a1LR.x = 360;
        a2UL.x = 0;
        a2LL.x = 0;
        asArea area1(a1UL, a1UR, a1LL, a1LR, m_flatAllowed, m_isLatLon);
        asArea area2(a2UL, a2UR, a2LL, a2LR, m_flatAllowed, m_isLatLon);
        m_composites.push_back(area1);
        m_composites.push_back(area2);
    } else {
        wxString error = "This case is not managed (asAreaComp::CreateComposites):\n ";
        error.Append(wxString::Format("m_cornerUL.x = %g\n", m_cornerUL.x));
        error.Append(wxString::Format("m_cornerUR.x = %g\n", m_cornerUR.x));
        error.Append(wxString::Format("m_cornerLL.x = %g\n", m_cornerLL.x));
        error.Append(wxString::Format("m_cornerLR.x = %g\n", m_cornerLR.x));
        error.Append(wxString::Format("m_cornerLL.y = %g\n", m_cornerLL.y));
        error.Append(wxString::Format("m_cornerUL.y = %g\n", m_cornerUL.y));
        error.Append(wxString::Format("m_cornerLR.y = %g\n", m_cornerLR.y));
        error.Append(wxString::Format("m_cornerUR.y = %g\n", m_cornerUR.y));
        wxLogError(error);
        asThrowException(_("This case is not managed."));
    }
}

double asAreaComp::GetXmin() const
{
    asThrowException(_("Not allowed on composite area."));
}

double asAreaComp::GetXmax() const
{
    asThrowException(_("Not allowed on composite area."));
}

double asAreaComp::GetYmin() const
{
    asThrowException(_("Not allowed on composite area."));
}

double asAreaComp::GetYmax() const
{
    asThrowException(_("Not allowed on composite area."));
}