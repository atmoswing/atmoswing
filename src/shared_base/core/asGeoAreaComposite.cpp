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

#include "asGeoAreaComposite.h"

asGeoAreaComposite::asGeoAreaComposite(const Coo &cornerUL, const Coo &cornerUR, const Coo &cornerLL,
                                       const Coo &cornerLR, float level, float height, int flatAllowed)
        : asGeo(Regular),
          m_level(level),
          m_height(height),
          m_cornerUL(cornerUL),
          m_cornerUR(cornerUR),
          m_cornerLL(cornerLL),
          m_cornerLR(cornerLR),
          m_flatAllowed(flatAllowed),
          m_absoluteXmin(m_cornerUL.x),
          m_absoluteXmax(m_cornerUR.x),
          m_absoluteYmin(m_cornerLL.y),
          m_absoluteYmax(m_cornerUL.y)
{
    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully created."));
}

asGeoAreaComposite::asGeoAreaComposite(double xMin, double xWidth, double yMin, double yWidth, float level,
                                       float height, int flatAllowed)
        : asGeo(Regular),
          m_level(level),
          m_height(height),
          m_cornerUL({xMin, yMin + yWidth}),
          m_cornerUR({xMin + xWidth, yMin + yWidth}),
          m_cornerLL({xMin, yMin}),
          m_cornerLR({xMin + xWidth, yMin}),
          m_flatAllowed(flatAllowed),
          m_absoluteXmin(m_cornerUL.x),
          m_absoluteXmax(m_cornerUR.x),
          m_absoluteYmin(m_cornerLL.y),
          m_absoluteYmax(m_cornerUL.y)
{
    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully created."));
}

asGeoAreaComposite::asGeoAreaComposite(float level, float height)
        : asGeo(Regular),
          m_level(level),
          m_height(height),
          m_cornerUL({0, 0}),
          m_cornerUR({0, 0}),
          m_cornerLL({0, 0}),
          m_cornerLR({0, 0}),
          m_flatAllowed(asFLAT_ALLOWED),
          m_absoluteXmin(0),
          m_absoluteXmax(0),
          m_absoluteYmin(0),
          m_absoluteYmax(0)
{

}

void asGeoAreaComposite::Generate(double xMin, double xWidth, double yMin, double yWidth, int flatAllowed)
{
    m_cornerUL.x = xMin;
    m_cornerUL.y = yMin + yWidth;
    m_cornerUR.x = xMin + xWidth;
    m_cornerUR.y = yMin + yWidth;
    m_cornerLL.x = xMin;
    m_cornerLL.y = yMin;
    m_cornerLR.x = xMin + xWidth;
    m_cornerLR.y = yMin;
    m_flatAllowed = flatAllowed;
    m_absoluteXmin = m_cornerUL.x;
    m_absoluteXmax = m_cornerUR.x;
    m_absoluteYmin = m_cornerLL.y;
    m_absoluteYmax = m_cornerUL.y;

    // Initialization and check points
    Init();
    CreateComposites();

    wxLogVerbose(_("The composite area was successfully generated."));
}

void asGeoAreaComposite::Init()
{
    DoCheckPoints();
    if (!CheckConsistency())
        asThrowException(_("Unable to build a consistent area with the given coordinates."));
}

bool asGeoAreaComposite::DoCheckPoints()
{
    // Check the points and proceed to changes if necessary
    CheckPoint(m_cornerUL, asEDIT_ALLOWED);
    CheckPoint(m_cornerUR, asEDIT_ALLOWED);
    CheckPoint(m_cornerLL, asEDIT_ALLOWED);
    CheckPoint(m_cornerLR, asEDIT_ALLOWED);
    return true;
}

bool asGeoAreaComposite::CheckConsistency()
{
    // Area is a single point
    if (m_flatAllowed == asFLAT_FORBIDDEN) {
        if ((m_cornerUL.x == m_cornerUR.x) || (m_cornerLL.x == m_cornerLR.x) || (m_cornerLL.y == m_cornerUL.y) ||
            (m_cornerLR.y == m_cornerUR.y)) {
            return false;
        }
    }

    // Lon min is on the edge and should be corrected
    if ((m_cornerUL.x > m_cornerUR.x) && (m_cornerUL.x == m_axisXmax)) {
        m_cornerUL.x -= m_axisXmax;
    }
    if ((m_cornerLL.x > m_cornerLR.x) && (m_cornerLL.x == m_axisXmax)) {
        m_cornerLL.x -= m_axisXmax;
    }

    // Coordinates order vary
    if ((m_cornerUL.x > m_cornerUR.x) || (m_cornerLL.x > m_cornerLR.x) || (m_cornerLL.y > m_cornerUL.y) ||
        (m_cornerLR.y > m_cornerUR.y)) {
        // Do not proceed to change
        wxLogVerbose(_("The given coordinates are not increasing. This is a normal behavior if the area is on the coordinates edge."));
    }

    return true;
}

double asGeoAreaComposite::GetXmin() const
{
    double realXmin = Infd;
    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        realXmin = wxMin(realXmin, m_composites[iArea].GetXmin());
    }
    return realXmin;
}

double asGeoAreaComposite::GetXmax() const
{
    double realXmax = -Infd;
    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        realXmax = wxMax(realXmax, m_composites[iArea].GetXmax());
    }
    return realXmax;
}

double asGeoAreaComposite::GetYmin() const
{
    double realYmin = Infd;
    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        realYmin = wxMin(realYmin, m_composites[iArea].GetYmin());
    }
    return realYmin;
}

double asGeoAreaComposite::GetYmax() const
{
    double realYmax = -Infd;
    for (int iArea = 0; iArea < GetNbComposites(); iArea++) {
        realYmax = wxMax(realYmax, m_composites[iArea].GetYmax());
    }
    return realYmax;
}

Coo asGeoAreaComposite::GetCenter() const
{
    Coo center;

    if ((m_cornerUL.x < m_cornerUR.x) & (m_cornerLL.x < m_cornerLR.x) & (m_cornerLL.y < m_cornerUL.y) &
        (m_cornerLR.y < m_cornerUR.y)) {
        center = m_composites[0].GetCenter();
    } else if ((m_cornerUL.x > m_cornerUR.x) & (m_cornerLL.x > m_cornerLR.x) & (m_cornerLL.y < m_cornerUL.y) &
               (m_cornerLR.y < m_cornerUR.y)) {
        double CornerUR = 360 + m_cornerUR.x;
        double CornerLR = 360 + m_cornerLR.x;
        double xMin = wxMin(m_cornerUL.x, m_cornerLL.x);
        double xMax = wxMin(CornerUR, CornerLR);
        center.x = xMin + (xMax - xMin) / 2;
        center.y = GetYmin() + (GetYmax() - GetYmin()) / 2;
        return center;
    } else {
        asThrowException(_("This case is not managed yet."));
    }


    center.x = GetXmin() + (GetXmax() - GetXmin()) / 2;
    center.y = GetYmin() + (GetYmax() - GetYmin()) / 2;
    return center;
}

bool asGeoAreaComposite::IsRectangle() const
{
    // Check that the area is a square
    return !((m_cornerUL.x != m_cornerLL.x) | (m_cornerUL.y != m_cornerUR.y) | (m_cornerUR.x != m_cornerLR.x) |
             (m_cornerLL.y != m_cornerLR.y));
}

void asGeoAreaComposite::CreateComposites()
{
    m_composites.clear();

    if ((m_cornerUL.x <= m_cornerUR.x) & (m_cornerLL.x <= m_cornerLR.x) & (m_cornerLL.y <= m_cornerUL.y) &
        (m_cornerLR.y <= m_cornerUR.y)) {
        asGeoArea area(m_cornerUL, m_cornerUR, m_cornerLL, m_cornerLR, m_level, m_height, m_flatAllowed);
        m_composites.push_back(area);
    } else if ((m_cornerUL.x >= m_cornerUR.x) & (m_cornerLL.x >= m_cornerLR.x) & (m_cornerLL.y <= m_cornerUL.y) &
               (m_cornerLR.y <= m_cornerUR.y) & (m_cornerLR.x == m_axisXmin) & (m_cornerUR.x == m_axisXmin)) {
        m_cornerLR.x = m_axisXmax;
        m_cornerUR.x = m_axisXmax;
        asGeoArea area(m_cornerUL, m_cornerUR, m_cornerLL, m_cornerLR, m_level, m_height, m_flatAllowed);
        m_composites.push_back(area);
    } else if ((m_cornerUL.x >= m_cornerUR.x) & (m_cornerLL.x >= m_cornerLR.x) & (m_cornerLL.y <= m_cornerUL.y) &
               (m_cornerLR.y <= m_cornerUR.y) & (m_cornerLR.x != m_axisXmin) & (m_cornerUR.x != m_axisXmin)) {
        Coo a1UL = m_cornerUL, a1UR = m_cornerUR, a1LL = m_cornerLL, a1LR = m_cornerLR;
        Coo a2UL = m_cornerUL, a2UR = m_cornerUR, a2LL = m_cornerLL, a2LR = m_cornerLR;
        a1UL.x = m_axisXmin;
        a1LL.x = m_axisXmin;
        a2UR.x = m_axisXmax;
        a2LR.x = m_axisXmax;
        asGeoArea area1(a1UL, a1UR, a1LL, a1LR, m_level, m_height, m_flatAllowed);
        asGeoArea area2(a2UL, a2UR, a2LL, a2LR, m_level, m_height, m_flatAllowed);
        m_composites.push_back(area1);
        m_composites.push_back(area2);
    } else {
        wxString error = "This case is not managed yet (asGeoAreaComposite::CreateComposites):\n ";
        error.Append(wxString::Format("m_cornerUL.x = %g\n", m_cornerUL.x));
        error.Append(wxString::Format("m_cornerUR.x = %g\n", m_cornerUR.x));
        error.Append(wxString::Format("m_cornerLL.x = %g\n", m_cornerLL.x));
        error.Append(wxString::Format("m_cornerLR.x = %g\n", m_cornerLR.x));
        error.Append(wxString::Format("m_cornerLL.y = %g\n", m_cornerLL.y));
        error.Append(wxString::Format("m_cornerUL.y = %g\n", m_cornerUL.y));
        error.Append(wxString::Format("m_cornerLR.y = %g\n", m_cornerLR.y));
        error.Append(wxString::Format("m_cornerUR.y = %g\n", m_cornerUR.y));
        wxLogError(error);
        asThrowException(_("This case is not managed yet."));
    }
}
