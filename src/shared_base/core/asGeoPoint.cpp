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
 
#include "asGeoPoint.h"

asGeoPoint::asGeoPoint(CoordSys coosys, const Coo &Point, float Level, float Height)
:
asGeo(coosys)
{
    // Set the members
    m_CoordSys = coosys;
    m_Point = Point;
    m_Level = Level;
    m_Height = Height;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The point was successfully created."));
}

asGeoPoint::asGeoPoint(CoordSys coosys, double U, double V, float Level, float Height)
:
asGeo(coosys)
{
    // Set the members
    m_CoordSys = coosys;
    m_Point.u = U;
    m_Point.v = V;
    m_Level = Level;
    m_Height = Height;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The point was successfully created."));
}

asGeoPoint::~asGeoPoint()
{
    //dtor
}

void asGeoPoint::Init()
{
    InitBounds();
    DoCheckPoints();
}

bool asGeoPoint::DoCheckPoints()
{
    // Check the point and proceed to changes if necessary
    CheckPoint(m_Point, asEDIT_ALLOWED);
    return true;
}

void asGeoPoint::ProjConvert(CoordSys newcoordsys)
{
    m_Point = ProjTransform(newcoordsys, m_Point);
    m_CoordSys = newcoordsys;

    // Initialization and check points
    Init();

    wxLogVerbose(_("The coordinate system was successfully converted."));
}
