/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
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
