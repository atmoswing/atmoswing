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

#include "include_tests.h"
#include "asGeoPoint.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorDefault)
{
	wxString str("Testing geo points management...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());
	
    Coo Point;
    Point.u = 7;
    Point.v = 46;
    asGeoPoint geopoint(WGS84, Point);

    CHECK_CLOSE(7, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ConstructorOther)
{
    double U = 7;
    double V = 46;
    asGeoPoint geopoint(WGS84, U, V);

    CHECK_CLOSE(7, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ConstructorOutBoundsLon)
{
    double U = -10;
    double V = 46;
    asGeoPoint geopoint(WGS84, U, V);

    CHECK_CLOSE(350, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ConstructorOutBoundsLat)
{
    double U = 10;
    double V = -100;
    asGeoPoint geopoint(WGS84, U, V);

    CHECK_CLOSE(190, geopoint.GetU(), 0.001);
    CHECK_CLOSE(-80, geopoint.GetV(), 0.001);
}

TEST(SetCooOutBounds)
{
    asGeoPoint geopoint(WGS84, 0, 0);
    Coo Point;
    Point.u = -10;
    Point.v = 46;
    geopoint.SetCoo(Point);

    CHECK_CLOSE(350, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ProjConvert)
{
    double U = 10;
    double V = 48;
    asGeoPoint geopoint(WGS84, U, V);
    geopoint.ProjConvert(CH1903);

    CHECK_CLOSE(791142.61, geopoint.GetU(), 2);
    CHECK_CLOSE(319746.83, geopoint.GetV(), 2);
}

}
