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
#include "asGeo.h"

#include "UnitTest++.h"

namespace
{
/*
TEST(CheckPointWGS84True)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 10;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UVMaxTrue)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 360;
    Point.y = 90;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UVMinTrue)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 0;
    Point.y = -90;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UTooHigh)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 360.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(360.1, Point.x, 0.000001);
}

TEST(CheckPointWGS84UTooLow)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = -0.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-0.1, Point.x, 0.000001);
}

TEST(CheckPointWGS84VTooHigh)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 10;
    Point.y = 90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(90.1, Point.y, 0.000001);
}

TEST(CheckPointWGS84VTooLow)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 10;
    Point.y = -90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-90.1, Point.y, 0.000001);
}

TEST(CheckPointWGS84UTooHighCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 360.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(0.1, Point.x, 0.000001);
}

TEST(CheckPointWGS84UTooLowCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = -0.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(359.9, Point.x, 0.000001);
}

TEST(CheckPointWGS84VTooHighCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 10;
    Point.y = 90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(89.9, Point.y, 0.000001);
}

TEST(CheckPointWGS84VTooLowCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 10;
    Point.y = -90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-89.9, Point.y, 0.000001);
}

TEST(ProjTransformWGS84toCH1903)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 7.19;
    Point.y = 46.06;
    // http://topo.epfl.ch/misc/transcoco/transcoco_compute.php
    Coo Result = geo.ProjTransform(CH1903, Point);
    CHECK_CLOSE(580758.18, Result.x, 1);
    CHECK_CLOSE(100972.32, Result.y, 1);
}

TEST(ProjTransformWGS84toCH1903p)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.x = 7.465270;
    Point.y = 46.877096;
    Coo Result = geo.ProjTransform(CH1903p, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(2602030.74, Result.x, 1);
    CHECK_CLOSE(1191775.03, Result.y, 1);
}

TEST(ProjTransformCH1903toWGS84)
{
    asGeo geo(CH1903);
    Coo Point;
    Point.x = 603460;
    Point.y = 201200;
    Coo Result = geo.ProjTransform(WGS84, Point);
    // http://topo.epfl.ch/misc/transcoco/transcoco_compute.php
    CHECK_CLOSE(7.484091, Result.x, 0.001);
    CHECK_CLOSE(46.961871, Result.y, 0.001);
}

TEST(ProjTransformCH1903toCH1903p)
{
    asGeo geo(CH1903);
    Coo Point;
    Point.x = 602030.68;
    Point.y = 191775.03;
    Coo Result = geo.ProjTransform(CH1903p, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(2602030.74, Result.x, 1);
    CHECK_CLOSE(1191775.03, Result.y, 1);
}

TEST(ProjTransformCH1903toWGS84High)
{
    asGeo geo(CH1903);
    Coo Point;
    Point.x = 803460;
    Point.y = 291200;
    Coo Result = geo.ProjTransform(WGS84, Point);
    // http://topo.epfl.ch/misc/transcoco/transcoco_compute.php
    CHECK_CLOSE(10.151699, Result.x, 0.001);
    CHECK_CLOSE(47.739712, Result.y, 0.001);
}

TEST(ProjTransformCH1903ptoWGS84)
{
    asGeo geo(CH1903p);
    Coo Point;
    Point.x = 2602030.74;
    Point.y = 1191775.03;
    Coo Result = geo.ProjTransform(WGS84, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(7.465270, Result.x, 0.001);
    CHECK_CLOSE(46.877096, Result.y, 0.001);
}

TEST(ProjTransformCH1903ptoCH1903)
{
    asGeo geo(CH1903p);
    Coo Point;
    Point.x = 2602030.74;
    Point.y = 1191775.03;
    Coo Result = geo.ProjTransform(CH1903, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(602030.68, Result.x, 1);
    CHECK_CLOSE(191775.03, Result.y, 1);
}

TEST(ProjTransformCH1903ptoCH1903pSame)
{
    asGeo geo(CH1903p);
    Coo Point;
    Point.x = 2803460;
    Point.y = 1291200;
    Coo Result = geo.ProjTransform(CH1903p, Point);
    CHECK_CLOSE(2803460, Result.x, 1);
    CHECK_CLOSE(1291200, Result.y, 1);
}
*/
}
