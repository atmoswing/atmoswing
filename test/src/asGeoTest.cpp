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

TEST(CheckPointWGS84True)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UVMaxTrue)
{
    asGeo geo;
    Coo Point;
    Point.x = 360;
    Point.y = 90;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UVMinTrue)
{
    asGeo geo;
    Coo Point;
    Point.x = 0;
    Point.y = -90;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UTooHigh)
{
    asGeo geo;
    Coo Point;
    Point.x = 360.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(360.1, Point.x, 0.000001);
}

TEST(CheckPointWGS84UTooLow)
{
    asGeo geo;
    Coo Point;
    Point.x = -0.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-0.1, Point.x, 0.000001);
}

TEST(CheckPointWGS84VTooHigh)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = 90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(90.1, Point.y, 0.000001);
}

TEST(CheckPointWGS84VTooLow)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = -90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-90.1, Point.y, 0.000001);
}

TEST(CheckPointWGS84UTooHighCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = 360.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(0.1, Point.x, 0.000001);
}

TEST(CheckPointWGS84UTooLowCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = -0.1;
    Point.y = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(359.9, Point.x, 0.000001);
}

TEST(CheckPointWGS84VTooHighCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = 90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(89.9, Point.y, 0.000001);
}

TEST(CheckPointWGS84VTooLowCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = -90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-89.9, Point.y, 0.000001);
}

}
