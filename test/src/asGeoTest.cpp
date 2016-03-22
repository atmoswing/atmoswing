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

#include "asGeo.h"
#include "gtest/gtest.h"


TEST(Geo, CheckPointWGS84True)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = 10;
    EXPECT_TRUE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
}

TEST(Geo, CheckPointWGS84UVMaxTrue)
{
    asGeo geo;
    Coo Point;
    Point.x = 360;
    Point.y = 90;
    EXPECT_TRUE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
}

TEST(Geo, CheckPointWGS84UVMinTrue)
{
    asGeo geo;
    Coo Point;
    Point.x = 0;
    Point.y = -90;
    EXPECT_TRUE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
}

TEST(Geo, CheckPointWGS84UTooHigh)
{
    asGeo geo;
    Coo Point;
    Point.x = 360.1;
    Point.y = 10;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
    EXPECT_DOUBLE_EQ(360.1, Point.x);
}

TEST(Geo, CheckPointWGS84UTooLow)
{
    asGeo geo;
    Coo Point;
    Point.x = -0.1;
    Point.y = 10;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
    EXPECT_DOUBLE_EQ(-0.1, Point.x);
}

TEST(Geo, CheckPointWGS84VTooHigh)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = 90.1;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
    EXPECT_DOUBLE_EQ(90.1, Point.y);
}

TEST(Geo, CheckPointWGS84VTooLow)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = -90.1;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_FORBIDEN));
    EXPECT_DOUBLE_EQ(-90.1, Point.y);
}

TEST(Geo, CheckPointWGS84UTooHighCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = 360.1;
    Point.y = 10;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_ALLOWED));
    EXPECT_FLOAT_EQ(0.1, Point.x);
}

TEST(Geo, CheckPointWGS84UTooLowCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = -0.1;
    Point.y = 10;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_ALLOWED));
    EXPECT_DOUBLE_EQ(359.9, Point.x);
}

TEST(Geo, CheckPointWGS84VTooHighCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = 90.1;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_ALLOWED));
    EXPECT_DOUBLE_EQ(89.9, Point.y);
}

TEST(Geo, CheckPointWGS84VTooLowCorr)
{
    asGeo geo;
    Coo Point;
    Point.x = 10;
    Point.y = -90.1;
    EXPECT_FALSE(geo.CheckPoint(Point, asEDIT_ALLOWED));
    EXPECT_DOUBLE_EQ(-89.9, Point.y);
}
