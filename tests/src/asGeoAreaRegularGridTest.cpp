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

#include "asGeoAreaRegularGrid.h"
#include "gtest/gtest.h"


TEST(GeoAreaRegularGrid, ConstructorLimitsException)
{
    wxLogNull logNo;

    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    double step = 2.5;

    ASSERT_THROW(asGeoAreaRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
}

TEST(GeoAreaRegularGrid, ConstructorAlternativeLimitsException)
{
    wxLogNull logNo;

    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;

    ASSERT_THROW(asGeoAreaRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
}

TEST(GeoAreaRegularGrid, ConstructorStepException)
{
    wxLogNull logNo;

    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.7;

    ASSERT_THROW(asGeoAreaRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
}

TEST(GeoAreaRegularGrid, IsRectangleTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoAreaRegularGrid, IsRectangleFalse)
{
    wxLogNull logNo;

    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 15;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    double step = 2.5;

    ASSERT_THROW(asGeoAreaRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
}

TEST(GeoAreaRegularGrid, GetBounds)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(20, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaRegularGrid, GetCenter)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaRegularGrid, GetAxes)
{
    double Xmin = 5;
    double Xwidth = 20;
    double Ymin = 45;
    double Ywidth = 2.5;
    double step = 2.5;
    asGeoAreaRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    a1d uaxis;
    uaxis.resize(geoArea.GetXaxisPtsnb());
    uaxis = geoArea.GetXaxis();

    a1d vaxis;
    vaxis.resize(geoArea.GetYaxisPtsnb());
    vaxis = geoArea.GetYaxis();

    EXPECT_DOUBLE_EQ(5, uaxis[0]);
    EXPECT_DOUBLE_EQ(7.5, uaxis[1]);
    EXPECT_DOUBLE_EQ(10, uaxis[2]);
    EXPECT_DOUBLE_EQ(15, uaxis[4]);
    EXPECT_DOUBLE_EQ(45, vaxis[0]);
    EXPECT_DOUBLE_EQ(47.5, vaxis[1]);
}
