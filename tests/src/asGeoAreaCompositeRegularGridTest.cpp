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

#include "asGeoAreaCompositeRegularGrid.h"
#include "gtest/gtest.h"


TEST(GeoAreaCompositeRegularGrid, ConstructorStepException)
{
    wxLogNull logNo;

    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = -10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.6;
    ASSERT_THROW(asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step),
                 asException);
}

TEST(GeoAreaCompositeRegularGrid, ConstructorOneArea)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, ConstructorTwoAreas)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = -10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, ConstructorAlternativeOneArea)
{
    double xMin = 10;
    double xWidth = 10;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, ConstructorAlternativeTwoAreas)
{
    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, CheckConsistency)
{
    double xMin = -5;
    double xWidth = 25;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, geoArea.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, geoArea.GetCornerLR().x);
}

TEST(GeoAreaCompositeRegularGrid, CheckConsistencyException)
{
    wxLogNull logNo;

    double xMin = 10;
    double xWidth = 0;
    double yMin = 40;
    double yWidth = -10;
    double step = 2.5;
    ASSERT_THROW(asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step), asException);
}

TEST(GeoAreaCompositeRegularGrid, IsRectangleTrue)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);
    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoAreaCompositeRegularGrid, IsRectangleFalse)
{
    wxLogNull logNo;

    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 15;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.5;
    ASSERT_THROW(asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step),
                 asException);
}

TEST(GeoAreaCompositeRegularGrid, GetBounds)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(20, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaCompositeRegularGrid, GetBoundsSplitted)
{
    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_DOUBLE_EQ(0, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(360, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaCompositeRegularGrid, GetCenter)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeRegularGrid, GetCenterSplitted)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -40;
    cornerUL.y = 40;
    cornerUR.x = 10;
    cornerUR.y = 40;
    cornerLL.x = -40;
    cornerLL.y = 30;
    cornerLR.x = 10;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(345, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeRegularGrid, GetCenterSplittedEdge)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -10;
    cornerUL.y = 40;
    cornerUR.x = 10;
    cornerUR.y = 40;
    cornerLL.x = -10;
    cornerLL.y = 30;
    cornerLR.x = 10;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(360, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeRegularGrid, GetCornersSplitted)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -40;
    cornerUL.y = 40;
    cornerUR.x = 10;
    cornerUR.y = 40;
    cornerLL.x = -40;
    cornerLL.y = 30;
    cornerLR.x = 10;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(0, geoArea.GetComposite(0).GetCornerUL().x);
    EXPECT_DOUBLE_EQ(40, geoArea.GetComposite(0).GetCornerUL().y);
    EXPECT_DOUBLE_EQ(10, geoArea.GetComposite(0).GetCornerUR().x);
    EXPECT_DOUBLE_EQ(40, geoArea.GetComposite(0).GetCornerUR().y);
    EXPECT_DOUBLE_EQ(0, geoArea.GetComposite(0).GetCornerLL().x);
    EXPECT_DOUBLE_EQ(30, geoArea.GetComposite(0).GetCornerLL().y);
    EXPECT_DOUBLE_EQ(10, geoArea.GetComposite(0).GetCornerLR().x);
    EXPECT_DOUBLE_EQ(30, geoArea.GetComposite(0).GetCornerLR().y);

    EXPECT_DOUBLE_EQ(320, geoArea.GetComposite(1).GetCornerUL().x);
    EXPECT_DOUBLE_EQ(40, geoArea.GetComposite(1).GetCornerUL().y);
    EXPECT_DOUBLE_EQ(360, geoArea.GetComposite(1).GetCornerUR().x);
    EXPECT_DOUBLE_EQ(40, geoArea.GetComposite(1).GetCornerUR().y);
    EXPECT_DOUBLE_EQ(320, geoArea.GetComposite(1).GetCornerLL().x);
    EXPECT_DOUBLE_EQ(30, geoArea.GetComposite(1).GetCornerLL().y);
    EXPECT_DOUBLE_EQ(360, geoArea.GetComposite(1).GetCornerLR().x);
    EXPECT_DOUBLE_EQ(30, geoArea.GetComposite(1).GetCornerLR().y);
}

TEST(GeoAreaCompositeRegularGrid, GetAxes)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -40;
    cornerUL.y = 40;
    cornerUR.x = 10;
    cornerUR.y = 40;
    cornerLL.x = -40;
    cornerLL.y = 30;
    cornerLR.x = 10;
    cornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    a1d uaxis0, vaxis0;
    uaxis0.resize(geoArea.GetXaxisCompositePtsnb(0));
    vaxis0.resize(geoArea.GetYaxisCompositePtsnb(0));

    uaxis0 = geoArea.GetXaxisComposite(0);
    vaxis0 = geoArea.GetYaxisComposite(0);

    EXPECT_DOUBLE_EQ(0, uaxis0(0));
    EXPECT_DOUBLE_EQ(2.5, uaxis0(1));
    EXPECT_DOUBLE_EQ(5, uaxis0(2));
    EXPECT_DOUBLE_EQ(7.5, uaxis0(3));
    EXPECT_DOUBLE_EQ(10, uaxis0(4));
    EXPECT_DOUBLE_EQ(30, vaxis0(0));
    EXPECT_DOUBLE_EQ(32.5, vaxis0(1));
    EXPECT_DOUBLE_EQ(40, vaxis0(4));

    a1d uaxis1, vaxis1;
    uaxis1.resize(geoArea.GetXaxisCompositePtsnb(1));
    vaxis1.resize(geoArea.GetYaxisCompositePtsnb(1));

    uaxis1 = geoArea.GetXaxisComposite(1);
    vaxis1 = geoArea.GetYaxisComposite(1);

    EXPECT_DOUBLE_EQ(320, uaxis1(0));
    EXPECT_DOUBLE_EQ(322.5, uaxis1(1));
    EXPECT_DOUBLE_EQ(325, uaxis1(2));
    EXPECT_DOUBLE_EQ(360, uaxis1(geoArea.GetXaxisCompositePtsnb(1) - 1));
    EXPECT_DOUBLE_EQ(30, vaxis1(0));
    EXPECT_DOUBLE_EQ(32.5, vaxis1(1));
    EXPECT_DOUBLE_EQ(40, vaxis1(4));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeSize)
{
    double xMin = -40;
    double xWidth = 50;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_EQ(5, geoArea.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(17, geoArea.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeSizeStepLon)
{
    double xMin = -40;
    double xWidth = 50;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_EQ(3, geoArea.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(9, geoArea.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeSizeStepLonMoved)
{
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_EQ(2, geoArea.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(2, geoArea.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeWidthStepLonMoved)
{
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_DOUBLE_EQ(7.5, geoArea.GetXaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(7.5, geoArea.GetXaxisCompositeWidth(1));
    EXPECT_DOUBLE_EQ(10, geoArea.GetYaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(10, geoArea.GetYaxisCompositeWidth(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisPtsnbStepLonMoved)
{
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_EQ(4, geoArea.GetXaxisPtsnb());
    EXPECT_EQ(5, geoArea.GetYaxisPtsnb());
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisWidthStepLonMoved)
{
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_DOUBLE_EQ(15, geoArea.GetXaxisWidth());
    EXPECT_DOUBLE_EQ(10, geoArea.GetYaxisWidth());
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeLimits)
{
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_DOUBLE_EQ(0, geoArea.GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(350, geoArea.GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, geoArea.GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, geoArea.GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(10, geoArea.GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360, geoArea.GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, geoArea.GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, geoArea.GetYaxisCompositeEnd(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeLimitsMoved)
{
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, xStep, yMin, yWidth, yStep);

    EXPECT_DOUBLE_EQ(2.5, geoArea.GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(360 - 7.5, geoArea.GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, geoArea.GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, geoArea.GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(7.5, geoArea.GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360 - 2.5, geoArea.GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, geoArea.GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, geoArea.GetYaxisCompositeEnd(1));
}

TEST(GeoAreaCompositeRegularGrid, SetLastRowAsNewComposite)
{
    double xMin = 340;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_EQ(1, geoArea.GetNbComposites());

    geoArea.SetLastRowAsNewComposite();

    EXPECT_EQ(2, geoArea.GetNbComposites());
    EXPECT_DOUBLE_EQ(0, geoArea.GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(0, geoArea.GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(340, geoArea.GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(357.5, geoArea.GetXaxisCompositeEnd(1));
}