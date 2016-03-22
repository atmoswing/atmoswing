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
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    double step = 2.6;
    ASSERT_THROW(asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
}

TEST(GeoAreaCompositeRegularGrid, ConstructorOneArea)
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
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, ConstructorTwoAreas)
{
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
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, ConstructorAlternativeOneArea)
{
    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, ConstructorAlternativeTwoAreas)
{
    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeRegularGrid, CheckConsistency)
{
    double Xmin = -5;
    double Xwidth = 25;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, geoArea.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, geoArea.GetCornerLR().x);
}

TEST(GeoAreaCompositeRegularGrid, CheckConsistencyException)
{
    double Xmin = 10;
    double Xwidth = 0;
    double Ymin = 40;
    double Ywidth = -10;
    double step = 2.5;
    ASSERT_THROW(asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
}

TEST(GeoAreaCompositeRegularGrid, IsRectangleTrue)
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
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);
    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoAreaCompositeRegularGrid, IsRectangleFalse)
{
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
    ASSERT_THROW(asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
}

TEST(GeoAreaCompositeRegularGrid, GetBounds)
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
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(20, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaCompositeRegularGrid, GetBoundsSplitted)
{
    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    EXPECT_DOUBLE_EQ(0, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(360, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaCompositeRegularGrid, GetCenter)
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
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeRegularGrid, GetCenterSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(345, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeRegularGrid, GetCenterSplittedEdge)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(360, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeRegularGrid, GetCornersSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

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
/*
TEST(GeoAreaCompositeRegularGrid, IsOnGridTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_TRUE(geoArea.IsOnGrid(2.5));
}

TEST(GeoAreaCompositeRegularGrid, IsOnGridTrueTwoAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_TRUE(geoArea.IsOnGrid(2.5, 5));
}

TEST(GeoAreaCompositeRegularGrid, IsOnGridFalseStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_FALSE(geoArea.IsOnGrid(6));
}

TEST(GeoAreaCompositeRegularGrid, IsOnGridFalseSecondStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    EXPECT_FALSE(geoArea.IsOnGrid(5, 6));
}
*/
TEST(GeoAreaCompositeRegularGrid, GetAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Array1DDouble uaxis0, vaxis0;
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

    Array1DDouble uaxis1, vaxis1;
    uaxis1.resize(geoArea.GetXaxisCompositePtsnb(1));
    vaxis1.resize(geoArea.GetYaxisCompositePtsnb(1));

    uaxis1 = geoArea.GetXaxisComposite(1);
    vaxis1 = geoArea.GetYaxisComposite(1);

    EXPECT_DOUBLE_EQ(320, uaxis1(0));
    EXPECT_DOUBLE_EQ(322.5, uaxis1(1));
    EXPECT_DOUBLE_EQ(325, uaxis1(2));
    EXPECT_DOUBLE_EQ(360, uaxis1(geoArea.GetXaxisCompositePtsnb(1)-1));
    EXPECT_DOUBLE_EQ(30, vaxis1(0));
    EXPECT_DOUBLE_EQ(32.5, vaxis1(1));
    EXPECT_DOUBLE_EQ(40, vaxis1(4));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeSize)
{
    double Xmin = -40;
    double Xwidth = 50;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    EXPECT_EQ(5, geoArea.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(17, geoArea.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeSizeStepLon)
{
    double Xmin = -40;
    double Xwidth = 50;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    EXPECT_EQ(3, geoArea.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(9, geoArea.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeSizeStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    EXPECT_EQ(2, geoArea.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(2, geoArea.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea.GetYaxisCompositePtsnb(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeWidthStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    EXPECT_DOUBLE_EQ(7.5, geoArea.GetXaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(7.5, geoArea.GetXaxisCompositeWidth(1));
    EXPECT_DOUBLE_EQ(10, geoArea.GetYaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(10, geoArea.GetYaxisCompositeWidth(1));
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisPtsnbStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    EXPECT_EQ(4, geoArea.GetXaxisPtsnb());
    EXPECT_EQ(5, geoArea.GetYaxisPtsnb());
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisWidthStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    EXPECT_DOUBLE_EQ(15, geoArea.GetXaxisWidth());
    EXPECT_DOUBLE_EQ(10, geoArea.GetYaxisWidth());
}

TEST(GeoAreaCompositeRegularGrid, GetUYaxisCompositeLimits)
{
    double Xmin = -10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

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
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoArea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    EXPECT_DOUBLE_EQ(2.5, geoArea.GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(360-7.5, geoArea.GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, geoArea.GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, geoArea.GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(7.5, geoArea.GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360-2.5, geoArea.GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, geoArea.GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, geoArea.GetYaxisCompositeEnd(1));
}
