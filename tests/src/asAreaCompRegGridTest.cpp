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

#include "asAreaCompRegGrid.h"
#include "gtest/gtest.h"

TEST(AreaCompRegGrid, ConstructorOneArea) {
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
    asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_EQ(1, area.GetNbComposites());
}

TEST(AreaCompRegGrid, ConstructorTwoAreas) {
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
    asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_EQ(2, area.GetNbComposites());
}

TEST(AreaCompRegGrid, ConstructorAlternativeOneArea) {
    double xMin = 10;
    double xWidth = 10;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_EQ(1, area.GetNbComposites());
}

TEST(AreaCompRegGrid, ConstructorAlternativeTwoAreas) {
    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_EQ(2, area.GetNbComposites());
}

TEST(AreaCompRegGrid, CheckConsistency) {
    double xMin = -5;
    double xWidth = 25;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_DOUBLE_EQ(355, area.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, area.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, area.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, area.GetCornerLR().x);
}

TEST(AreaCompRegGrid, CheckConsistencyException) {
    wxLogNull logNo;

    double xMin = 10;
    double xWidth = 0;
    double yMin = 40;
    double yWidth = -10;
    double step = 2.5;
    ASSERT_THROW(asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step), std::exception);
}

TEST(AreaCompRegGrid, IsRectangleTrue) {
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
    asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);
    EXPECT_TRUE(area.IsRectangle());
}

TEST(AreaCompRegGrid, IsRectangleFalse) {
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
    ASSERT_THROW(asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step), std::exception);
}

TEST(AreaCompRegGrid, GetBounds) {
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
    asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(10, area.GetComposite(0).GetXmin());
    EXPECT_DOUBLE_EQ(30, area.GetComposite(0).GetYmin());
    EXPECT_DOUBLE_EQ(20, area.GetComposite(0).GetXmax());
    EXPECT_DOUBLE_EQ(40, area.GetComposite(0).GetYmax());
}

TEST(AreaCompRegGrid, GetBoundsSplitted) {
    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_DOUBLE_EQ(350, area.GetComposite(0).GetXmin());
    EXPECT_DOUBLE_EQ(360, area.GetComposite(0).GetXmax());
    EXPECT_DOUBLE_EQ(0, area.GetComposite(1).GetXmin());
    EXPECT_DOUBLE_EQ(20, area.GetComposite(1).GetXmax());
    EXPECT_DOUBLE_EQ(30, area.GetComposite(0).GetYmin());
    EXPECT_DOUBLE_EQ(40, area.GetComposite(0).GetYmax());
}

TEST(AreaCompRegGrid, GetCornersSplitted) {
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
    asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(320, area.GetComposite(0).GetCornerUL().x);
    EXPECT_DOUBLE_EQ(40, area.GetComposite(0).GetCornerUL().y);
    EXPECT_DOUBLE_EQ(360, area.GetComposite(0).GetCornerUR().x);
    EXPECT_DOUBLE_EQ(40, area.GetComposite(0).GetCornerUR().y);
    EXPECT_DOUBLE_EQ(320, area.GetComposite(0).GetCornerLL().x);
    EXPECT_DOUBLE_EQ(30, area.GetComposite(0).GetCornerLL().y);
    EXPECT_DOUBLE_EQ(360, area.GetComposite(0).GetCornerLR().x);
    EXPECT_DOUBLE_EQ(30, area.GetComposite(0).GetCornerLR().y);

    EXPECT_DOUBLE_EQ(0, area.GetComposite(1).GetCornerUL().x);
    EXPECT_DOUBLE_EQ(40, area.GetComposite(1).GetCornerUL().y);
    EXPECT_DOUBLE_EQ(10, area.GetComposite(1).GetCornerUR().x);
    EXPECT_DOUBLE_EQ(40, area.GetComposite(1).GetCornerUR().y);
    EXPECT_DOUBLE_EQ(0, area.GetComposite(1).GetCornerLL().x);
    EXPECT_DOUBLE_EQ(30, area.GetComposite(1).GetCornerLL().y);
    EXPECT_DOUBLE_EQ(10, area.GetComposite(1).GetCornerLR().x);
    EXPECT_DOUBLE_EQ(30, area.GetComposite(1).GetCornerLR().y);
}

TEST(AreaCompRegGrid, GetAxes) {
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
    asAreaCompRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    a1d uaxis0, vaxis0;
    uaxis0.resize(area.GetXaxisCompositePtsnb(0));
    vaxis0.resize(area.GetYaxisCompositePtsnb(0));

    uaxis0 = area.GetXaxisComposite(0);
    vaxis0 = area.GetYaxisComposite(0);

    EXPECT_DOUBLE_EQ(320, uaxis0(0));
    EXPECT_DOUBLE_EQ(322.5, uaxis0(1));
    EXPECT_DOUBLE_EQ(325, uaxis0(2));
    EXPECT_DOUBLE_EQ(360, uaxis0(area.GetXaxisCompositePtsnb(0) - 1));
    EXPECT_DOUBLE_EQ(30, vaxis0(0));
    EXPECT_DOUBLE_EQ(32.5, vaxis0(1));
    EXPECT_DOUBLE_EQ(40, vaxis0(4));

    a1d uaxis1, vaxis1;
    uaxis1.resize(area.GetXaxisCompositePtsnb(1));
    vaxis1.resize(area.GetYaxisCompositePtsnb(1));

    uaxis1 = area.GetXaxisComposite(1);
    vaxis1 = area.GetYaxisComposite(1);

    EXPECT_DOUBLE_EQ(0, uaxis1(0));
    EXPECT_DOUBLE_EQ(2.5, uaxis1(1));
    EXPECT_DOUBLE_EQ(5, uaxis1(2));
    EXPECT_DOUBLE_EQ(7.5, uaxis1(3));
    EXPECT_DOUBLE_EQ(10, uaxis1(4));
    EXPECT_DOUBLE_EQ(30, vaxis1(0));
    EXPECT_DOUBLE_EQ(32.5, vaxis1(1));
    EXPECT_DOUBLE_EQ(40, vaxis1(4));
}

TEST(AreaCompRegGrid, GetUYaxisCompositeSize) {
    double xMin = -40;
    double xWidth = 50;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_EQ(17, area.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(5, area.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area.GetYaxisCompositePtsnb(1));
}

TEST(AreaCompRegGrid, GetUYaxisCompositeSizeStepLon) {
    double xMin = -40;
    double xWidth = 50;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_EQ(9, area.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(2, area.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area.GetYaxisCompositePtsnb(1));
}

TEST(AreaCompRegGrid, GetUYaxisCompositeSizeStepLonMoved) {
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_EQ(2, area.GetXaxisCompositePtsnb(0));
    EXPECT_EQ(2, area.GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area.GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area.GetYaxisCompositePtsnb(1));
}

TEST(AreaCompRegGrid, GetUYaxisCompositeLimits) {
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_DOUBLE_EQ(350, area.GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(5, area.GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, area.GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, area.GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(360, area.GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(10, area.GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, area.GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, area.GetYaxisCompositeEnd(1));
}

TEST(AreaCompRegGrid, GetUYaxisCompositeLimitsMoved) {
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaCompRegGrid area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, 0.0, 360.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_DOUBLE_EQ(360 - 7.5, area.GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(2.5, area.GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, area.GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, area.GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(360 - 2.5, area.GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(7.5, area.GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, area.GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, area.GetYaxisCompositeEnd(1));
}