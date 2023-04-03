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

#include <gtest/gtest.h>

#include "asAreaGridRegular.h"

TEST(AreaGridRegular, CheckConsistency) {
    double xMin = -5;
    double xWidth = 25;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaGridRegular area(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_DOUBLE_EQ(-5, area.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(-5, area.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, area.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, area.GetCornerLR().x);
}

TEST(AreaGridRegular, CheckConsistencyException) {
    wxLogNull logNo;

    double xMin = 10;
    double xWidth = 0;
    double yMin = 40;
    double yWidth = -10;
    double step = 2.5;
    ASSERT_THROW(asAreaGridRegular area(xMin, xWidth, step, yMin, yWidth, step), std::exception);
}

TEST(AreaGridRegular, IsRectangleTrue) {
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
    asAreaGridRegular area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);
    EXPECT_TRUE(area.IsRectangle());
}

TEST(AreaGridRegular, IsRectangleFalse) {
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
    ASSERT_THROW(asAreaGridRegular area(cornerUL, cornerUR, cornerLL, cornerLR, step, step), std::exception);
}

TEST(AreaGridRegular, GetBounds) {
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
    asAreaGridRegular area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(10, area.GetXmin());
    EXPECT_DOUBLE_EQ(30, area.GetYmin());
    EXPECT_DOUBLE_EQ(20, area.GetXmax());
    EXPECT_DOUBLE_EQ(40, area.GetYmax());
}

TEST(AreaGridRegular, GetBoundsSplitted) {
    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaGridRegular area(xMin, xWidth, step, yMin, yWidth, step);

    EXPECT_DOUBLE_EQ(-10, area.GetXmin());
    EXPECT_DOUBLE_EQ(20, area.GetXmax());
    EXPECT_DOUBLE_EQ(30, area.GetYmin());
    EXPECT_DOUBLE_EQ(40, area.GetYmax());
}

TEST(AreaGridRegular, GetCornersSplitted) {
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
    asAreaGridRegular area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(-40, area.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(40, area.GetCornerUL().y);
    EXPECT_DOUBLE_EQ(10, area.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(40, area.GetCornerUR().y);
    EXPECT_DOUBLE_EQ(-40, area.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(30, area.GetCornerLL().y);
    EXPECT_DOUBLE_EQ(10, area.GetCornerLR().x);
    EXPECT_DOUBLE_EQ(30, area.GetCornerLR().y);
}

TEST(AreaGridRegular, GetAxes) {
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
    asAreaGridRegular area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    a1d lons = a1d::LinSpaced(145, -180.0, 180.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    a1d uAxis, vAxis;
    uAxis.resize(area.GetXaxisPtsnb());
    vAxis.resize(area.GetYaxisPtsnb());

    uAxis = area.GetXaxis();
    vAxis = area.GetYaxis();

    EXPECT_DOUBLE_EQ(-40, uAxis(0));
    EXPECT_DOUBLE_EQ(-37.5, uAxis(1));
    EXPECT_DOUBLE_EQ(-35, uAxis(2));
    EXPECT_DOUBLE_EQ(10, uAxis(area.GetXaxisPtsnb() - 1));
    EXPECT_DOUBLE_EQ(30, vAxis(0));
    EXPECT_DOUBLE_EQ(32.5, vAxis(1));
    EXPECT_DOUBLE_EQ(40, vAxis(4));
}

TEST(AreaGridRegular, GetUYaxisSize) {
    double xMin = -40;
    double xWidth = 50;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;
    asAreaGridRegular area(xMin, xWidth, step, yMin, yWidth, step);

    a1d lons = a1d::LinSpaced(145, -180.0, 180.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_EQ(21, area.GetXaxisPtsnb());
    EXPECT_EQ(5, area.GetYaxisPtsnb());
}

TEST(AreaGridRegular, GetUYaxisSizeStepLon) {
    double xMin = -40;
    double xWidth = 50;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, -180.0, 180.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_EQ(11, area.GetXaxisPtsnb());
    EXPECT_EQ(5, area.GetYaxisPtsnb());
}

TEST(AreaGridRegular, GetUYaxisSizeStepLonMoved) {
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, -180.0, 180.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_EQ(4, area.GetXaxisPtsnb());
    EXPECT_EQ(5, area.GetYaxisPtsnb());
}

TEST(AreaGridRegular, GetUYaxisLimits) {
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, -180.0, 180.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_DOUBLE_EQ(-10, area.GetXaxisStart());
    EXPECT_DOUBLE_EQ(30, area.GetYaxisStart());
    EXPECT_DOUBLE_EQ(10, area.GetXaxisEnd());
    EXPECT_DOUBLE_EQ(40, area.GetYaxisEnd());
    EXPECT_DOUBLE_EQ(40, area.GetYaxisEnd());
}

TEST(AreaGridRegular, GetUYaxisLimitsMoved) {
    double xMin = -7.5;
    double xWidth = 15;
    double yMin = 30;
    double yWidth = 10;
    double xStep = 5;
    double yStep = 2.5;
    asAreaGridRegular area(xMin, xWidth, xStep, yMin, yWidth, yStep);

    a1d lons = a1d::LinSpaced(145, -180.0, 180.0);
    a1d lats = a1d::LinSpaced(73, -90.0, 90.0);
    area.InitializeAxes(lons, lats);

    EXPECT_DOUBLE_EQ(-7.5, area.GetXaxisStart());
    EXPECT_DOUBLE_EQ(30, area.GetYaxisStart());
    EXPECT_DOUBLE_EQ(7.5, area.GetXaxisEnd());
    EXPECT_DOUBLE_EQ(40, area.GetYaxisEnd());
}