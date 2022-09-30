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

#include "asArea.h"

TEST(Area, CheckPointWGS84True) {
    asArea area;
    Coo point;
    point.x = 10;
    point.y = 10;
    EXPECT_TRUE(area.CheckPoint(point, asEDIT_FORBIDDEN));
}

TEST(Area, CheckPointWGS84UVMaxTrue) {
    asArea area;
    Coo point;
    point.x = 360;
    point.y = 90;
    EXPECT_TRUE(area.CheckPoint(point, asEDIT_FORBIDDEN));
}

TEST(Area, CheckPointWGS84UVMinTrue) {
    asArea area;
    Coo point;
    point.x = 0;
    point.y = -90;
    EXPECT_TRUE(area.CheckPoint(point, asEDIT_FORBIDDEN));
}

TEST(Area, CheckPointWGS84UTooHigh) {
    asArea area;
    Coo point;
    point.x = 360.1;
    point.y = 10;
    EXPECT_FALSE(area.CheckPoint(point, asEDIT_FORBIDDEN));
    EXPECT_DOUBLE_EQ(360.1, point.x);
}

TEST(Area, CheckPointWGS84UTooLow) {
    asArea area;
    Coo point;
    point.x = -0.1;
    point.y = 10;
    EXPECT_TRUE(area.CheckPoint(point, asEDIT_FORBIDDEN));
    EXPECT_DOUBLE_EQ(-0.1, point.x);
}

TEST(Area, CheckPointWGS84VTooHigh) {
    asArea area;
    Coo point;
    point.x = 10;
    point.y = 90.1;
    EXPECT_FALSE(area.CheckPoint(point, asEDIT_FORBIDDEN));
    EXPECT_DOUBLE_EQ(90.1, point.y);
}

TEST(Area, CheckPointWGS84VTooLow) {
    asArea area;
    Coo point;
    point.x = 10;
    point.y = -90.1;
    EXPECT_FALSE(area.CheckPoint(point, asEDIT_FORBIDDEN));
    EXPECT_DOUBLE_EQ(-90.1, point.y);
}

TEST(Area, CheckPointWGS84UTooHighCorr) {
    asArea area;
    Coo point;
    point.x = 360.1;
    point.y = 10;
    EXPECT_FALSE(area.CheckPoint(point, asEDIT_ALLOWED));
    EXPECT_FLOAT_EQ(0.1f, point.x);
}

TEST(Area, CheckPointWGS84UTooLowCorr) {
    asArea area;
    Coo point;
    point.x = -0.1;
    point.y = 10;
    EXPECT_TRUE(area.CheckPoint(point, asEDIT_ALLOWED));
    EXPECT_DOUBLE_EQ(-0.1, point.x);
}

TEST(Area, CheckPointWGS84VTooHighCorr) {
    asArea area;
    Coo point;
    point.x = 10;
    point.y = 90.1;
    EXPECT_FALSE(area.CheckPoint(point, asEDIT_ALLOWED));
}

TEST(Area, CheckPointWGS84VTooLowCorr) {
    asArea area;
    Coo point;
    point.x = 10;
    point.y = -90.1;
    EXPECT_FALSE(area.CheckPoint(point, asEDIT_ALLOWED));
}

TEST(Area, IsRectangleTrue) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asArea area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_TRUE(area.IsRectangle());
}

TEST(Area, IsRectangleFalse) {
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

    EXPECT_THROW(asArea area(cornerUL, cornerUR, cornerLL, cornerLR), std::exception);
}

TEST(Area, GetBounds) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asArea area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_DOUBLE_EQ(10, area.GetXmin());
    EXPECT_DOUBLE_EQ(30, area.GetYmin());
    EXPECT_DOUBLE_EQ(10, area.GetXwidth());
    EXPECT_DOUBLE_EQ(10, area.GetYwidth());
    EXPECT_DOUBLE_EQ(20, area.GetXmax());
    EXPECT_DOUBLE_EQ(40, area.GetYmax());
}

TEST(Area, NegativeSize) {
    double xMin = 10;
    double xWidth = -7;
    double yMin = 46;
    double yWidth = -2;

    asArea area(xMin, xWidth, yMin, yWidth, asFLAT_ALLOWED);

    EXPECT_DOUBLE_EQ(10, area.GetXmin());
    EXPECT_DOUBLE_EQ(46, area.GetYmin());
    EXPECT_DOUBLE_EQ(0, area.GetXwidth());
    EXPECT_DOUBLE_EQ(0, area.GetYwidth());
}

TEST(Area, CheckConsistencyWithPositive) {
    double xMin = 10;
    double xWidth = 10;
    double yMin = 30;
    double yWidth = 10;
    asArea area(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(30, area.GetCornerLL().y);
    EXPECT_DOUBLE_EQ(30, area.GetCornerLR().y);
    EXPECT_DOUBLE_EQ(40, area.GetCornerUL().y);
    EXPECT_DOUBLE_EQ(40, area.GetCornerUR().y);
}

TEST(Area, CheckConsistencyWithNegative) {
    double xMin = -5;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asArea area(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(-5, area.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(-5, area.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(15, area.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(15, area.GetCornerLR().x);
}

TEST(Area, IsSquareTrue) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asArea area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_TRUE(area.IsRectangle());
}

TEST(Area, IsSquareFalse) {
    wxLogNull noLog;

    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 15;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;

    EXPECT_THROW(asArea area(cornerUL, cornerUR, cornerLL, cornerLR), std::exception);
}
