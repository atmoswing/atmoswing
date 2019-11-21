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

#include "asAreaComp.h"
#include "gtest/gtest.h"

TEST(AreaComp, ConstructorOneArea) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asAreaComp area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_EQ(1, area.GetNbComposites());
}

TEST(AreaComp, ConstructorTwoAreas) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = -10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asAreaComp area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_EQ(2, area.GetNbComposites());
}

TEST(AreaComp, ConstructorAlternativeOneArea) {
    double xMin = 10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asAreaComp area(xMin, xWidth, yMin, yWidth);

    EXPECT_EQ(1, area.GetNbComposites());
}

TEST(AreaComp, ConstructorAlternativeTwoAreas) {
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asAreaComp area(xMin, xWidth, yMin, yWidth);

    EXPECT_EQ(2, area.GetNbComposites());
}

TEST(AreaComp, CheckConsistency) {
    double xMin = -5;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asAreaComp area(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(355, area.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, area.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(15, area.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(15, area.GetCornerLR().x);
}

TEST(AreaComp, IsSquareTrue) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asAreaComp area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_TRUE(area.IsRectangle());
}

TEST(AreaComp, IsSquareFalse) {
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

    EXPECT_THROW(asAreaComp area(cornerUL, cornerUR, cornerLL, cornerLR), std::exception);
}

TEST(AreaComp, GetBounds) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 10;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asAreaComp area(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_DOUBLE_EQ(10, area.GetComposite(0).GetXmin());
    EXPECT_DOUBLE_EQ(30, area.GetComposite(0).GetYmin());
    EXPECT_DOUBLE_EQ(20, area.GetComposite(0).GetXmax());
    EXPECT_DOUBLE_EQ(40, area.GetComposite(0).GetYmax());
}

TEST(AreaComp, GetBoundsSplitted) {
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asAreaComp area(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(350, area.GetComposite(0).GetXmin());
    EXPECT_DOUBLE_EQ(360, area.GetComposite(0).GetXmax());
    EXPECT_DOUBLE_EQ(0, area.GetComposite(1).GetXmin());
    EXPECT_DOUBLE_EQ(10, area.GetComposite(1).GetXmax());
    EXPECT_DOUBLE_EQ(30, area.GetComposite(0).GetYmin());
    EXPECT_DOUBLE_EQ(70, area.GetComposite(0).GetYmax());
}

TEST(AreaComp, GetCornersSplitted) {
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = -40;
    cornerUL.y = 40;
    cornerUR.x = 10;
    cornerUR.y = 40;
    cornerLL.x = -40;
    cornerLL.y = 30;
    cornerLR.x = 10;
    cornerLR.y = 30;
    asAreaComp area(cornerUL, cornerUR, cornerLL, cornerLR);

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
