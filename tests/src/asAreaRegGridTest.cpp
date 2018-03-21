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

#include "asAreaRegGrid.h"
#include "gtest/gtest.h"


TEST(AreaRegGrid, ConstructorLimitsException)
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
    double step = 2.5;

    ASSERT_THROW(asAreaRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step), asException);
}

TEST(AreaRegGrid, ConstructorAlternativeLimitsException)
{
    wxLogNull logNo;

    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.5;

    ASSERT_THROW(asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step), asException);
}

TEST(AreaRegGrid, ConstructorStepException)
{
    wxLogNull logNo;

    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    double step = 2.7;

    ASSERT_THROW(asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step), asException);
}

TEST(AreaRegGrid, IsRectangleTrue)
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
    asAreaRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_TRUE(area.IsRectangle());
}

TEST(AreaRegGrid, IsRectangleFalse)
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

    ASSERT_THROW(asAreaRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step), asException);
}

TEST(AreaRegGrid, GetBounds)
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
    asAreaRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    EXPECT_DOUBLE_EQ(10, area.GetXmin());
    EXPECT_DOUBLE_EQ(30, area.GetYmin());
    EXPECT_DOUBLE_EQ(20, area.GetXmax());
    EXPECT_DOUBLE_EQ(40, area.GetYmax());
}

TEST(AreaRegGrid, GetCenter)
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
    asAreaRegGrid area(cornerUL, cornerUR, cornerLL, cornerLR, step, step);

    Coo center = area.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(AreaRegGrid, GetAxes)
{
    double xMin = 5;
    double xWidth = 20;
    double yMin = 45;
    double yWidth = 2.5;
    double step = 2.5;
    asAreaRegGrid area(xMin, xWidth, step, yMin, yWidth, step);

    a1d uaxis;
    uaxis.resize(area.GetXaxisPtsnb());
    uaxis = area.GetXaxis();

    a1d vaxis;
    vaxis.resize(area.GetYaxisPtsnb());
    vaxis = area.GetYaxis();

    EXPECT_DOUBLE_EQ(5, uaxis[0]);
    EXPECT_DOUBLE_EQ(7.5, uaxis[1]);
    EXPECT_DOUBLE_EQ(10, uaxis[2]);
    EXPECT_DOUBLE_EQ(15, uaxis[4]);
    EXPECT_DOUBLE_EQ(45, vaxis[0]);
    EXPECT_DOUBLE_EQ(47.5, vaxis[1]);
}
