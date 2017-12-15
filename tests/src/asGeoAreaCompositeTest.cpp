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

#include "asGeoAreaComposite.h"
#include "gtest/gtest.h"


TEST(GeoAreaCompositeGrid, ConstructorOneArea)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, ConstructorTwoAreas)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, ConstructorAlternativeOneArea)
{
    double xMin = 10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asGeoAreaComposite geoArea(xMin, xWidth, yMin, yWidth);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, ConstructorAlternativeTwoAreas)
{
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asGeoAreaComposite geoArea(xMin, xWidth, yMin, yWidth);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, CheckConsistency)
{
    double xMin = -5;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asGeoAreaComposite geoArea(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(15, geoArea.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(15, geoArea.GetCornerLR().x);
}

TEST(GeoAreaCompositeGrid, IsSquareTrue)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoAreaCompositeGrid, IsSquareFalse)
{
    Coo cornerUL, cornerUR, cornerLL, cornerLR;
    cornerUL.x = 10;
    cornerUL.y = 40;
    cornerUR.x = 20;
    cornerUR.y = 40;
    cornerLL.x = 15;
    cornerLL.y = 30;
    cornerLR.x = 20;
    cornerLR.y = 30;
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_FALSE(geoArea.IsRectangle());
}

TEST(GeoAreaCompositeGrid, GetBounds)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(20, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaCompositeGrid, GetBoundsSplitted)
{
    double xMin = -10;
    double xWidth = 20;
    double yMin = 30;
    double yWidth = 40;
    asGeoAreaComposite geoArea(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(0, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(360, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(70, geoArea.GetYmax());
}

TEST(GeoAreaCompositeGrid, GetCenter)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeGrid, GetCenterSplitted)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(345, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeGrid, GetCenterSplittedEdge)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(360, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeGrid, GetCornersSplitted)
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
    asGeoAreaComposite geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

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
