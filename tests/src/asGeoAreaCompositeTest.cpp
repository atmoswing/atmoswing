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
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = 10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = 10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, ConstructorTwoAreas)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, ConstructorAlternativeOneArea)
{
    double Xmin = 10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoArea(Xmin, Xwidth, Ymin, Ywidth);

    EXPECT_EQ(1, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, ConstructorAlternativeTwoAreas)
{
    double Xmin = -10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoArea(Xmin, Xwidth, Ymin, Ywidth);

    EXPECT_EQ(2, geoArea.GetNbComposites());
}

TEST(GeoAreaCompositeGrid, CheckConsistency)
{
    double Xmin = -5;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoArea(Xmin, Xwidth, Ymin, Ywidth);

    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, geoArea.GetCornerLL().x);
    EXPECT_DOUBLE_EQ(15, geoArea.GetCornerUR().x);
    EXPECT_DOUBLE_EQ(15, geoArea.GetCornerLR().x);
}

TEST(GeoAreaCompositeGrid, IsSquareTrue)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoAreaCompositeGrid, IsSquareFalse)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_FALSE(geoArea.IsRectangle());
}

TEST(GeoAreaCompositeGrid, GetBounds)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(20, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea.GetYmax());
}

TEST(GeoAreaCompositeGrid, GetBoundsSplitted)
{
    double Xmin = -10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 40;
    asGeoAreaComposite geoArea(Xmin, Xwidth, Ymin, Ywidth);

    EXPECT_DOUBLE_EQ(0, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(360, geoArea.GetXmax());
    EXPECT_DOUBLE_EQ(70, geoArea.GetYmax());
}

TEST(GeoAreaCompositeGrid, GetCenter)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeGrid, GetCenterSplitted)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(345, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeGrid, GetCenterSplittedEdge)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(360, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoAreaCompositeGrid, GetCornersSplitted)
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
    asGeoAreaComposite geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

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
