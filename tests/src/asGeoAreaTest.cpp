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

#include "asGeoArea.h"
#include "gtest/gtest.h"


TEST(GeoArea, ConstructorLimitsException)
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

    ASSERT_THROW(asGeoArea geoArea(cornerUL, cornerUR, cornerLL, cornerLR), asException);
}

TEST(GeoArea, ConstructorAlternativeLimitsException)
{
    wxLogNull logNo;

    double xMin = -10;
    double xWidth = 30;
    double yMin = 30;
    double yWidth = 10;
    ASSERT_THROW(asGeoArea geoArea(xMin, xWidth, yMin, yWidth), asException);
}

TEST(GeoArea, CheckConsistency)
{
    double xMin = 10;
    double xWidth = 10;
    double yMin = 30;
    double yWidth = 10;
    asGeoArea geoArea(xMin, xWidth, yMin, yWidth);

    EXPECT_DOUBLE_EQ(30, geoArea.GetCornerLL().y);
    EXPECT_DOUBLE_EQ(30, geoArea.GetCornerLR().y);
    EXPECT_DOUBLE_EQ(40, geoArea.GetCornerUL().y);
    EXPECT_DOUBLE_EQ(40, geoArea.GetCornerUR().y);
}

TEST(GeoArea, IsRectangleTrue)
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
    asGeoArea geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoArea, IsRectangleFalse)
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
    asGeoArea geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_FALSE(geoArea.IsRectangle());
}

TEST(GeoArea, GetBounds)
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
    asGeoArea geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(10, geoArea.GetXwidth());
    EXPECT_DOUBLE_EQ(10, geoArea.GetYwidth());
}

TEST(GeoArea, GetCenter)
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
    asGeoArea geoArea(cornerUL, cornerUR, cornerLL, cornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoArea, NegativeSize)
{
    double xMin = 10;
    double xWidth = -7;
    double yMin = 46;
    double yWidth = -2;

    asGeoArea geoArea(xMin, xWidth, yMin, yWidth, asNONE, asNONE, asFLAT_ALLOWED);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(46, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(0, geoArea.GetXwidth());
    EXPECT_DOUBLE_EQ(0, geoArea.GetYwidth());
}
