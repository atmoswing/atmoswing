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
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 20;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 20;
    CornerLR.y = 30;

    ASSERT_THROW(asGeoArea geoArea(CornerUL, CornerUR, CornerLL, CornerLR), asException);
}

TEST(GeoArea, ConstructorAlternativeLimitsException)
{
    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    ASSERT_THROW(asGeoArea geoArea(Xmin, Xwidth, Ymin, Ywidth), asException);
}

TEST(GeoArea, CheckConsistency)
{
    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 30;
    double Ywidth = 10;
    asGeoArea geoArea(Xmin, Xwidth, Ymin, Ywidth);

    EXPECT_DOUBLE_EQ(30, geoArea.GetCornerLL().y);
    EXPECT_DOUBLE_EQ(30, geoArea.GetCornerLR().y);
    EXPECT_DOUBLE_EQ(40, geoArea.GetCornerUL().y);
    EXPECT_DOUBLE_EQ(40, geoArea.GetCornerUR().y);
}

TEST(GeoArea, IsRectangleTrue)
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
    asGeoArea geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_TRUE(geoArea.IsRectangle());
}

TEST(GeoArea, IsRectangleFalse)
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
    asGeoArea geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_FALSE(geoArea.IsRectangle());
}

TEST(GeoArea, GetBounds)
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
    asGeoArea geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(10, geoArea.GetXwidth());
    EXPECT_DOUBLE_EQ(10, geoArea.GetYwidth());
}

TEST(GeoArea, GetCenter)
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
    asGeoArea geoArea(CornerUL, CornerUR, CornerLL, CornerLR);

    Coo center = geoArea.GetCenter();
    EXPECT_DOUBLE_EQ(15, center.x);
    EXPECT_DOUBLE_EQ(35, center.y);
}

TEST(GeoArea, NegativeSize)
{
    double Xmin = 10;
    double Xwidth = -7;
    double Ymin = 46;
    double Ywidth = -2;

    asGeoArea geoArea(Xmin, Xwidth, Ymin, Ywidth, asNONE, asNONE, asFLAT_ALLOWED);

    EXPECT_DOUBLE_EQ(10, geoArea.GetXmin());
    EXPECT_DOUBLE_EQ(46, geoArea.GetYmin());
    EXPECT_DOUBLE_EQ(0, geoArea.GetXwidth());
    EXPECT_DOUBLE_EQ(0, geoArea.GetYwidth());
}
