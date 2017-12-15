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

#include "asGeoAreaCompositeGrid.h"
#include "gtest/gtest.h"


TEST(GeoAreaCompositeGaussianGrid, GaussianT62OneArea)
{
    double xMin = 9.375;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, geoArea->GetNbComposites());
    EXPECT_DOUBLE_EQ(16.875, geoArea->GetXmax());
    EXPECT_DOUBLE_EQ(37.142, geoArea->GetYmax());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GaussianT382OneArea)
{
    double xMin = 9.375;
    int xPtsNb = 20;
    double yMin = 29.193;
    int yPtsNb = 20;
    double step = 0;
    wxString gridType = "GaussianT382";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, geoArea->GetNbComposites());
    EXPECT_NEAR(15.312, geoArea->GetXmax(), 0.001);
    EXPECT_NEAR(35.126, geoArea->GetYmax(), 0.001);
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, ConstructorAlternativeTwoAreas)
{
    double xMin = -9.375;
    int xPtsNb = 10;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(2, geoArea->GetNbComposites());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, CheckConsistency)
{
    double xMin = -9.375;
    int xPtsNb = 10;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(350.625, geoArea->GetCornerUL().x);
    EXPECT_DOUBLE_EQ(350.625, geoArea->GetCornerLL().x);
    EXPECT_DOUBLE_EQ(7.5, geoArea->GetCornerUR().x);
    EXPECT_DOUBLE_EQ(7.5, geoArea->GetCornerLR().x);
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetBoundsSplitted)
{
    double xMin = -9.375;
    int xPtsNb = 10;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(0, geoArea->GetXmin());
    EXPECT_DOUBLE_EQ(29.523, geoArea->GetYmin());
    EXPECT_DOUBLE_EQ(360, geoArea->GetXmax());
    EXPECT_DOUBLE_EQ(37.142, geoArea->GetYmax());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisCompositeSize)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(2, geoArea->GetNbComposites());
    EXPECT_EQ(12, geoArea->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(9, geoArea->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisCompositeSizeAllWest)
{
    double xMin = -15;
    int xPtsNb = 4;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, geoArea->GetNbComposites());
    EXPECT_EQ(4, geoArea->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(345, geoArea->GetXmin());
    EXPECT_DOUBLE_EQ(350.625, geoArea->GetXmax());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisCompositeSizeEdge)
{
    double xMin = -15;
    int xPtsNb = 9;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, geoArea->GetNbComposites());
    EXPECT_EQ(9, geoArea->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(345, geoArea->GetXmin());
    EXPECT_DOUBLE_EQ(360, geoArea->GetXmax());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisCompositeWidth)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(20.625, geoArea->GetXaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(15, geoArea->GetXaxisCompositeWidth(1));
    EXPECT_DOUBLE_EQ(37.142 - 29.523, geoArea->GetYaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(37.142 - 29.523, geoArea->GetYaxisCompositeWidth(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisPtsnb)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(20, geoArea->GetXaxisPtsnb());
    EXPECT_EQ(5, geoArea->GetYaxisPtsnb());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisWidth)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(35.625, geoArea->GetXaxisWidth());
    EXPECT_DOUBLE_EQ(37.142 - 29.523, geoArea->GetYaxisWidth());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGaussianGrid, GetUYaxisCompositeLimits)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid *geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(0, geoArea->GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(345, geoArea->GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(29.523, geoArea->GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(29.523, geoArea->GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(20.625, geoArea->GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360, geoArea->GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(37.142, geoArea->GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(37.142, geoArea->GetYaxisCompositeEnd(1));
    wxDELETE(geoArea);
}
