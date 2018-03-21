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

#include "asAreaCompGrid.h"
#include "gtest/gtest.h"


TEST(AreaCompGaussGrid, GaussianT62OneArea)
{
    double xMin = 9.375;
    int xPtsNb = 5;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, area->GetNbComposites());
    EXPECT_DOUBLE_EQ(16.875, area->GetXmax());
    EXPECT_DOUBLE_EQ(37.142, area->GetYmax());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GaussianT382OneArea)
{
    double xMin = 9.375;
    int xPtsNb = 20;
    double yMin = 29.193;
    int yPtsNb = 20;
    double step = 0;
    wxString gridType = "GaussianT382";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, area->GetNbComposites());
    EXPECT_NEAR(15.312, area->GetXmax(), 0.001);
    EXPECT_NEAR(35.126, area->GetYmax(), 0.001);
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, ConstructorAlternativeTwoAreas)
{
    double xMin = -9.375;
    int xPtsNb = 10;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(2, area->GetNbComposites());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, CheckConsistency)
{
    double xMin = -9.375;
    int xPtsNb = 10;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(350.625, area->GetCornerUL().x);
    EXPECT_DOUBLE_EQ(350.625, area->GetCornerLL().x);
    EXPECT_DOUBLE_EQ(7.5, area->GetCornerUR().x);
    EXPECT_DOUBLE_EQ(7.5, area->GetCornerLR().x);
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetBoundsSplitted)
{
    double xMin = -9.375;
    int xPtsNb = 10;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(0, area->GetXmin());
    EXPECT_DOUBLE_EQ(29.523, area->GetYmin());
    EXPECT_DOUBLE_EQ(360, area->GetXmax());
    EXPECT_DOUBLE_EQ(37.142, area->GetYmax());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisCompositeSize)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(2, area->GetNbComposites());
    EXPECT_EQ(12, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(9, area->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(1));
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisCompositeSizeAllWest)
{
    double xMin = -15;
    int xPtsNb = 4;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, area->GetNbComposites());
    EXPECT_EQ(4, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(345, area->GetXmin());
    EXPECT_DOUBLE_EQ(350.625, area->GetXmax());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisCompositeSizeEdge)
{
    double xMin = -15;
    int xPtsNb = 9;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(1, area->GetNbComposites());
    EXPECT_EQ(9, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(0));
    EXPECT_DOUBLE_EQ(345, area->GetXmin());
    EXPECT_DOUBLE_EQ(360, area->GetXmax());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisCompositeWidth)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(20.625, area->GetXaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(15, area->GetXaxisCompositeWidth(1));
    EXPECT_DOUBLE_EQ(37.142 - 29.523, area->GetYaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(37.142 - 29.523, area->GetYaxisCompositeWidth(1));
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisPtsnb)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_EQ(20, area->GetXaxisPtsnb());
    EXPECT_EQ(5, area->GetYaxisPtsnb());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisWidth)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(35.625, area->GetXaxisWidth());
    EXPECT_DOUBLE_EQ(37.142 - 29.523, area->GetYaxisWidth());
    wxDELETE(area);
}

TEST(AreaCompGaussGrid, GetUYaxisCompositeLimits)
{
    double xMin = -15;
    int xPtsNb = 20;
    double yMin = 29.523;
    int yPtsNb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb,
                                                                          step);

    EXPECT_DOUBLE_EQ(0, area->GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(345, area->GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(29.523, area->GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(29.523, area->GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(20.625, area->GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360, area->GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(37.142, area->GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(37.142, area->GetYaxisCompositeEnd(1));
    wxDELETE(area);
}
