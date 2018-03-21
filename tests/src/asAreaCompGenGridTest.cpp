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


TEST(AreaCompGenGrid, ConstructorAlternativeOneArea)
{
    double xMin = 10;
    int xPtsNb = 5;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    EXPECT_EQ(1, area->GetNbComposites());
    wxDELETE(area);
}

TEST(AreaCompGenGrid, ConstructorAlternativeTwoAreas)
{
    double xMin = -10;
    int xPtsNb = 13;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    EXPECT_EQ(2, area->GetNbComposites());
    wxDELETE(area);
}

TEST(AreaCompGenGrid, CheckConsistency)
{
    double xMin = -5;
    int xPtsNb = 11;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    EXPECT_DOUBLE_EQ(355, area->GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, area->GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, area->GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, area->GetCornerLR().x);
    wxDELETE(area);
}

TEST(AreaCompGenGrid, CheckConsistencyException)
{
    wxLogNull logNo;

    double xMin = 10;
    int xPtsNb = 1;
    double yMin = 40;
    int yPtsNb = -5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = NULL;
    ASSERT_THROW(area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step), asException);
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetBoundsSplitted)
{
    double xMin = -10;
    int xPtsNb = 13;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    EXPECT_DOUBLE_EQ(0, area->GetXmin());
    EXPECT_DOUBLE_EQ(30, area->GetYmin());
    EXPECT_DOUBLE_EQ(360, area->GetXmax());
    EXPECT_DOUBLE_EQ(40, area->GetYmax());
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisCompositeSize)
{
    double xMin = -40;
    int xPtsNb = 21;
    double yMin = 30;
    int yPtsNb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, step, yMin, yPtsNb, step);

    EXPECT_EQ(5, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(17, area->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(1));
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisCompositeSizeStepLon)
{
    double xMin = -40;
    int xPtsNb = 11;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_EQ(3, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(9, area->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(1));
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisCompositeSizeStepLonMoved)
{
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_EQ(2, area->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(2, area->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, area->GetYaxisCompositePtsnb(1));
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisCompositeWidthStepLonMoved)
{
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_DOUBLE_EQ(7.5, area->GetXaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(7.5, area->GetXaxisCompositeWidth(1));
    EXPECT_DOUBLE_EQ(10, area->GetYaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(10, area->GetYaxisCompositeWidth(1));
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisPtsnbStepLonMoved)
{
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_EQ(4, area->GetXaxisPtsnb());
    EXPECT_EQ(5, area->GetYaxisPtsnb());
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisWidthStepLonMoved)
{
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_DOUBLE_EQ(15, area->GetXaxisWidth());
    EXPECT_DOUBLE_EQ(10, area->GetYaxisWidth());
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisCompositeLimits)
{
    double xMin = -10;
    int xPtsNb = 5;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_DOUBLE_EQ(0, area->GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(350, area->GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, area->GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, area->GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(10, area->GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360, area->GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, area->GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, area->GetYaxisCompositeEnd(1));
    wxDELETE(area);
}

TEST(AreaCompGenGrid, GetUYaxisCompositeLimitsMoved)
{
    double xMin = -7.5;
    int xPtsNb = 4;
    double yMin = 30;
    int yPtsNb = 5;
    double xStep = 5;
    double yStep = 2.5;
    wxString gridType = "Regular";
    asAreaCompGrid *area = asAreaCompGrid::GetInstance(gridType, xMin, xPtsNb, xStep, yMin, yPtsNb, yStep);

    EXPECT_DOUBLE_EQ(2.5, area->GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(360 - 7.5, area->GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, area->GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, area->GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(7.5, area->GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360 - 2.5, area->GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, area->GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, area->GetYaxisCompositeEnd(1));
    wxDELETE(area);
}
