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


TEST(GeoAreaCompositeGeneralGrid, ConstructorAlternativeOneArea)
{
    double Xmin = 10;
    int Xptsnb = 5;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    EXPECT_EQ(1, geoArea->GetNbComposites());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, ConstructorAlternativeTwoAreas)
{
    double Xmin = -10;
    int Xptsnb = 13;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    EXPECT_EQ(2, geoArea->GetNbComposites());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, CheckConsistency)
{
    double Xmin = -5;
    int Xptsnb = 11;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    EXPECT_DOUBLE_EQ(355, geoArea->GetCornerUL().x);
    EXPECT_DOUBLE_EQ(355, geoArea->GetCornerLL().x);
    EXPECT_DOUBLE_EQ(20, geoArea->GetCornerUR().x);
    EXPECT_DOUBLE_EQ(20, geoArea->GetCornerLR().x);
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, CheckConsistencyException)
{
    double Xmin = 10;
    int Xptsnb = 1;
    double Ymin = 40;
    int Yptsnb = -5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = NULL;
    ASSERT_THROW(geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step), asException);
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetBoundsSplitted)
{
    double Xmin = -10;
    int Xptsnb = 13;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    EXPECT_DOUBLE_EQ(0, geoArea->GetXmin());
    EXPECT_DOUBLE_EQ(30, geoArea->GetYmin());
    EXPECT_DOUBLE_EQ(360, geoArea->GetXmax());
    EXPECT_DOUBLE_EQ(40, geoArea->GetYmax());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisCompositeSize)
{
    double Xmin = -40;
    int Xptsnb = 21;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    EXPECT_EQ(5, geoArea->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(17, geoArea->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisCompositeSizeStepLon)
{
    double Xmin = -40;
    int Xptsnb = 11;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_EQ(3, geoArea->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(9, geoArea->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisCompositeSizeStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_EQ(2, geoArea->GetXaxisCompositePtsnb(0));
    EXPECT_EQ(2, geoArea->GetXaxisCompositePtsnb(1));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(0));
    EXPECT_EQ(5, geoArea->GetYaxisCompositePtsnb(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisCompositeWidthStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_DOUBLE_EQ(7.5, geoArea->GetXaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(7.5, geoArea->GetXaxisCompositeWidth(1));
    EXPECT_DOUBLE_EQ(10, geoArea->GetYaxisCompositeWidth(0));
    EXPECT_DOUBLE_EQ(10, geoArea->GetYaxisCompositeWidth(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisPtsnbStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_EQ(4, geoArea->GetXaxisPtsnb());
    EXPECT_EQ(5, geoArea->GetYaxisPtsnb());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisWidthStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_DOUBLE_EQ(15, geoArea->GetXaxisWidth());
    EXPECT_DOUBLE_EQ(10, geoArea->GetYaxisWidth());
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisCompositeLimits)
{
    double Xmin = -10;
    int Xptsnb = 5;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_DOUBLE_EQ(0, geoArea->GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(350, geoArea->GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, geoArea->GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, geoArea->GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(10, geoArea->GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360, geoArea->GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, geoArea->GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, geoArea->GetYaxisCompositeEnd(1));
    wxDELETE(geoArea);
}

TEST(GeoAreaCompositeGeneralGrid, GetUYaxisCompositeLimitsMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoArea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    EXPECT_DOUBLE_EQ(2.5, geoArea->GetXaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(360-7.5, geoArea->GetXaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(30, geoArea->GetYaxisCompositeStart(0));
    EXPECT_DOUBLE_EQ(30, geoArea->GetYaxisCompositeStart(1));
    EXPECT_DOUBLE_EQ(7.5, geoArea->GetXaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(360-2.5, geoArea->GetXaxisCompositeEnd(1));
    EXPECT_DOUBLE_EQ(40, geoArea->GetYaxisCompositeEnd(0));
    EXPECT_DOUBLE_EQ(40, geoArea->GetYaxisCompositeEnd(1));
    wxDELETE(geoArea);
}
