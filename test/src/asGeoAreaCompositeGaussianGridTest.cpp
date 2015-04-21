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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */

#include "include_tests.h"
#include "asGeoAreaCompositeGrid.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorAlternativeOneArea)
{
	wxPrintf("Testing composite gaussian grids...\n");
	
    double Xmin = 9.375;
    int Xptsnb = 5;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    CHECK_CLOSE(16.875, geoarea->GetXmax(), 0.001);
    CHECK_CLOSE(37.142, geoarea->GetYmax(), 0.001);
    wxDELETE(geoarea);
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Xmin = -9.375;
    int Xptsnb = 10;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(2, geoarea->GetNbComposites());
    wxDELETE(geoarea);
}

TEST(CheckConsistency)
{
    double Xmin = -9.375;
    int Xptsnb = 10;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(350.625, geoarea->GetCornerUL().x, 0.01);
    CHECK_CLOSE(350.625, geoarea->GetCornerLL().x, 0.01);
    CHECK_CLOSE(7.5, geoarea->GetCornerUR().x, 0.01);
    CHECK_CLOSE(7.5, geoarea->GetCornerLR().x, 0.01);
    wxDELETE(geoarea);
}

TEST(GetBoundsSplitted)
{
    double Xmin = -9.375;
    int Xptsnb = 10;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(0, geoarea->GetXmin(), 0.01);
    CHECK_CLOSE(29.523, geoarea->GetYmin(), 0.01);
    CHECK_CLOSE(360, geoarea->GetXmax(), 0.01);
    CHECK_CLOSE(37.142, geoarea->GetYmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeSize)
{
    double Xmin = -15;
    int Xptsnb = 20;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(2, geoarea->GetNbComposites());
    CHECK_CLOSE(12, geoarea->GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(9, geoarea->GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeSizeAllWest)
{
    double Xmin = -15;
    int Xptsnb = 4;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    CHECK_CLOSE(4, geoarea->GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(345, geoarea->GetXmin(), 0.01);
    CHECK_CLOSE(350.625, geoarea->GetXmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeSizeEdge)
{
    double Xmin = -15;
    int Xptsnb = 9;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    CHECK_CLOSE(9, geoarea->GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(345, geoarea->GetXmin(), 0.01);
    CHECK_CLOSE(360, geoarea->GetXmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeWidth)
{
    double Xmin = -15;
    int Xptsnb = 20;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(20.625, geoarea->GetXaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(15, geoarea->GetXaxisCompositeWidth(1), 0.01);
    CHECK_CLOSE(37.142-29.523, geoarea->GetYaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(37.142-29.523, geoarea->GetYaxisCompositeWidth(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisPtsnb)
{
    double Xmin = -15;
    int Xptsnb = 20;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(20, geoarea->GetXaxisPtsnb(), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisPtsnb(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisWidth)
{
    double Xmin = -15;
    int Xptsnb = 20;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(35.625, geoarea->GetXaxisWidth(), 0.01);
    CHECK_CLOSE(37.142-29.523, geoarea->GetYaxisWidth(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeLimits)
{
    double Xmin = -15;
    int Xptsnb = 20;
    double Ymin = 29.523;
    int Yptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(0, geoarea->GetXaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(345, geoarea->GetXaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(29.523, geoarea->GetYaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(29.523, geoarea->GetYaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(20.625, geoarea->GetXaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360, geoarea->GetXaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(37.142, geoarea->GetYaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(37.142, geoarea->GetYaxisCompositeEnd(1), 0.01);
    wxDELETE(geoarea);
}

}
