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
	wxPrintf("Testing composite grids...\n");
	
    double Xmin = 10;
    int Xptsnb = 5;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    wxDELETE(geoarea);
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Xmin = -10;
    int Xptsnb = 13;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_EQUAL(2, geoarea->GetNbComposites());
    wxDELETE(geoarea);
}

TEST(CheckConsistency)
{
    double Xmin = -5;
    int Xptsnb = 11;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(355, geoarea->GetCornerUL().x, 0.01);
    CHECK_CLOSE(355, geoarea->GetCornerLL().x, 0.01);
    CHECK_CLOSE(20, geoarea->GetCornerUR().x, 0.01);
    CHECK_CLOSE(20, geoarea->GetCornerLR().x, 0.01);
    wxDELETE(geoarea);
}

TEST(CheckConsistencyException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = 10;
        int Xptsnb = 1;
        double Ymin = 40;
        int Yptsnb = -5;
        double step = 2.5;
        wxString gridType = "Regular";
        asGeoAreaCompositeGrid* geoarea = NULL;
        CHECK_THROW(geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step), asException);
        wxDELETE(geoarea);
    }
}

TEST(GetBoundsSplitted)
{
    double Xmin = -10;
    int Xptsnb = 13;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(0, geoarea->GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea->GetYmin(), 0.01);
    CHECK_CLOSE(360, geoarea->GetXmax(), 0.01);
    CHECK_CLOSE(40, geoarea->GetYmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeSize)
{
    double Xmin = -40;
    int Xptsnb = 21;
    double Ymin = 30;
    int Yptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, step, Ymin, Yptsnb, step);

    CHECK_CLOSE(5, geoarea->GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(17, geoarea->GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeSizeStepLon)
{
    double Xmin = -40;
    int Xptsnb = 11;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(3, geoarea->GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(9, geoarea->GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeSizeStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(2, geoarea->GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(2, geoarea->GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeWidthStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(7.5, geoarea->GetXaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(7.5, geoarea->GetXaxisCompositeWidth(1), 0.01);
    CHECK_CLOSE(10, geoarea->GetYaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(10, geoarea->GetYaxisCompositeWidth(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisPtsnbStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(4, geoarea->GetXaxisPtsnb(), 0.01);
    CHECK_CLOSE(5, geoarea->GetYaxisPtsnb(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisWidthStepLonMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(15, geoarea->GetXaxisWidth(), 0.01);
    CHECK_CLOSE(10, geoarea->GetYaxisWidth(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeLimits)
{
    double Xmin = -10;
    int Xptsnb = 5;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(0, geoarea->GetXaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(350, geoarea->GetXaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea->GetYaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea->GetYaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(10, geoarea->GetXaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360, geoarea->GetXaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea->GetYaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea->GetYaxisCompositeEnd(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUYaxisCompositeLimitsMoved)
{
    double Xmin = -7.5;
    int Xptsnb = 4;
    double Ymin = 30;
    int Yptsnb = 5;
    double Xstep = 5;
    double Ystep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(gridType, Xmin, Xptsnb, Xstep, Ymin, Yptsnb, Ystep);

    CHECK_CLOSE(2.5, geoarea->GetXaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(360-7.5, geoarea->GetXaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea->GetYaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea->GetYaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(7.5, geoarea->GetXaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360-2.5, geoarea->GetXaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea->GetYaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea->GetYaxisCompositeEnd(1), 0.01);
    wxDELETE(geoarea);
}

}
