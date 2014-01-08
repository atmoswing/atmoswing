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
    double Umin = 9.375;
    int Uptsnb = 5;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    CHECK_CLOSE(16.875, geoarea->GetUmax(), 0.001);
    CHECK_CLOSE(37.142, geoarea->GetVmax(), 0.001);
    wxDELETE(geoarea);
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Umin = -9.375;
    int Uptsnb = 10;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(2, geoarea->GetNbComposites());
    wxDELETE(geoarea);
}

TEST(CheckConsistency)
{
    double Umin = -9.375;
    int Uptsnb = 10;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(350.625, geoarea->GetCornerUL().u, 0.01);
    CHECK_CLOSE(350.625, geoarea->GetCornerLL().u, 0.01);
    CHECK_CLOSE(7.5, geoarea->GetCornerUR().u, 0.01);
    CHECK_CLOSE(7.5, geoarea->GetCornerLR().u, 0.01);
    wxDELETE(geoarea);
}

TEST(GetBoundsSplitted)
{
    double Umin = -9.375;
    int Uptsnb = 10;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(0, geoarea->GetUmin(), 0.01);
    CHECK_CLOSE(29.523, geoarea->GetVmin(), 0.01);
    CHECK_CLOSE(360, geoarea->GetUmax(), 0.01);
    CHECK_CLOSE(37.142, geoarea->GetVmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeSize)
{
    double Umin = -15;
    int Uptsnb = 20;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(2, geoarea->GetNbComposites());
    CHECK_CLOSE(12, geoarea->GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(9, geoarea->GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeSizeAllWest)
{
    double Umin = -15;
    int Uptsnb = 4;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    CHECK_CLOSE(4, geoarea->GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(345, geoarea->GetUmin(), 0.01);
    CHECK_CLOSE(350.625, geoarea->GetUmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeSizeEdge)
{
    double Umin = -15;
    int Uptsnb = 9;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    CHECK_CLOSE(9, geoarea->GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(345, geoarea->GetUmin(), 0.01);
    CHECK_CLOSE(360, geoarea->GetUmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeWidth)
{
    double Umin = -15;
    int Uptsnb = 20;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(20.625, geoarea->GetUaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(15, geoarea->GetUaxisCompositeWidth(1), 0.01);
    CHECK_CLOSE(37.142-29.523, geoarea->GetVaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(37.142-29.523, geoarea->GetVaxisCompositeWidth(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisPtsnb)
{
    double Umin = -15;
    int Uptsnb = 20;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(20, geoarea->GetUaxisPtsnb(), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisPtsnb(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisWidth)
{
    double Umin = -15;
    int Uptsnb = 20;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(35.625, geoarea->GetUaxisWidth(), 0.01);
    CHECK_CLOSE(37.142-29.523, geoarea->GetVaxisWidth(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeLimits)
{
    double Umin = -15;
    int Uptsnb = 20;
    double Vmin = 29.523;
    int Vptsnb = 5;
    double step = 0;
    wxString gridType = "GaussianT62";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(0, geoarea->GetUaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(345, geoarea->GetUaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(29.523, geoarea->GetVaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(29.523, geoarea->GetVaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(20.625, geoarea->GetUaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360, geoarea->GetUaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(37.142, geoarea->GetVaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(37.142, geoarea->GetVaxisCompositeEnd(1), 0.01);
    wxDELETE(geoarea);
}

}
