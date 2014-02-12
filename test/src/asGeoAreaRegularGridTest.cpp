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
#include "asGeoAreaRegularGrid.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorLimitsException)
{
	wxString str("Testing regular grids...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());
	
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = -10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
    }
}

TEST(ConstructorAlternativeLimitsException)
{
    double Umin = -10;
    double Uwidth = 30;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.5;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step), asException);
    }
}

TEST(ConstructorStepException)
{
    double Umin = -10;
    double Uwidth = 30;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.7;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step), asException);
    }
}

TEST(CheckConsistencyException)
{
    double Umin = 10;
    double Uwidth = 0;
    double Vmin = 40;
    double Vwidth = 0;
    double step = 2.5;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step), asException);
    }
}

TEST(IsRectangleTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsRectangle());
}

TEST(IsRectangleFalse)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 15;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
    }
}

TEST(GetBounds)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_CLOSE(10, geoarea.GetUmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetVmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetUmax(), 0.01);
    CHECK_CLOSE(40, geoarea.GetVmax(), 0.01);
}

TEST(GetCenter)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.u, 0.01);
    CHECK_CLOSE(35, center.v, 0.01);
}
/*
TEST(IsOnGridTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5));
}

TEST(IsOnGridTrueTwoAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5, 5));
}

TEST(IsOnGridFalseStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(6));
}

TEST(IsOnGridFalseSecondStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = 10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = 10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(5, 6));
}
*/
TEST(GetAxes)
{
    double Umin = 5;
    double Uwidth = 20;
    double Vmin = 45;
    double Vwidth = 2.5;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step);

    Array1DDouble uaxis;
    uaxis.resize(geoarea.GetUaxisPtsnb());
    uaxis = geoarea.GetUaxis();

    Array1DDouble vaxis;
    vaxis.resize(geoarea.GetVaxisPtsnb());
    vaxis = geoarea.GetVaxis();

    CHECK_CLOSE(5, uaxis[0], 0.000001);
    CHECK_CLOSE(7.5, uaxis[1], 0.000001);
    CHECK_CLOSE(10, uaxis[2], 0.000001);
    CHECK_CLOSE(15, uaxis[4], 0.000001);
    CHECK_CLOSE(45, vaxis[0], 0.000001);
    CHECK_CLOSE(47.5, vaxis[1], 0.000001);
}

}
