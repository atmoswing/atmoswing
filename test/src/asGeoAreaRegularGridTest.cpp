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
    if(g_unitTestExceptions)
    {
	    wxString str("Testing regular grids...\n");
        printf("%s", str.mb_str(wxConvUTF8).data());
	
        Coo CornerUL, CornerUR, CornerLL, CornerLR;
        CornerUL.x = -10;
        CornerUL.y = 40;
        CornerUR.x = 20;
        CornerUR.y = 40;
        CornerLL.x = -10;
        CornerLL.y = 30;
        CornerLR.x = 20;
        CornerLR.y = 30;
        double step = 2.5;

        CHECK_THROW(asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
    }
}

TEST(ConstructorAlternativeLimitsException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = -10;
        double Xwidth = 30;
        double Ymin = 30;
        double Ywidth = 10;
        double step = 2.5;
        CHECK_THROW(asGeoAreaRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
    }
}

TEST(ConstructorStepException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = -10;
        double Xwidth = 30;
        double Ymin = 30;
        double Ywidth = 10;
        double step = 2.7;
        CHECK_THROW(asGeoAreaRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
    }
}

TEST(CheckConsistencyException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = 10;
        double Xwidth = 0;
        double Ymin = 40;
        double Ywidth = 0;
        double step = 2.5;
        CHECK_THROW(asGeoAreaRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
    }
}

TEST(IsRectangleTrue)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsRectangle());
}

TEST(IsRectangleFalse)
{
    if(g_unitTestExceptions)
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
        double step = 2.5;
        CHECK_THROW(asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
    }
}

TEST(GetBounds)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetXmax(), 0.01);
    CHECK_CLOSE(40, geoarea.GetYmax(), 0.01);
}

TEST(GetCenter)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}
/*
TEST(IsOnGridTrue)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5));
}

TEST(IsOnGridTrueTwoAxes)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5, 5));
}

TEST(IsOnGridFalseStep)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(6));
}

TEST(IsOnGridFalseSecondStep)
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
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(5, 6));
}
*/
TEST(GetAxes)
{
    double Xmin = 5;
    double Xwidth = 20;
    double Ymin = 45;
    double Ywidth = 2.5;
    double step = 2.5;
    asGeoAreaRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    Array1DDouble uaxis;
    uaxis.resize(geoarea.GetXaxisPtsnb());
    uaxis = geoarea.GetXaxis();

    Array1DDouble vaxis;
    vaxis.resize(geoarea.GetYaxisPtsnb());
    vaxis = geoarea.GetYaxis();

    CHECK_CLOSE(5, uaxis[0], 0.000001);
    CHECK_CLOSE(7.5, uaxis[1], 0.000001);
    CHECK_CLOSE(10, uaxis[2], 0.000001);
    CHECK_CLOSE(15, uaxis[4], 0.000001);
    CHECK_CLOSE(45, vaxis[0], 0.000001);
    CHECK_CLOSE(47.5, vaxis[1], 0.000001);
}

}
