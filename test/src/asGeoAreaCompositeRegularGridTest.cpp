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
#include "asGeoAreaCompositeRegularGrid.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorStepException)
{
	wxPrintf("Testing composite regular grids...\n");
	
    if(g_unitTestExceptions)
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
        double step = 2.6;
        CHECK_THROW(asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
    }
}

TEST(ConstructorOneArea)
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
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(1, geoarea.GetNbComposites());
}

TEST(ConstructorTwoAreas)
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
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(2, geoarea.GetNbComposites());
}

TEST(ConstructorAlternativeOneArea)
{
    double Xmin = 10;
    double Xwidth = 10;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    CHECK_EQUAL(1, geoarea.GetNbComposites());
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    CHECK_EQUAL(2, geoarea.GetNbComposites());
}

TEST(CheckConsistency)
{
    double Xmin = -5;
    double Xwidth = 25;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    CHECK_CLOSE(355, geoarea.GetCornerUL().x, 0.01);
    CHECK_CLOSE(355, geoarea.GetCornerLL().x, 0.01);
    CHECK_CLOSE(20, geoarea.GetCornerUR().x, 0.01);
    CHECK_CLOSE(20, geoarea.GetCornerLR().x, 0.01);
}

TEST(CheckConsistencyException)
{
    if(g_unitTestExceptions)
    {
        double Xmin = 10;
        double Xwidth = 0;
        double Ymin = 40;
        double Ywidth = -10;
        double step = 2.5;
        CHECK_THROW(asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step), asException);
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
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

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
        CHECK_THROW(asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
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
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_CLOSE(10, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetXmax(), 0.01);
    CHECK_CLOSE(40, geoarea.GetYmax(), 0.01);
}

TEST(GetBoundsSplitted)
{
    double Xmin = -10;
    double Xwidth = 30;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    CHECK_CLOSE(0, geoarea.GetXmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetYmin(), 0.01);
    CHECK_CLOSE(360, geoarea.GetXmax(), 0.01);
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
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(GetCenterSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(345, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(GetCenterSplittedEdge)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -10;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -10;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(360, center.x, 0.01);
    CHECK_CLOSE(35, center.y, 0.01);
}

TEST(GetCornersSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_CLOSE(0, geoarea.GetComposite(0).GetCornerUL().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(0).GetCornerUL().y, 0.01);
    CHECK_CLOSE(10, geoarea.GetComposite(0).GetCornerUR().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(0).GetCornerUR().y, 0.01);
    CHECK_CLOSE(0, geoarea.GetComposite(0).GetCornerLL().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(0).GetCornerLL().y, 0.01);
    CHECK_CLOSE(10, geoarea.GetComposite(0).GetCornerLR().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(0).GetCornerLR().y, 0.01);

    CHECK_CLOSE(320, geoarea.GetComposite(1).GetCornerUL().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(1).GetCornerUL().y, 0.01);
    CHECK_CLOSE(360, geoarea.GetComposite(1).GetCornerUR().x, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(1).GetCornerUR().y, 0.01);
    CHECK_CLOSE(320, geoarea.GetComposite(1).GetCornerLL().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(1).GetCornerLL().y, 0.01);
    CHECK_CLOSE(360, geoarea.GetComposite(1).GetCornerLR().x, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(1).GetCornerLR().y, 0.01);
}
/*
TEST(IsOnGridTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5));
}

TEST(IsOnGridTrueTwoAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5, 5));
}

TEST(IsOnGridFalseStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(6));
}

TEST(IsOnGridFalseSecondStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(5, 6));
}
*/
TEST(GetAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.x = -40;
    CornerUL.y = 40;
    CornerUR.x = 10;
    CornerUR.y = 40;
    CornerLL.x = -40;
    CornerLL.y = 30;
    CornerLR.x = 10;
    CornerLR.y = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Array1DDouble uaxis0, vaxis0;
    uaxis0.resize(geoarea.GetXaxisCompositePtsnb(0));
    vaxis0.resize(geoarea.GetYaxisCompositePtsnb(0));

    uaxis0 = geoarea.GetXaxisComposite(0);
    vaxis0 = geoarea.GetYaxisComposite(0);

    CHECK_CLOSE(0, uaxis0(0), 0.000001);
    CHECK_CLOSE(2.5, uaxis0(1), 0.000001);
    CHECK_CLOSE(5, uaxis0(2), 0.000001);
    CHECK_CLOSE(7.5, uaxis0(3), 0.000001);
    CHECK_CLOSE(10, uaxis0(4), 0.000001);
    CHECK_CLOSE(30, vaxis0(0), 0.000001);
    CHECK_CLOSE(32.5, vaxis0(1), 0.000001);
    CHECK_CLOSE(40, vaxis0(4), 0.000001);

    Array1DDouble uaxis1, vaxis1;
    uaxis1.resize(geoarea.GetXaxisCompositePtsnb(1));
    vaxis1.resize(geoarea.GetYaxisCompositePtsnb(1));

    uaxis1 = geoarea.GetXaxisComposite(1);
    vaxis1 = geoarea.GetYaxisComposite(1);

    CHECK_CLOSE(320, uaxis1(0), 0.000001);
    CHECK_CLOSE(322.5, uaxis1(1), 0.000001);
    CHECK_CLOSE(325, uaxis1(2), 0.000001);
    CHECK_CLOSE(360, uaxis1(geoarea.GetXaxisCompositePtsnb(1)-1), 0.000001);
    CHECK_CLOSE(30, vaxis1(0), 0.000001);
    CHECK_CLOSE(32.5, vaxis1(1), 0.000001);
    CHECK_CLOSE(40, vaxis1(4), 0.000001);
}

TEST(GetUYaxisCompositeSize)
{
    double Xmin = -40;
    double Xwidth = 50;
    double Ymin = 30;
    double Ywidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, step, Ymin, Ywidth, step);

    CHECK_CLOSE(5, geoarea.GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(17, geoarea.GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisCompositePtsnb(1), 0.01);
}

TEST(GetUYaxisCompositeSizeStepLon)
{
    double Xmin = -40;
    double Xwidth = 50;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(3, geoarea.GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(9, geoarea.GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisCompositePtsnb(1), 0.01);
}

TEST(GetUYaxisCompositeSizeStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(2, geoarea.GetXaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(2, geoarea.GetXaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisCompositePtsnb(1), 0.01);
}

TEST(GetUYaxisCompositeWidthStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(7.5, geoarea.GetXaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(7.5, geoarea.GetXaxisCompositeWidth(1), 0.01);
    CHECK_CLOSE(10, geoarea.GetYaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(10, geoarea.GetYaxisCompositeWidth(1), 0.01);
}

TEST(GetUYaxisPtsnbStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(4, geoarea.GetXaxisPtsnb(), 0.01);
    CHECK_CLOSE(5, geoarea.GetYaxisPtsnb(), 0.01);
}

TEST(GetUYaxisWidthStepLonMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(15, geoarea.GetXaxisWidth(), 0.01);
    CHECK_CLOSE(10, geoarea.GetYaxisWidth(), 0.01);
}

TEST(GetUYaxisCompositeLimits)
{
    double Xmin = -10;
    double Xwidth = 20;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(0, geoarea.GetXaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(350, geoarea.GetXaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea.GetYaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea.GetYaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(10, geoarea.GetXaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360, geoarea.GetXaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea.GetYaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea.GetYaxisCompositeEnd(1), 0.01);
}

TEST(GetUYaxisCompositeLimitsMoved)
{
    double Xmin = -7.5;
    double Xwidth = 15;
    double Ymin = 30;
    double Ywidth = 10;
    double Xstep = 5;
    double Ystep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(Xmin, Xwidth, Xstep, Ymin, Ywidth, Ystep);

    CHECK_CLOSE(2.5, geoarea.GetXaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(360-7.5, geoarea.GetXaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea.GetYaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea.GetYaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(7.5, geoarea.GetXaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360-2.5, geoarea.GetXaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea.GetYaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea.GetYaxisCompositeEnd(1), 0.01);
}

}
