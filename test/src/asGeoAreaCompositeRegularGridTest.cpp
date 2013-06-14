#include "include_tests.h"
#include "asGeoAreaCompositeRegularGrid.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorStepException)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -10;
    CornerUL.v = 40;
    CornerUR.u = 20;
    CornerUR.v = 40;
    CornerLL.u = -10;
    CornerLL.v = 30;
    CornerLR.u = 20;
    CornerLR.v = 30;
    double step = 2.6;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
    }
}

TEST(ConstructorOneArea)
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
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(1, geoarea.GetNbComposites());
}

TEST(ConstructorTwoAreas)
{
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
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(2, geoarea.GetNbComposites());
}

TEST(ConstructorAlternativeOneArea)
{
    double Umin = 10;
    double Uwidth = 10;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step);

    CHECK_EQUAL(1, geoarea.GetNbComposites());
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Umin = -10;
    double Uwidth = 30;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step);

    CHECK_EQUAL(2, geoarea.GetNbComposites());
}

TEST(CheckConsistency)
{
    double Umin = -5;
    double Uwidth = 25;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step);

    CHECK_CLOSE(355, geoarea.GetCornerUL().u, 0.01);
    CHECK_CLOSE(355, geoarea.GetCornerLL().u, 0.01);
    CHECK_CLOSE(20, geoarea.GetCornerUR().u, 0.01);
    CHECK_CLOSE(20, geoarea.GetCornerLR().u, 0.01);
}

TEST(CheckConsistencyException)
{
    double Umin = 10;
    double Uwidth = 0;
    double Vmin = 40;
    double Vwidth = -10;
    double step = 2.5;

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step), asException);
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
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

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
        CHECK_THROW(asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step), asException);
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
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_CLOSE(10, geoarea.GetUmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetVmin(), 0.01);
    CHECK_CLOSE(20, geoarea.GetUmax(), 0.01);
    CHECK_CLOSE(40, geoarea.GetVmax(), 0.01);
}

TEST(GetBoundsSplitted)
{
    double Umin = -10;
    double Uwidth = 30;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step);

    CHECK_CLOSE(0, geoarea.GetUmin(), 0.01);
    CHECK_CLOSE(30, geoarea.GetVmin(), 0.01);
    CHECK_CLOSE(360, geoarea.GetUmax(), 0.01);
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
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(15, center.u, 0.01);
    CHECK_CLOSE(35, center.v, 0.01);
}

TEST(GetCenterSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(345, center.u, 0.01);
    CHECK_CLOSE(35, center.v, 0.01);
}

TEST(GetCenterSplittedEdge)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -10;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -10;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Coo center = geoarea.GetCenter();
    CHECK_CLOSE(360, center.u, 0.01);
    CHECK_CLOSE(35, center.v, 0.01);
}

TEST(GetCornersSplitted)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_CLOSE(0, geoarea.GetComposite(0).GetCornerUL().u, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(0).GetCornerUL().v, 0.01);
    CHECK_CLOSE(10, geoarea.GetComposite(0).GetCornerUR().u, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(0).GetCornerUR().v, 0.01);
    CHECK_CLOSE(0, geoarea.GetComposite(0).GetCornerLL().u, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(0).GetCornerLL().v, 0.01);
    CHECK_CLOSE(10, geoarea.GetComposite(0).GetCornerLR().u, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(0).GetCornerLR().v, 0.01);

    CHECK_CLOSE(320, geoarea.GetComposite(1).GetCornerUL().u, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(1).GetCornerUL().v, 0.01);
    CHECK_CLOSE(360, geoarea.GetComposite(1).GetCornerUR().u, 0.01);
    CHECK_CLOSE(40, geoarea.GetComposite(1).GetCornerUR().v, 0.01);
    CHECK_CLOSE(320, geoarea.GetComposite(1).GetCornerLL().u, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(1).GetCornerLL().v, 0.01);
    CHECK_CLOSE(360, geoarea.GetComposite(1).GetCornerLR().u, 0.01);
    CHECK_CLOSE(30, geoarea.GetComposite(1).GetCornerLR().v, 0.01);
}
/*
TEST(IsOnGridTrue)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5));
}

TEST(IsOnGridTrueTwoAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(true, geoarea.IsOnGrid(2.5, 5));
}

TEST(IsOnGridFalseStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(6));
}

TEST(IsOnGridFalseSecondStep)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    CHECK_EQUAL(false, geoarea.IsOnGrid(5, 6));
}
*/
TEST(GetAxes)
{
    Coo CornerUL, CornerUR, CornerLL, CornerLR;
    CornerUL.u = -40;
    CornerUL.v = 40;
    CornerUR.u = 10;
    CornerUR.v = 40;
    CornerLL.u = -40;
    CornerLL.v = 30;
    CornerLR.u = 10;
    CornerLR.v = 30;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, CornerUL, CornerUR, CornerLL, CornerLR, step, step);

    Array1DDouble uaxis0, vaxis0;
    uaxis0.resize(geoarea.GetUaxisCompositePtsnb(0));
    vaxis0.resize(geoarea.GetVaxisCompositePtsnb(0));

    uaxis0 = geoarea.GetUaxisComposite(0);
    vaxis0 = geoarea.GetVaxisComposite(0);

    CHECK_CLOSE(0, uaxis0(0), 0.000001);
    CHECK_CLOSE(2.5, uaxis0(1), 0.000001);
    CHECK_CLOSE(5, uaxis0(2), 0.000001);
    CHECK_CLOSE(7.5, uaxis0(3), 0.000001);
    CHECK_CLOSE(10, uaxis0(4), 0.000001);
    CHECK_CLOSE(30, vaxis0(0), 0.000001);
    CHECK_CLOSE(32.5, vaxis0(1), 0.000001);
    CHECK_CLOSE(40, vaxis0(4), 0.000001);

    Array1DDouble uaxis1, vaxis1;
    uaxis1.resize(geoarea.GetUaxisCompositePtsnb(1));
    vaxis1.resize(geoarea.GetVaxisCompositePtsnb(1));

    uaxis1 = geoarea.GetUaxisComposite(1);
    vaxis1 = geoarea.GetVaxisComposite(1);

    CHECK_CLOSE(320, uaxis1(0), 0.000001);
    CHECK_CLOSE(322.5, uaxis1(1), 0.000001);
    CHECK_CLOSE(325, uaxis1(2), 0.000001);
    CHECK_CLOSE(360, uaxis1(geoarea.GetUaxisCompositePtsnb(1)-1), 0.000001);
    CHECK_CLOSE(30, vaxis1(0), 0.000001);
    CHECK_CLOSE(32.5, vaxis1(1), 0.000001);
    CHECK_CLOSE(40, vaxis1(4), 0.000001);
}

TEST(GetUVaxisCompositeSize)
{
    double Umin = -40;
    double Uwidth = 50;
    double Vmin = 30;
    double Vwidth = 10;
    double step = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, step, Vmin, Vwidth, step);

    CHECK_CLOSE(5, geoarea.GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(17, geoarea.GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisCompositePtsnb(1), 0.01);
}

TEST(GetUVaxisCompositeSizeStepLon)
{
    double Umin = -40;
    double Uwidth = 50;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(3, geoarea.GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(9, geoarea.GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisCompositePtsnb(1), 0.01);
}

TEST(GetUVaxisCompositeSizeStepLonMoved)
{
    double Umin = -7.5;
    double Uwidth = 15;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(2, geoarea.GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(2, geoarea.GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisCompositePtsnb(1), 0.01);
}

TEST(GetUVaxisCompositeWidthStepLonMoved)
{
    double Umin = -7.5;
    double Uwidth = 15;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(7.5, geoarea.GetUaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(7.5, geoarea.GetUaxisCompositeWidth(1), 0.01);
    CHECK_CLOSE(10, geoarea.GetVaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(10, geoarea.GetVaxisCompositeWidth(1), 0.01);
}

TEST(GetUVaxisPtsnbStepLonMoved)
{
    double Umin = -7.5;
    double Uwidth = 15;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(4, geoarea.GetUaxisPtsnb(), 0.01);
    CHECK_CLOSE(5, geoarea.GetVaxisPtsnb(), 0.01);
}

TEST(GetUVaxisWidthStepLonMoved)
{
    double Umin = -7.5;
    double Uwidth = 15;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(15, geoarea.GetUaxisWidth(), 0.01);
    CHECK_CLOSE(10, geoarea.GetVaxisWidth(), 0.01);
}

TEST(GetUVaxisCompositeLimits)
{
    double Umin = -10;
    double Uwidth = 20;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(0, geoarea.GetUaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(350, geoarea.GetUaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea.GetVaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea.GetVaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(10, geoarea.GetUaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360, geoarea.GetUaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea.GetVaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea.GetVaxisCompositeEnd(1), 0.01);
}

TEST(GetUVaxisCompositeLimitsMoved)
{
    double Umin = -7.5;
    double Uwidth = 15;
    double Vmin = 30;
    double Vwidth = 10;
    double Ustep = 5;
    double Vstep = 2.5;
    asGeoAreaCompositeRegularGrid geoarea(WGS84, Umin, Uwidth, Ustep, Vmin, Vwidth, Vstep);

    CHECK_CLOSE(2.5, geoarea.GetUaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(360-7.5, geoarea.GetUaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea.GetVaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea.GetVaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(7.5, geoarea.GetUaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360-2.5, geoarea.GetUaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea.GetVaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea.GetVaxisCompositeEnd(1), 0.01);
}

}
