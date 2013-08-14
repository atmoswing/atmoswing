#include "include_tests.h"
#include "asGeo.h"

#include "UnitTest++.h"

namespace
{
/*
TEST(CheckPointWGS84True)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 10;
    Point.v = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UVMaxTrue)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 360;
    Point.v = 90;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UVMinTrue)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 0;
    Point.v = -90;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(true, Result);
}

TEST(CheckPointWGS84UTooHigh)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 360.1;
    Point.v = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(360.1, Point.u, 0.000001);
}

TEST(CheckPointWGS84UTooLow)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = -0.1;
    Point.v = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-0.1, Point.u, 0.000001);
}

TEST(CheckPointWGS84VTooHigh)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 10;
    Point.v = 90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(90.1, Point.v, 0.000001);
}

TEST(CheckPointWGS84VTooLow)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 10;
    Point.v = -90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_FORBIDEN);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-90.1, Point.v, 0.000001);
}

TEST(CheckPointWGS84UTooHighCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 360.1;
    Point.v = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(0.1, Point.u, 0.000001);
}

TEST(CheckPointWGS84UTooLowCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = -0.1;
    Point.v = 10;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(359.9, Point.u, 0.000001);
}

TEST(CheckPointWGS84VTooHighCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 10;
    Point.v = 90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(89.9, Point.v, 0.000001);
}

TEST(CheckPointWGS84VTooLowCorr)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 10;
    Point.v = -90.1;
    const bool Result = geo.CheckPoint(Point, asEDIT_ALLOWED);
    CHECK_EQUAL(false, Result);
    CHECK_CLOSE(-89.9, Point.v, 0.000001);
}

TEST(ProjTransformWGS84toCH1903)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 7.19;
    Point.v = 46.06;
    // http://topo.epfl.ch/misc/transcoco/transcoco_compute.php
    Coo Result = geo.ProjTransform(CH1903, Point);
    CHECK_CLOSE(580758.18, Result.u, 1);
    CHECK_CLOSE(100972.32, Result.v, 1);
}

TEST(ProjTransformWGS84toCH1903p)
{
    asGeo geo(WGS84);
    Coo Point;
    Point.u = 7.465270;
    Point.v = 46.877096;
    Coo Result = geo.ProjTransform(CH1903p, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(2602030.74, Result.u, 1);
    CHECK_CLOSE(1191775.03, Result.v, 1);
}

TEST(ProjTransformCH1903toWGS84)
{
    asGeo geo(CH1903);
    Coo Point;
    Point.u = 603460;
    Point.v = 201200;
    Coo Result = geo.ProjTransform(WGS84, Point);
    // http://topo.epfl.ch/misc/transcoco/transcoco_compute.php
    CHECK_CLOSE(7.484091, Result.u, 0.001);
    CHECK_CLOSE(46.961871, Result.v, 0.001);
}

TEST(ProjTransformCH1903toCH1903p)
{
    asGeo geo(CH1903);
    Coo Point;
    Point.u = 602030.68;
    Point.v = 191775.03;
    Coo Result = geo.ProjTransform(CH1903p, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(2602030.74, Result.u, 1);
    CHECK_CLOSE(1191775.03, Result.v, 1);
}

TEST(ProjTransformCH1903toWGS84High)
{
    asGeo geo(CH1903);
    Coo Point;
    Point.u = 803460;
    Point.v = 291200;
    Coo Result = geo.ProjTransform(WGS84, Point);
    // http://topo.epfl.ch/misc/transcoco/transcoco_compute.php
    CHECK_CLOSE(10.151699, Result.u, 0.001);
    CHECK_CLOSE(47.739712, Result.v, 0.001);
}

TEST(ProjTransformCH1903ptoWGS84)
{
    asGeo geo(CH1903p);
    Coo Point;
    Point.u = 2602030.74;
    Point.v = 1191775.03;
    Coo Result = geo.ProjTransform(WGS84, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(7.465270, Result.u, 0.001);
    CHECK_CLOSE(46.877096, Result.v, 0.001);
}

TEST(ProjTransformCH1903ptoCH1903)
{
    asGeo geo(CH1903p);
    Coo Point;
    Point.u = 2602030.74;
    Point.v = 1191775.03;
    Coo Result = geo.ProjTransform(CH1903, Point);
    // http://en.wikipedia.org/wiki/Swiss_coordinate_system
    CHECK_CLOSE(602030.68, Result.u, 1);
    CHECK_CLOSE(191775.03, Result.v, 1);
}

TEST(ProjTransformCH1903ptoCH1903pSame)
{
    asGeo geo(CH1903p);
    Coo Point;
    Point.u = 2803460;
    Point.v = 1291200;
    Coo Result = geo.ProjTransform(CH1903p, Point);
    CHECK_CLOSE(2803460, Result.u, 1);
    CHECK_CLOSE(1291200, Result.v, 1);
}
*/
}
