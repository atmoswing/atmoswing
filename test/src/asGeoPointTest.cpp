#include "include_tests.h"
#include "asGeoPoint.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorDefault)
{
    Coo Point;
    Point.u = 7;
    Point.v = 46;
    asGeoPoint geopoint(WGS84, Point);

    CHECK_CLOSE(7, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ConstructorOther)
{
    double U = 7;
    double V = 46;
    asGeoPoint geopoint(WGS84, U, V);

    CHECK_CLOSE(7, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ConstructorOutBoundsLon)
{
    double U = -10;
    double V = 46;
    asGeoPoint geopoint(WGS84, U, V);

    CHECK_CLOSE(350, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ConstructorOutBoundsLat)
{
    double U = 10;
    double V = -100;
    asGeoPoint geopoint(WGS84, U, V);

    CHECK_CLOSE(190, geopoint.GetU(), 0.001);
    CHECK_CLOSE(-80, geopoint.GetV(), 0.001);
}

TEST(SetCooOutBounds)
{
    asGeoPoint geopoint(WGS84, 0, 0);
    Coo Point;
    Point.u = -10;
    Point.v = 46;
    geopoint.SetCoo(Point);

    CHECK_CLOSE(350, geopoint.GetU(), 0.001);
    CHECK_CLOSE(46, geopoint.GetV(), 0.001);
}

TEST(ProjConvert)
{
    double U = 10;
    double V = 48;
    asGeoPoint geopoint(WGS84, U, V);
    geopoint.ProjConvert(CH1903);

    CHECK_CLOSE(791142.61, geopoint.GetU(), 2);
    CHECK_CLOSE(319746.83, geopoint.GetV(), 2);
}

}
