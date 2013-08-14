#include "include_tests.h"
#include "asGeoAreaCompositeGrid.h"

#include "UnitTest++.h"

namespace
{

TEST(ConstructorAlternativeOneArea)
{
    double Umin = 10;
    int Uptsnb = 5;
    double Vmin = 30;
    int Vptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(1, geoarea->GetNbComposites());
    wxDELETE(geoarea);
}

TEST(ConstructorAlternativeTwoAreas)
{
    double Umin = -10;
    int Uptsnb = 13;
    double Vmin = 30;
    int Vptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_EQUAL(2, geoarea->GetNbComposites());
    wxDELETE(geoarea);
}

TEST(CheckConsistency)
{
    double Umin = -5;
    int Uptsnb = 11;
    double Vmin = 30;
    int Vptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(355, geoarea->GetCornerUL().u, 0.01);
    CHECK_CLOSE(355, geoarea->GetCornerLL().u, 0.01);
    CHECK_CLOSE(20, geoarea->GetCornerUR().u, 0.01);
    CHECK_CLOSE(20, geoarea->GetCornerLR().u, 0.01);
    wxDELETE(geoarea);
}

TEST(CheckConsistencyException)
{
    double Umin = 10;
    int Uptsnb = 1;
    double Vmin = 40;
    int Vptsnb = -5;
    double step = 2.5;
    wxString gridType = "Regular";

    if(g_UnitTestExceptions)
    {
        asGeoAreaCompositeGrid* geoarea = NULL;
        CHECK_THROW(geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step), asException);
        wxDELETE(geoarea);
    }
}

TEST(GetBoundsSplitted)
{
    double Umin = -10;
    int Uptsnb = 13;
    double Vmin = 30;
    int Vptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(0, geoarea->GetUmin(), 0.01);
    CHECK_CLOSE(30, geoarea->GetVmin(), 0.01);
    CHECK_CLOSE(360, geoarea->GetUmax(), 0.01);
    CHECK_CLOSE(40, geoarea->GetVmax(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeSize)
{
    double Umin = -40;
    int Uptsnb = 21;
    double Vmin = 30;
    int Vptsnb = 5;
    double step = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, step, Vmin, Vptsnb, step);

    CHECK_CLOSE(5, geoarea->GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(17, geoarea->GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeSizeStepLon)
{
    double Umin = -40;
    int Uptsnb = 11;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(3, geoarea->GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(9, geoarea->GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeSizeStepLonMoved)
{
    double Umin = -7.5;
    int Uptsnb = 4;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(2, geoarea->GetUaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(2, geoarea->GetUaxisCompositePtsnb(1), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(0), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisCompositePtsnb(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeWidthStepLonMoved)
{
    double Umin = -7.5;
    int Uptsnb = 4;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(7.5, geoarea->GetUaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(7.5, geoarea->GetUaxisCompositeWidth(1), 0.01);
    CHECK_CLOSE(10, geoarea->GetVaxisCompositeWidth(0), 0.01);
    CHECK_CLOSE(10, geoarea->GetVaxisCompositeWidth(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisPtsnbStepLonMoved)
{
    double Umin = -7.5;
    int Uptsnb = 4;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(4, geoarea->GetUaxisPtsnb(), 0.01);
    CHECK_CLOSE(5, geoarea->GetVaxisPtsnb(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisWidthStepLonMoved)
{
    double Umin = -7.5;
    int Uptsnb = 4;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(15, geoarea->GetUaxisWidth(), 0.01);
    CHECK_CLOSE(10, geoarea->GetVaxisWidth(), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeLimits)
{
    double Umin = -10;
    int Uptsnb = 5;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(0, geoarea->GetUaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(350, geoarea->GetUaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea->GetVaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea->GetVaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(10, geoarea->GetUaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360, geoarea->GetUaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea->GetVaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea->GetVaxisCompositeEnd(1), 0.01);
    wxDELETE(geoarea);
}

TEST(GetUVaxisCompositeLimitsMoved)
{
    double Umin = -7.5;
    int Uptsnb = 4;
    double Vmin = 30;
    int Vptsnb = 5;
    double Ustep = 5;
    double Vstep = 2.5;
    wxString gridType = "Regular";
    asGeoAreaCompositeGrid* geoarea = asGeoAreaCompositeGrid::GetInstance(WGS84, gridType, Umin, Uptsnb, Ustep, Vmin, Vptsnb, Vstep);

    CHECK_CLOSE(2.5, geoarea->GetUaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(360-7.5, geoarea->GetUaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(30, geoarea->GetVaxisCompositeStart(0), 0.01);
    CHECK_CLOSE(30, geoarea->GetVaxisCompositeStart(1), 0.01);
    CHECK_CLOSE(7.5, geoarea->GetUaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(360-2.5, geoarea->GetUaxisCompositeEnd(1), 0.01);
    CHECK_CLOSE(40, geoarea->GetVaxisCompositeEnd(0), 0.01);
    CHECK_CLOSE(40, geoarea->GetVaxisCompositeEnd(1), 0.01);
    wxDELETE(geoarea);
}

}
