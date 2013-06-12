#include "include_tests.h"
#include "asGlobEnums.h"

#include "UnitTest++.h"

namespace
{






TEST(StringToCoordSysWGS84)
{
    wxString CoordSysChar = "WGS84";
    const CoordSys Result = asGlobEnums::StringToCoordSysEnum(CoordSysChar);
    const CoordSys Ref = WGS84;
    CHECK_EQUAL(Ref, Result);
}

TEST(StringToCoordSysCH1903p)
{
    wxString CoordSysChar = "CH1903p";
    const CoordSys Result = asGlobEnums::StringToCoordSysEnum(CoordSysChar);
    const CoordSys Ref = CH1903p;
    CHECK_EQUAL(Ref, Result);
}

TEST(StringToCoordSysException)
{
    wxString CoordSysChar = "wrongname";

    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asGlobEnums::StringToCoordSysEnum(CoordSysChar), asException);
    }
}

}
