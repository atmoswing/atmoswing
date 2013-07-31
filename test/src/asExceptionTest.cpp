#include "include_tests.h"
#include "asException.h"

#include "UnitTest++.h"

namespace
{

TEST(asThrowException)
{
    if(g_UnitTestExceptions)
    {
        CHECK_THROW(asThrowException("My exception"),asException);
    }
}

}
