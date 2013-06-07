#include "asGlobVars.h"
#include "include_tests.h"
#include "UnitTest++.h"
#include <wx/app.h>

#ifndef UNIT_TESTING
    #define UNIT_TESTING
#endif

int main()
{
    g_UnitTesting = true;
    g_SilentMode = true;

    // Option to test or not the exception throwing
    g_UnitTestExceptions = false;

    // Option to process time demanding processing
    g_UnitTestLongerProcessing = false;
    g_UnitTestLongestProcessing = false;
    if(g_UnitTestLongestProcessing) g_UnitTestLongerProcessing = true;

    // Test random distribution: write ouput in files
    g_UnitTestRandomDistributions = false;

    wxInitialize(); // Initialize the library because wxApp is not called

    bool result = UnitTest::RunAllTests();

    wxUninitialize();

    return result;
}
