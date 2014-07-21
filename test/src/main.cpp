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

#include "asGlobVars.h"
#include "include_tests.h"
#include "UnitTest++.h"
#include <TestReporterStdout.h>
#include <wx/app.h>
#include <wx/filename.h>

#ifndef UNIT_TESTING
    #define UNIT_TESTING
#endif

/*
 * Provide tests names as arguments in order to test specific tests. Otherwise process all tests.
 * Examples:
 * ./atmoswing-tests IsRoundFloatTrue LoadCatalogProp -> test specific tests
 * ./atmoswing-tests quick/short/fast -> test only the fast ones (not the method calibration)
 * ./atmoswing-tests -> test everything
 */

int main( int argc, char** argv )
{

    /*
    // In order to debug just one test
    argc=2;
    char* chars = "GrenobleComparison1ProcessingMethodCuda";
    argv[1] = chars;
    */

    // Override some globals
    g_UnitTesting = true;
    g_SilentMode = true;
    g_GuiMode = false;

    // Option to test or not the exception throwing
    g_UnitTestExceptions = false;

    // Test random distribution: write ouput in files
    g_UnitTestRandomDistributions = false;

    // Initialize the library because wxApp is not called
    wxInitialize();

    // Set the log
    Log().CreateFile("AtmoSwingUnitTest.log");
    Log().SetTarget(asLog::Both);
    Log().SetLevel(2);
    Log().DisableMessageBoxOnError();

    // Set the local config object
    wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,asConfig::GetTempDir()+"AtmoSwing.ini",asConfig::GetTempDir()+"AtmoSwing.ini",wxCONFIG_USE_LOCAL_FILE);
    wxFileConfig::Set(pConfig);

	// Check path
	wxString filepath = wxFileName::GetCwd();
	wxString filepath1 = filepath;
    filepath1.Append("/test/files");
	if (!wxFileName::DirExists(filepath1))
	{
		wxString filepath2 = filepath;
		filepath2.Append("/../test/files");
		if (wxFileName::DirExists(filepath2))
		{
			filepath.Append("/../test");
			wxSetWorkingDirectory(filepath);
		}
		else
		{
			wxString filepath3 = filepath;
			filepath3.Append("/../../test/files");
			if (wxFileName::DirExists(filepath3))
			{
				filepath.Append("/../../test");
				wxSetWorkingDirectory(filepath);
			}
			else
			{
				wxString str("Cannot find the files directory\n");
				printf("%s", str.mb_str(wxConvUTF8).data());
				return 0;
			}
		}
    }

    // Process only the selected tests or all of them is none is selected
    // from http://stackoverflow.com/questions/3546054/how-do-i-run-a-single-test-with-unittest
    int result;
    if( argc > 1 )
    {
        // If first arg is quick/short/fast, we only process the fast ones
        bool shortOnly = false;
        if(strcmp( "quick", argv[ 1 ] ) == 0) shortOnly = true;
        if(strcmp( "short", argv[ 1 ] ) == 0) shortOnly = true;
        if(strcmp( "fast", argv[ 1 ] ) == 0) shortOnly = true;
        if (shortOnly)
        {
            // Option to process time demanding processing
            g_UnitTestLongProcessing = false;

            result = UnitTest::RunAllTests();
        }
        else
        {
            // If first arg is "suite", we search for suite names instead of test names
            const bool suite = strcmp( "suite", argv[ 1 ] ) == 0;

            // Walk list of all tests, add those with a name that matches one of the arguments to a new TestList
            const UnitTest::TestList& allTests( UnitTest::Test::GetTestList() );
            UnitTest::TestList selectedTests;
            UnitTest::Test* p = allTests.GetHead();
            while( p )
            {
                for( int i=1 ; i<argc ; ++i )
                {
                    if( strcmp( suite ? p->m_details.suiteName
                                    : p->m_details.testName, argv[ i ] ) == 0 )
                    {
                        selectedTests.Add( p );
                    }
                }
                p = p->m_nextTest;
            }

            // To close the queue (otherwise it continues on the allTests list)
            selectedTests.Add( 0 );

            //run selected test(s) only
            UnitTest::TestReporterStdout reporter;
            UnitTest::TestRunner runner( reporter );
            result = runner.RunTestsIf( selectedTests, 0, UnitTest::True(), 0 );
        }
    }
    else
    {
        result = UnitTest::RunAllTests();
    }

    wxUninitialize();
    DeleteThreadsManager();
    DeleteLog();
    delete wxFileConfig::Set((wxFileConfig *) NULL);

    return result;
}
