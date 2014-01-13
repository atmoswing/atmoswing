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
#include <wx/app.h>
#include <wx/filename.h>

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

	// Check path
	wxString filepath = wxFileName::GetCwd();
	wxString filepath1 = filepath;
    filepath1.Append("/files");
	if (!wxFileName::DirExists(filepath1))
	{
		wxString filepath2 = filepath;
		filepath2.Append("/../files");
		if (wxFileName::DirExists(filepath2))
		{
			filepath.Append("/..");
			wxSetWorkingDirectory(filepath);
		}
		else
		{
			wxString filepath3 = filepath;
			filepath3.Append("/../../files");
			if (wxFileName::DirExists(filepath3))
			{
				filepath.Append("/../..");
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

    bool result = UnitTest::RunAllTests();

    wxUninitialize();

    return result;
}
