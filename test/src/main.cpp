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
    g_UnitTestLongerProcessing = true;
    g_UnitTestLongestProcessing = true;
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
