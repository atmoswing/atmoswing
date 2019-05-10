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
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 * Portions Copyright 2016 Pascal Horton, University of Bern.
 */

#include "asGlobVars.h"
#include "gtest/gtest.h"

#ifndef UNIT_TESTING
#define UNIT_TESTING
#endif

int main(int argc, char **argv)
{
    int resultTest = -2;

    wxPrintf("Tests starting.\n");
    wxPrintf("Present directory: %s\n", wxFileName::GetCwd());

    try {
        ::testing::InitGoogleTest(&argc, argv);

        // Override some globals
        g_unitTesting = true;
        g_silentMode = true;
        g_guiMode = false;

        // Initialize the library because wxApp is not called
        wxInitialize();

        // Set the log
        Log().CreateFile("AtmoSwingTests.log");
        Log().SetLevel(2);

        // Set the local config object
        wxFileConfig *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString, asConfig::GetTempDir() + "AtmoSwing.ini",
                                                 asConfig::GetTempDir() + "AtmoSwing.ini", wxCONFIG_USE_LOCAL_FILE);
        wxFileConfig::Set(pConfig);

        // Check path
        wxString filePath = wxFileName::GetCwd();
        wxString filePath1 = filePath;
        filePath1.Append("/tests/files");
        if (wxFileName::DirExists(filePath1)) {
            filePath.Append("/tests");
            wxSetWorkingDirectory(filePath);
        } else {
            wxString filePath2 = filePath;
            filePath2.Append("/../tests/files");
            if (wxFileName::DirExists(filePath2)) {
                filePath.Append("/../tests");
                wxSetWorkingDirectory(filePath);
            } else {
                wxString filePath3 = filePath;
                filePath3.Append("/../../tests/files");
                if (wxFileName::DirExists(filePath3)) {
                    filePath.Append("/../../tests");
                    wxSetWorkingDirectory(filePath);
                } else {
                    wxPrintf("Cannot find the files directory\n");
                    wxPrintf("Original working directory: %s\n", filePath);
                    return 0;
                }
            }
        }
        wxPrintf("Updated working directory: %s\n", wxFileName::GetCwd());

        resultTest = RUN_ALL_TESTS();

        // Cleanup
        wxUninitialize();
        DeleteThreadsManager();
        DeleteLog();
        delete wxFileConfig::Set((wxFileConfig *) nullptr);

    } catch (std::exception &e) {
        wxString msg(e.what(), wxConvUTF8);
        wxPrintf(_("Exception caught: %s\n"), msg);
    }

    return resultTest;
}
