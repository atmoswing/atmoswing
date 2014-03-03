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

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "AtmoswingAppForecaster.h"
#if wxUSE_GUI
    #include "AtmoswingMainForecaster.h"
#endif

IMPLEMENT_APP(AtmoswingAppForecaster);

#include <wx/debug.h>
#include "wx/fileconf.h"
#include "wx/cmdline.h"
#include <asIncludes.h>
#include <asThreadsManager.h>
#include <asInternet.h>
#include <asMethodForecasting.h>
#include <asFileAscii.h>
#if wxUSE_GUI
    #include "img_bullets.h"
    #include "img_toolbar.h"
    #include "img_logo.h"
#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
{
    { wxCMD_LINE_SWITCH, "h", "help", "displays help on the command line parameters" },
    { wxCMD_LINE_SWITCH, "v", "version", "print version" },
    { wxCMD_LINE_SWITCH, "s", "silent", "silent mode" },
    { wxCMD_LINE_OPTION, "ll", "loglevel", "set a log level"
                                "\n \t\t\t\t 0: minimum"
                                "\n \t\t\t\t 1: errors"
                                "\n \t\t\t\t 2: warnings"
                                "\n \t\t\t\t 3: verbose" },
    { wxCMD_LINE_OPTION, "lt", "logtarget", "set log target"
                                "\n \t\t\t\t file: file only"
                                "\n \t\t\t\t prompt: command prompt"
                                "\n \t\t\t\t both: command prompt and file (default)" },
    { wxCMD_LINE_SWITCH, "fn", "forecastnow", "run forecast for the latest available data" },
    { wxCMD_LINE_OPTION, "fp", "forecastpast", "run forecast for the given number of past days" },
    { wxCMD_LINE_OPTION, "fd", "forecastdate", "run forecast for a specified date (YYYYMMDDHH)" },
    { wxCMD_LINE_NONE }
};

bool AtmoswingAppForecaster::OnInit()
{
    #if _DEBUG
        _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
    #endif

    // Set application name and create user directory
    wxString appName = "AtmoSwing forecaster";
    wxApp::SetAppName(appName);
    wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
    userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

    g_AppViewer = false;
    g_AppForecaster = true;
    g_GuiMode = true;

    // Set the local config object
    wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,asConfig::GetUserDataDir()+"AtmoSwing.ini",asConfig::GetUserDataDir()+"AtmoSwing.ini",wxCONFIG_USE_LOCAL_FILE);
    wxFileConfig::Set(pConfig);

    #if wxUSE_GUI
        // Check that it is the unique instance
        bool multipleInstances;
        pConfig->Read("/Standard/MultiInstances", &multipleInstances, false);

        if (!multipleInstances)
        {
            const wxString instanceName = wxString::Format(wxT("AtmoSwingForecaster-%s"),wxGetUserId().c_str());
            m_SingleInstanceChecker = new wxSingleInstanceChecker(instanceName);
            if ( m_SingleInstanceChecker->IsAnotherRunning() )
            {
                wxMessageBox(_("Program already running, aborting."));
                
                // Cleanup
                delete wxFileConfig::Set((wxFileConfig *) NULL);
                DeleteThreadsManager();
                DeleteLog();
                delete m_SingleInstanceChecker;

                return false;
            }
        }
    #endif

    // Init cURL
    asInternet::Init();

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return false;

    #if wxUSE_GUI
        // Following for GUI only
        wxInitAllImageHandlers();

        // Initialize images
        initialize_images_bullets();
        initialize_images_toolbar();
        initialize_images_logo();

        // Create frame
        AtmoswingFrameForecaster* frame = new AtmoswingFrameForecaster(0L);
        frame->OnInit();

        #ifdef __WXMSW__
            frame->SetIcon(wxICON(myicon)); // To Set App Icon
        #endif
        frame->Show();
        SetTopWindow(frame);
    #endif

    return true;
}

bool AtmoswingAppForecaster::InitForCmdLineOnly(long logLevel)
{
    g_GuiMode = false;
    g_UnitTesting = false;
    g_SilentMode = true;

    // Set log level
    if (logLevel<0)
    {
        logLevel = wxFileConfig::Get()->Read("/Standard/LogLevel", 2l);
    }
    Log().CreateFileOnly("AtmoSwingForecaster.log");
    Log().SetLevel((int)logLevel);
    Log().DisableMessageBoxOnError();

    return true;
}

void AtmoswingAppForecaster::OnInitCmdLine(wxCmdLineParser& parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments
    parser.SetDesc (g_cmdLineDesc);
    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars (wxT("-"));
}

bool AtmoswingAppForecaster::OnCmdLineParsed(wxCmdLineParser& parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments

    // Check if the user asked for command-line help
    if (parser.Found("h"))
    {
        parser.Usage();
        return false;
    }

    // Check if the user asked for the version
    if (parser.Found("v"))
    {
        wxMessageOutput* msgOut = wxMessageOutput::Get();
        if ( msgOut )
        {
            wxString msg;
            wxString date(wxString::FromAscii(__DATE__));
            msg.Printf("AtmoSwing version %s, %s", g_Version.c_str(), (const wxChar*) date);

            msgOut->Printf( wxT("%s"), msg.c_str() );
        }
        else
        {
            wxFAIL_MSG( _("No wxMessageOutput object?") );
        }

        return false;
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    long logLevel = -1;
    if (parser.Found("ll", & logLevelStr))
    {
        logLevelStr.ToLong(&logLevel);

        // Check and apply
        if (logLevel==0)
        {
            Log().SetLevel(0);
        }
        else if (logLevel==1)
        {
            Log().SetLevel(1);
        }
        else if (logLevel==2)
        {
            Log().SetLevel(2);
        }
        else
        {
            Log().SetLevel(3);
        }
    }

    // Check for the log target option
    wxString logTargetStr = wxEmptyString;
    if (parser.Found("lt", & logTargetStr))
    {
        // Check and apply
        if (logTargetStr.IsSameAs("file", false))
        {
            Log().SetTarget(asLog::File);
        }
        else if (logTargetStr.IsSameAs("screen", false))
        {
            Log().SetTarget(asLog::Screen);
        }
        else if (logTargetStr.IsSameAs("both", false))
        {
            Log().SetTarget(asLog::Both);
        }
        else
        {
            Log().SetTarget(asLog::Both);

            wxMessageOutput* msgOut = wxMessageOutput::Get();
            if ( msgOut )
            {
                msgOut->Printf( _("The given log target (%s) does not correspond to any possible option."), logTargetStr.c_str() );
            }
        }
    }

    // Check if the user asked for the silent mode
    if (parser.Found("s"))
    {
        Log().SetTarget(asLog::File);
        return false;
    }

    // Check for a present forecast
    if (parser.Found("fn"))
    {
        InitForCmdLineOnly(logLevel);

        double forecastDate = asTime::NowMJD();

        // Log message
        wxString forecastDateStr = asTime::GetStringTime(forecastDate, "DD.MM.YYYY hh:mm");
        asLogMessageImportant(wxString::Format(_("Running the forecast for the date %s UTC"), forecastDateStr.c_str()));

        // Force use of default models list
        wxString defFilePath = asConfig::GetDefaultUserConfigDir() + "DefaultForecastingModelsList.xml";
        wxString curFilePath = asConfig::GetDefaultUserConfigDir() + "CurrentForecastingModelsList.xml";
        if (!wxFileName::FileExists(defFilePath))
        {
            asLogError(_("The default forecasting models list could not be found."));
            return false;
        }
        wxCopyFile(defFilePath,curFilePath);

        // Launch forecasting
        asMethodForecasting forecaster = asMethodForecasting();
        forecaster.SetForecastDate(forecastDate);
        if (!forecaster.Manager())
        {
            asLogError(_("Failed processing the forecast."));
            return false;
        }
        double realForecastDate = forecaster.GetForecastDate();

        // Log message
        wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
        asLogMessageImportant(wxString::Format(_("Forecast processed for the date %s UTC"), realForecastDateStr.c_str()));

        // Write the resulting files path into a temp file.
        wxString tempFile = asConfig::GetTempDir() + "AtmoSwingForecatsFilePaths.txt";
        asFileAscii filePaths(tempFile, asFile::Replace);
        VectorString filePathsVect = forecaster.GetResultsFilePaths();

        filePaths.Open();

        for (int i=0; (unsigned)i<filePathsVect.size(); i++)
        {
            filePaths.AddLineContent(filePathsVect[i]);
        }
        filePaths.Close();

        return false;
    }

    // Check for a past forecast option
    wxString numberOfDaysStr = wxEmptyString;
    if (parser.Found("fp", & numberOfDaysStr))
    {
        InitForCmdLineOnly(logLevel);

        wxMessageOutput* msgOut = wxMessageOutput::Get();
        if ( msgOut )
        {
            msgOut->Printf( "Forecast started for the %s past days", numberOfDaysStr.c_str() );
        }

        long numberOfDays;
        numberOfDaysStr.ToLong(&numberOfDays);

        double now = asTime::NowMJD();
        double startDate = floor(now) + 23.0/24.0;
        double endDate = floor(now) - numberOfDays;
        double increment = 1.0/24.0;

        for (double date=startDate; date>=endDate; date-=increment)
        {
            if (now<date) // ulterior to present
                continue;

            // Log message
            wxString forecastDateStr = asTime::GetStringTime(date, "DD.MM.YYYY hh:mm");
            asLogMessageImportant(wxString::Format(_("Running the forecast for the date %s UTC"), forecastDateStr.c_str()));

            // Launch forecasting
            asMethodForecasting forecaster = asMethodForecasting();
            forecaster.SetForecastDate(date);
            if (!forecaster.Manager())
            {
                asLogError(_("Failed processing the forecast."));
                return false;
            }
            double realForecastDate = forecaster.GetForecastDate();

            // Log message
            wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
            asLogMessageImportant(wxString::Format(_("Forecast processed for the date %s UTC"), realForecastDateStr.c_str()));

            // Apply real forecast date to increment
            date = realForecastDate;
        }

        return false;
    }

    // Check for a forecast date option
    wxString dateForecastStr = wxEmptyString;
    if (parser.Found("fd", & dateForecastStr))
    {
        InitForCmdLineOnly(logLevel);

        double forecastDate = asTime::GetTimeFromString(dateForecastStr, YYYYMMDDhh);

        wxString forecastDateStr = asTime::GetStringTime(forecastDate, "DD.MM.YYYY hh:mm");

        wxMessageOutput* msgOut = wxMessageOutput::Get();
        if ( msgOut )
        {
            msgOut->Printf( "Forecast started for the %s.", forecastDateStr.c_str() );
        }

        // Log message
        asLogMessageImportant(wxString::Format(_("Running the forecast for the date %s UTC"), forecastDateStr.c_str()));

        // Launch forecasting
        asMethodForecasting forecaster = asMethodForecasting();
        forecaster.SetForecastDate(forecastDate);
        if (!forecaster.Manager())
        {
            asLogError(_("Failed processing the forecast."));
            return false;
        }
        double realForecastDate = forecaster.GetForecastDate();

        // Log message
        wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
        asLogMessageImportant(wxString::Format(_("Forecast processed for the date %s UTC"), realForecastDateStr.c_str()));


        return false;
    }

    return true;
}

int AtmoswingAppForecaster::OnExit()
{
    #if wxUSE_GUI
        // Instance checker
        delete m_SingleInstanceChecker;
    #endif

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig *) NULL);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

    // Cleanup cURL
    asInternet::Cleanup();

    return 1;
}


