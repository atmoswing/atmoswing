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
    #include "img_treectrl.h"
    #include "img_logo.h"
#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
{
    { wxCMD_LINE_SWITCH, "h", "help", "displays help on the command line parameters" },
    { wxCMD_LINE_SWITCH, "c", "config", "configure the forecaster" },
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
    { wxCMD_LINE_OPTION, "f", "batchfile", "batch file to use for the forecast (full path)" },
    { wxCMD_LINE_NONE }
};

bool AtmoswingAppForecaster::OnInit()
{
    #if _DEBUG
		#ifdef __WXMSW__
            _CrtSetDbgFlag ( _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF );
        #endif
    #endif

    // Set application name and create user directory
    wxString appName = "AtmoSwing forecaster";
    wxApp::SetAppName(appName);
    wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir());
    userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

    g_guiMode = true;
    m_doConfig = false;
    m_doForecast = false;
    m_doForecastPast = false;
    m_forecastDate = 0.0;
    m_forecastPastDays = 0;

    // Set the local config object
    wxFileConfig *pConfig = new wxFileConfig("AtmoSwing",wxEmptyString,asConfig::GetUserDataDir()+"AtmoSwingForecaster.ini",asConfig::GetUserDataDir()+"AtmoSwingForecaster.ini",wxCONFIG_USE_LOCAL_FILE);
    wxFileConfig::Set(pConfig);

    #if wxUSE_GUI
        // Check that it is the unique instance
        bool multipleInstances;
        pConfig->Read("/General/MultiInstances", &multipleInstances, false);

        if (!multipleInstances)
        {
            const wxString instanceName = wxString::Format(wxT("AtmoSwingForecaster-%s"),wxGetUserId().c_str());
            m_singleInstanceChecker = new wxSingleInstanceChecker(instanceName);
            if ( m_singleInstanceChecker->IsAnotherRunning() )
            {
                wxMessageBox(_("Program already running, aborting."));

                // Cleanup
                delete wxFileConfig::Set((wxFileConfig *) NULL);
                DeleteThreadsManager();
                DeleteLog();
                delete m_singleInstanceChecker;

                return false;
            }
        }
    #endif

    // Init cURL
    asInternet::Init();

    // Call default behaviour (mandatory for command-line mode)
    if (!wxApp::OnInit()) // When false, we are in CL mode
        return true;

    #if wxUSE_GUI
        // Following for GUI only
        wxInitAllImageHandlers();

        // Initialize images
        initialize_images_bullets();
        initialize_images_toolbar();
        initialize_images_treectrl();
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
    g_guiMode = false;
    g_unitTesting = false;
    g_silentMode = true;

    // Set log level
    if (logLevel<0)
    {
        logLevel = wxFileConfig::Get()->Read("/General/LogLevel", 2l);
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
            msg.Printf("AtmoSwing version %s, %s", g_version.c_str(), (const wxChar*) date);

            msgOut->Printf( wxT("%s"), msg.c_str() );
        }
        else
        {
            wxFAIL_MSG( _("No wxMessageOutput object?") );
        }

        return false;
    }

    // Check if the user asked to configure
    if (parser.Found("c"))
    {
        InitForCmdLineOnly(2);
        m_doConfig = true;

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
    }

    // Batch file to use for the forecast
    wxString batchFile;
    if (parser.Found("f", &batchFile))
    {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", batchFile);
    }

    // Check for a present forecast
    if (parser.Found("fn"))
    {
        InitForCmdLineOnly(logLevel);
        m_doForecast = true;
        m_forecastDate = asTime::NowMJD();

        return false;
    }

    // Check for a past forecast option
    wxString numberOfDaysStr = wxEmptyString;
    if (parser.Found("fp", & numberOfDaysStr))
    {
        long numberOfDays;
        numberOfDaysStr.ToLong(&numberOfDays);

        InitForCmdLineOnly(logLevel);
        m_doForecastPast = true;
        m_forecastPastDays = (int)numberOfDays;

        return false;
    }

    // Check for a forecast date option
    wxString dateForecastStr = wxEmptyString;
    if (parser.Found("fd", & dateForecastStr))
    {
        InitForCmdLineOnly(logLevel);
        m_doForecast = true;
        m_forecastDate = asTime::GetTimeFromString(dateForecastStr, YYYYMMDDhh);

        return false;
    }

    return true;
}

int AtmoswingAppForecaster::OnRun()
{
    if (!g_guiMode)
    {
        if (m_doConfig)
        {
            wxMessageOutput* msgOut = wxMessageOutput::Get();
            if ( !msgOut )
            {
                wxFAIL_MSG( _("No wxMessageOutput object?") );
                return 0;
            }

            #if wxUSE_GUI

                msgOut->Printf(_("This configuration mode is only available when AtmoSwing is built as a console application. Please use the GUI instead."));
                return 0;

            #else

                asBatchForecasts batchForecasts;

                // Load batch file if exists
                wxConfigBase *pConfig = wxFileConfig::Get();
                wxString batchFilePath = wxEmptyString;
                pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

                if(!batchFilePath.IsEmpty())
                {
                    if (batchForecasts.Load(batchFilePath))
                    {
                        cout << _("An existing batch file was found and has been loaded.\n");
                    }
                    else
                    {
                        batchFilePath = wxEmptyString;
                    }
                }

                // User inputs
                std::string stdinVal;
                wxString wxinVal;

                // Batch file path
                cout << _("Please provide a path to save the batch file.\n");
                cout << _("Current value (enter to keep): ") << batchForecasts.GetFilePath().c_str() << "\n";
                cout << _("New value: ");
                getline (cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (wxinVal.IsEmpty())
                {
                    wxinVal = batchForecasts.GetFilePath();
                }
                batchForecasts.SetFilePath(wxinVal);
                cout << "\n";

                // Check if exists and load
                if (wxFile::Exists(batchForecasts.GetFilePath()))
                {
                    cout << _("The batch file exists and will be loaded.\n");

                    if (batchForecasts.Load(batchForecasts.GetFilePath()))
                    {
                        cout << _("Failed opening the batch file.\n");
                    }
                }

                // Directory to save the forecasts
                cout << _("Please provide a directory to save the forecasts.\n");
                cout << _("Current value (enter to keep): ") << batchForecasts.GetForecastsOutputDirectory().c_str() << "\n";
                cout << _("New value: ");
                getline (cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (wxinVal.IsEmpty())
                {
                    wxinVal = batchForecasts.GetForecastsOutputDirectory();
                }
                batchForecasts.SetForecastsOutputDirectory(wxinVal);
                cout << "\n";

                // Directory containing the parameters files
                cout << _("Please provide the directory containing the parameters files.\n");
                cout << _("Current value (enter to keep): ") << batchForecasts.GetParametersFileDirectory().c_str() << "\n";
                cout << _("New value: ");
                getline (cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (wxinVal.IsEmpty())
                {
                    wxinVal = batchForecasts.GetParametersFileDirectory();
                }
                batchForecasts.SetParametersFileDirectory(wxinVal);
                cout << "\n";

                // Directory containing the archive predictors
                cout << _("Please provide the directory containing the archive predictors.\n");
                cout << _("Current value (enter to keep): ") << batchForecasts.GetPredictorsArchiveDirectory().c_str() << "\n";
                cout << _("New value: ");
                getline (cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (wxinVal.IsEmpty())
                {
                    wxinVal = batchForecasts.GetPredictorsArchiveDirectory();
                }
                batchForecasts.SetPredictorsArchiveDirectory(wxinVal);
                cout << "\n";

                // Directory to save the downloaded predictors
                cout << _("Please provide a directory to save the downloaded predictors.\n");
                cout << _("Current value (enter to keep): ") << batchForecasts.GetPredictorsRealtimeDirectory().c_str() << "\n";
                cout << _("New value: ");
                getline (cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (wxinVal.IsEmpty())
                {
                    wxinVal = batchForecasts.GetPredictorsRealtimeDirectory();
                }
                batchForecasts.SetPredictorsRealtimeDirectory(wxinVal);
                cout << "\n";

                // Directory containing the predictand database
                cout << _("Please provide the directory containing the predictand database.\n");
                cout << _("Current value (enter to keep): ") << batchForecasts.GetPredictandDBDirectory().c_str() << "\n";
                cout << _("New value: ");
                getline (cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (wxinVal.IsEmpty())
                {
                    wxinVal = batchForecasts.GetPredictandDBDirectory();
                }
                batchForecasts.SetPredictandDBDirectory(wxinVal);
                cout << "\n";

                batchForecasts.Save();
                cout << _("Batch file created successfully.\n");
            
                pConfig->Write("/BatchForecasts/LastOpened", batchForecasts.GetFilePath());

                // Check if any forecast exist
                if(batchForecasts.GetForecastsNb()==0)
                {
                    cout << _("Warning: there is no forecast listed in the batch file. Please create the batch file on a version with the graphical interface or edit the generated file manually.\n");
                }

            #endif
        }

        if (m_doForecast)
        {
            wxMessageOutput* msgOut = wxMessageOutput::Get();

            // Log message
            wxString forecastDateStr = asTime::GetStringTime(m_forecastDate, "DD.MM.YYYY hh:mm");
            asLogMessageImportant(wxString::Format(_("Forecast started for the %s UTC"), forecastDateStr.c_str()));
            if ( msgOut )
            {
                msgOut->Printf( "Forecast started for the %s UTC", forecastDateStr.c_str() );
            }

            // Open last batch file
            wxConfigBase *pConfig = wxFileConfig::Get();
            wxString batchFilePath = wxEmptyString;
            pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

            asBatchForecasts batchForecasts;

            if(!batchFilePath.IsEmpty())
            {
                if (!batchForecasts.Load(batchFilePath))
                {
                    asLogWarning(_("Failed to open the batch file ") + batchFilePath);
                    if ( msgOut )
                    {
                        msgOut->Printf( _("Failed to open the batch file %s"), batchFilePath.c_str() );
                    }
                    return 0;
                }
            }
            else
            {
                asLogError(_("Please run 'atmoswing-forecaster -c' first in order to configure."));
                if ( msgOut )
                {
                    msgOut->Printf( _("Please run 'atmoswing-forecaster -c' first in order to configure."));
                }
                return 0;
            }

            // Launch forecasting
            asMethodForecasting forecaster = asMethodForecasting(&batchForecasts);
            forecaster.SetForecastDate(m_forecastDate);
            if (!forecaster.Manager())
            {
                asLogError(_("Failed processing the forecast."));
                if ( msgOut )
                {
                    msgOut->Printf(_("Failed processing the forecast."));
                }
                return 0;
            }
            double realForecastDate = forecaster.GetForecastDate();

            // Log message
            wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
            asLogMessageImportant(wxString::Format(_("Forecast processed for the date %s UTC"), realForecastDateStr.c_str()));
            if ( msgOut )
            {
                msgOut->Printf( "Forecast processed for the date %s UTC", realForecastDateStr.c_str() );
            }

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
        }

        if (m_doForecastPast)
        {
            wxMessageOutput* msgOut = wxMessageOutput::Get();
            if ( msgOut )
            {
                msgOut->Printf( "Forecast started for the last %d days", m_forecastPastDays );
            }

            // Open last batch file
            wxConfigBase *pConfig = wxFileConfig::Get();
            wxString batchFilePath = wxEmptyString;
            pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

            asBatchForecasts batchForecasts;

            if(!batchFilePath.IsEmpty())
            {
                if (!batchForecasts.Load(batchFilePath))
                {
                    asLogWarning(_("Failed to open the batch file ") + batchFilePath);
                    if ( msgOut )
                    {
                        msgOut->Printf( _("Failed to open the batch file %s"), batchFilePath.c_str() );
                    }
                    return 0;
                }
            }
            else
            {
                asLogError(_("Please run 'atmoswing-forecaster -c' first in order to configure."));
                if ( msgOut )
                {
                    msgOut->Printf( _("Please run 'atmoswing-forecaster -c' first in order to configure."));
                }
                return 0;
            }

            double now = asTime::NowMJD();
            double startDate = floor(now) + 23.0/24.0;
            double endDate = floor(now) - m_forecastPastDays;
            double increment = 1.0/24.0;

            for (double date=startDate; date>=endDate; date-=increment)
            {
                if (now<date) // ulterior to present
                    continue;

                // Log message
                wxString forecastDateStr = asTime::GetStringTime(date, "DD.MM.YYYY hh:mm");
                asLogMessageImportant(wxString::Format(_("Forecast started for the %s UTC"), forecastDateStr.c_str()));
                if ( msgOut )
                {
                    msgOut->Printf( "Forecast started for the %s UTC", forecastDateStr.c_str() );
                }

                // Launch forecasting
                asMethodForecasting forecaster = asMethodForecasting(&batchForecasts);
                forecaster.SetForecastDate(date);
                if (!forecaster.Manager())
                {
                    asLogError(_("Failed processing the forecast."));
                    if ( msgOut )
                    {
                        msgOut->Printf( _("Failed processing the forecast."));
                    }
                    return 0;
                }
                double realForecastDate = forecaster.GetForecastDate();

                // Log message
                wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
                asLogMessageImportant(wxString::Format(_("Forecast processed for the date %s UTC"), realForecastDateStr.c_str()));
                if ( msgOut )
                {
                    msgOut->Printf( "Forecast processed for the date %s UTC", realForecastDateStr.c_str() );
                }

                // Apply real forecast date to increment
                date = realForecastDate;
            }
        }

        return 0;
    }

    return wxApp::OnRun();
}

int AtmoswingAppForecaster::OnExit()
{
    #if wxUSE_GUI
        // Instance checker
        delete m_singleInstanceChecker;
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

