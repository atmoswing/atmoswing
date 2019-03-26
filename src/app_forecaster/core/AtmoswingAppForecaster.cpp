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

IMPLEMENT_APP(AtmoswingAppForecaster)

#include "asInternet.h"
#include "asFileAscii.h"
#include "asBatchForecasts.h"
#include "asMethodForecasting.h"

#if wxUSE_GUI

#include "images.h"

#endif

static const wxCmdLineEntryDesc g_cmdLineDesc[] =
{
    {wxCMD_LINE_SWITCH, "h",  "help",          "This help text"},
    {wxCMD_LINE_SWITCH, "c",  "config",        "Configure the forecaster"},
    {wxCMD_LINE_SWITCH, "v",  "version",       "Show version number and quit"},
    {wxCMD_LINE_OPTION, "f",  "batch-file",    "Batch file to use for the forecast (full path)"},
    {wxCMD_LINE_SWITCH, "n",  "forecast-now",  "Run forecast for the latest available data"},
    {wxCMD_LINE_OPTION, "p",  "forecast-past", "Run forecast for the given number of past days"},
    {wxCMD_LINE_OPTION, "d",  "forecast-date", "YYYYMMDDHH Run forecast for a specified date"},
    {wxCMD_LINE_OPTION, "l",  "log-level",     "Set a log level"
                                               "\n \t\t\t\t\t - 1: errors"
                                               "\n \t\t\t\t\t - 2: warnings"
                                               "\n \t\t\t\t\t - 3: verbose"
                                               "\n \t\t\t\t\t - 4: debug"},
    {wxCMD_LINE_OPTION, NULL, "proxy",         "HOST[:PORT] Use proxy on given port"},
    {wxCMD_LINE_OPTION, NULL, "proxy-user",    "USER[:PASSWORD] Proxy user and password"},
    {wxCMD_LINE_PARAM,  NULL, NULL,            "batch file", wxCMD_LINE_VAL_STRING, wxCMD_LINE_PARAM_OPTIONAL},
    {wxCMD_LINE_NONE}};

static const wxString cmdLineLogo = wxT("\n"\
"_________________________________________\n"\
"____ ___ _  _ ____ ____ _ _ _ _ _  _ ____ \n"\
"|__|  |  |\\/| |  | [__  | | | | |\\ | | __ \n"\
"|  |  |  |  | |__| ___] |_|_| | | \\| |__] \n"\
"_________________________________________\n"\
"\n");

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
    userDir.Mkdir(wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL);

    g_guiMode = true;
    m_doConfig = false;
    m_doForecast = false;
    m_doForecastPast = false;
    m_forecastDate = 0.0;
    m_forecastPastDays = 0;

    // Set the local config object
    wxFileConfig *pConfig = new wxFileConfig("AtmoSwing", wxEmptyString,
                                             asConfig::GetUserDataDir() + "AtmoSwingForecaster.ini",
                                             asConfig::GetUserDataDir() + "AtmoSwingForecaster.ini",
                                             wxCONFIG_USE_LOCAL_FILE);
    wxFileConfig::Set(pConfig);

#if wxUSE_GUI

    // Set PPI
    wxMemoryDC dcTestPpi;
    wxSize ppiDC = dcTestPpi.GetPPI();
    g_ppiScaleDc = wxMax(static_cast<double>(ppiDC.x) / 96.0, 1.0);

    // Check that it is the unique instance
    bool multipleInstances;
    pConfig->Read("/General/MultiInstances", &multipleInstances, false);

    if (!multipleInstances) {
        const wxString instanceName = wxString::Format(wxT("AtmoSwingForecaster-%s"), wxGetUserId());
        m_singleInstanceChecker = new wxSingleInstanceChecker(instanceName);
        if (m_singleInstanceChecker->IsAnotherRunning()) {
            wxMessageBox(_("Program already running, aborting."));

            // Cleanup
            delete wxFileConfig::Set((wxFileConfig *) nullptr);
            DeleteThreadsManager();
            DeleteLog();
            delete m_singleInstanceChecker;

            return false;
        }
    }
#else
    g_guiMode = false;
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
    initialize_images(g_ppiScaleDc);

    // Create frame
    AtmoswingFrameForecaster *frame = new AtmoswingFrameForecaster(0L);
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
    g_unitTesting = false;
    g_silentMode = true;

    // Set log level
    if (logLevel < 0) {
        logLevel = wxFileConfig::Get()->Read("/General/LogLevel", 2l);
    }
    Log().CreateFileOnly("AtmoSwingForecaster.log");
    Log().SetLevel((int) logLevel);

    return true;
}

void AtmoswingAppForecaster::OnInitCmdLine(wxCmdLineParser &parser)
{
    parser.SetDesc(g_cmdLineDesc);
    parser.SetLogo(cmdLineLogo);

    // Must refuse '/' as parameter starter or cannot use "/path" style paths
    parser.SetSwitchChars(wxT("-"));
}

bool AtmoswingAppForecaster::OnCmdLineParsed(wxCmdLineParser &parser)
{
    // From http://wiki.wxwidgets.org/Command-Line_Arguments

    // Check if the user asked for command-line help
    if (parser.Found("help")) {
        parser.Usage();

        return false;
    }

    // Check if the user asked for the version
    if (parser.Found("version")) {
        wxString date(wxString::FromAscii(__DATE__));
        wxPrintf("AtmoSwing version %s, %s\n", g_version, (const wxChar *) date);

        return false;
    }

    // Check if the user asked to configure
    if (parser.Found("config")) {
        InitForCmdLineOnly(2);
        m_doConfig = true;

        return false;
    }

    // Check for a log level option
    wxString logLevelStr = wxEmptyString;
    long logLevel = -1;
    if (parser.Found("log-level", &logLevelStr)) {
        if (logLevelStr.ToLong(&logLevel)) {
            if (logLevel == 0) {
                Log().SetLevel(0);
            } else if (logLevel == 1) {
                Log().SetLevel(1);
            } else if (logLevel == 2) {
                Log().SetLevel(2);
            } else if (logLevel == 3) {
                Log().SetLevel(3);
            } else {
                Log().SetLevel(2);
                wxPrintf(_("The given log level (%s) does not correspond to any possible option (0-3).\n"), logLevelStr);
            }
        } else {
            Log().SetLevel(2);
        }

    }

    // Proxy activated
    wxString proxy;
    if (parser.Found("proxy", &proxy)) {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Internet/UsesProxy", true);

        wxString proxyAddress = proxy.BeforeFirst(':');
        wxString proxyPort = proxy.AfterFirst(':');

        pConfig->Write("/Internet/ProxyAddress", proxyAddress);
        pConfig->Write("/Internet/ProxyPort", proxyPort);
    } else {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Internet/UsesProxy", false);
    }

    // Proxy user
    wxString proxyUserPwd;
    if (parser.Found("proxy-user", &proxyUserPwd)) {
        wxString proxyUser = proxyUserPwd.BeforeFirst(':');
        wxString proxyPwd = proxyUserPwd.AfterFirst(':');

        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/Internet/ProxyUser", proxyUser);
        pConfig->Write("/Internet/ProxyPasswd", proxyPwd);
    }

    // Batch file to use for the forecast
    wxString batchFile;
    if (parser.Found("batch-file", &batchFile)) {
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", batchFile);
    }

    // Check for input files
    if (parser.GetParamCount() > 0) {
        InitForCmdLineOnly(logLevel);

        g_cmdFilename = parser.GetParam(0);

        // Under Windows when invoking via a document in Explorer, we are passed the short form.
        // So normalize and make the long form.
        wxFileName fName(g_cmdFilename);
        fName.Normalize(wxPATH_NORM_LONG | wxPATH_NORM_DOTS | wxPATH_NORM_TILDE | wxPATH_NORM_ABSOLUTE);
        g_cmdFilename = fName.GetFullPath();

        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", g_cmdFilename);
    }

    // Check for a present forecast
    if (parser.Found("forecast-now")) {
        InitForCmdLineOnly(logLevel);
        m_doForecast = true;
        m_forecastDate = asTime::NowMJD();

        return true;
    }

    // Check for a past forecast option
    wxString numberOfDaysStr = wxEmptyString;
    if (parser.Found("forecast-past", &numberOfDaysStr)) {
        long numberOfDays;
        if (!numberOfDaysStr.ToLong(&numberOfDays)) {
            wxPrintf(_("The value provided for 'forecast-past' could not be interpreted.\n"));
            return false;
        }

        InitForCmdLineOnly(logLevel);
        m_doForecastPast = true;
        m_forecastPastDays = (int) numberOfDays;

        return true;
    }

    // Check for a forecast date option
    wxString dateForecastStr = wxEmptyString;
    if (parser.Found("forecast-date", &dateForecastStr)) {
        InitForCmdLineOnly(logLevel);
        m_doForecast = true;
        m_forecastDate = asTime::GetTimeFromString(dateForecastStr, YYYYMMDDhh);

        return true;
    }

    // Finally, if no option is given in CL mode, display help.
    if (!g_guiMode) {
        parser.Usage();

        return false;
    }

    return true;
}

int AtmoswingAppForecaster::OnRun()
{
    if (!g_guiMode) {
        if (m_doConfig) {

#if wxUSE_GUI
            wxPrintf(_("This configuration mode is only available when AtmoSwing is built as a console application. Please use the GUI instead.\n"));
            return 1;
#else

            asBatchForecasts batchForecasts;

            // Load batch file if exists
            wxConfigBase *pConfig = wxFileConfig::Get();
            wxString batchFilePath = wxEmptyString;
            pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

            if (!batchFilePath.IsEmpty()) {
                if (batchForecasts.Load(batchFilePath)) {
                    std::cout << _("An existing batch file was found and has been loaded.\n");
                } else {
                    batchFilePath = wxEmptyString;
                }
            }

            // User inputs
            std::string stdinVal;
            wxString wxinVal;

            // Batch file path
            std::cout << _("Please provide a path to save the batch file.\n");
            std::cout << _("Current value (enter to keep): ") << batchForecasts.GetFilePath() << "\n";
            std::cout << _("New value: ");
            std::getline(std::cin, stdinVal);
            wxinVal = wxString(stdinVal);
            if (!wxinVal.IsEmpty()) {
                batchForecasts.SetFilePath(wxinVal);
            }
            std::cout << "\n";

            // Check if exists and load
            if (wxFile::Exists(batchForecasts.GetFilePath())) {
                std::cout << _("The batch file exists and will be loaded.\n");

                if (batchForecasts.Load(batchForecasts.GetFilePath())) {
                    std::cout << _("Failed opening the batch file.\n");
                }
            }

            // Directory to save the forecasts
            std::cout << _("Please provide a directory to save the forecasts.\n");
            std::cout << _("Current value (enter to keep): ") << batchForecasts.GetForecastsOutputDirectory() << "\n";
            std::cout << _("New value: ");
            std::getline(std::cin, stdinVal);
            wxinVal = wxString(stdinVal);
            if (!wxinVal.IsEmpty()) {
                batchForecasts.SetForecastsOutputDirectory(wxinVal);
            }
            std::cout << "\n";

            // Exports
            std::cout << _("Do you want to export synthetic xml files of the forecasts ?\n");
            wxString currVal;
            if (batchForecasts.ExportSyntheticXml()) {
                currVal = "Y";
            } else {
                currVal = "N";
            }
            std::cout << _("Current value (enter to keep): ") << currVal << "\n";
            bool acceptValue = false;
            while (!acceptValue) {
                std::cout << _("New value (Y/N): ");
                std::getline(std::cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (!wxinVal.IsEmpty()) {
                    if (wxinVal.IsSameAs("Y", false)) {
                        batchForecasts.SetExportSyntheticXml(true);
                        acceptValue = true;
                    } else if (wxinVal.IsSameAs("N", false)) {
                        batchForecasts.SetExportSyntheticXml(false);
                        acceptValue = true;
                    } else {
                        std::cout << _("The provided value is not allowed. Please enter Y or N.\n");
                    }
                }
            }
            std::cout << "\n";

            // Directory to save the exports
            if (batchForecasts.HasExports()) {
                std::cout << _("Please provide a directory to save the exports.\n");
                std::cout << _("Current value (enter to keep): ") << batchForecasts.GetExportsOutputDirectory() << "\n";
                std::cout << _("New value: ");
                std::getline(std::cin, stdinVal);
                wxinVal = wxString(stdinVal);
                if (!wxinVal.IsEmpty()) {
                    batchForecasts.SetExportsOutputDirectory(wxinVal);
                }
                std::cout << "\n";
            }

            // Directory containing the parameters files
            std::cout << _("Please provide the directory containing the parameters files.\n");
            std::cout << _("Current value (enter to keep): ") << batchForecasts.GetParametersFileDirectory() << "\n";
            std::cout << _("New value: ");
            std::getline(std::cin, stdinVal);
            wxinVal = wxString(stdinVal);
            if (!wxinVal.IsEmpty()) {
                batchForecasts.SetParametersFileDirectory(wxinVal);
            }
            std::cout << "\n";

            // Directory containing the archive predictors
            std::cout << _("Please provide the directory containing the archive predictors.\n");
            std::cout << _("Current value (enter to keep): ") << batchForecasts.GetPredictorsArchiveDirectory() << "\n";
            std::cout << _("New value: ");
            std::getline(std::cin, stdinVal);
            wxinVal = wxString(stdinVal);
            if (!wxinVal.IsEmpty()) {
                batchForecasts.SetPredictorsArchiveDirectory(wxinVal);
            }
            std::cout << "\n";

            // Directory to save the downloaded predictors
            std::cout << _("Please provide a directory to save the downloaded predictors.\n");
            std::cout << _("Current value (enter to keep): ") << batchForecasts.GetPredictorsRealtimeDirectory() <<
            "\n";
            std::cout << _("New value: ");
            getline(std::cin, stdinVal);
            wxinVal = wxString(stdinVal);
            if (!wxinVal.IsEmpty()) {
                batchForecasts.SetPredictorsRealtimeDirectory(wxinVal);
            }
            std::cout << "\n";

            // Directory containing the predictand database
            std::cout << _("Please provide the directory containing the predictand database.\n");
            std::cout << _("Current value (enter to keep): ") << batchForecasts.GetPredictandDBDirectory() << "\n";
            std::cout << _("New value: ");
            std::getline(std::cin, stdinVal);
            wxinVal = wxString(stdinVal);
            if (!wxinVal.IsEmpty()) {
                batchForecasts.SetPredictandDBDirectory(wxinVal);
            }
            std::cout << "\n";

            batchForecasts.Save();
            std::cout << _("Batch file created successfully.\n");

            pConfig->Write("/BatchForecasts/LastOpened", batchForecasts.GetFilePath());

            // Check if any forecast exist
            if (batchForecasts.GetForecastsNb() == 0) {
                std::cout <<
                _("Warning: there is no forecast listed in the batch file. Please create the batch file on a version with the graphical interface or edit the generated file manually.\n");
            }

#endif
        }

        if (m_doForecast) {
            // Log message
            wxString forecastDateStr = asTime::GetStringTime(m_forecastDate, "DD.MM.YYYY hh:mm");
            wxLogMessage(_("Forecast started for the %s UTC"), forecastDateStr);
            wxPrintf("Forecast started for the %s UTC\n", forecastDateStr);

            // Open last batch file
            wxConfigBase *pConfig = wxFileConfig::Get();
            wxString batchFilePath = wxEmptyString;
            pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

            asBatchForecasts batchForecasts;

            if (!batchFilePath.IsEmpty()) {
                if (!batchForecasts.Load(batchFilePath)) {
                    wxLogWarning(_("Failed to open the batch file ") + batchFilePath);
                    wxPrintf(_("Failed to open the batch file %s\n"), batchFilePath);
                    return 1;
                }
            } else {
                wxLogError(_("Please run 'atmoswing-forecaster -c' first in order to configure."));
                wxPrintf(_("Please run 'atmoswing-forecaster -c' first in order to configure.\n"));
                return 1;
            }

            // Launch forecasting
            asMethodForecasting forecaster = asMethodForecasting(&batchForecasts);
            forecaster.SetForecastDate(m_forecastDate);
            if (!forecaster.Manager()) {
                wxLogError(_("Failed processing the forecast."));
                wxPrintf(_("Failed processing the forecast.\n"));
                return 1;
            }
            double realForecastDate = forecaster.GetForecastDate();

            // Log message
            wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
            wxLogMessage(_("Forecast processed for the date %s UTC"), realForecastDateStr);
            wxPrintf("Forecast processed for the date %s UTC\n", realForecastDateStr);

            // Write the resulting files path into a temp file.
            wxString tempFile = asConfig::GetTempDir() + "AtmoSwingForecastFilePaths.txt";
            asFileAscii filePaths(tempFile, asFile::Replace);
            vwxs filePathsVect = forecaster.GetResultsFilePaths();

            filePaths.Open();

            for (int i = 0; (unsigned) i < filePathsVect.size(); i++) {
                filePaths.AddLineContent(filePathsVect[i]);
            }
            filePaths.Close();
        }

        if (m_doForecastPast) {
            wxPrintf("Forecast started for the last %d days\n", m_forecastPastDays);

            // Open last batch file
            wxConfigBase *pConfig = wxFileConfig::Get();
            wxString batchFilePath = wxEmptyString;
            pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

            asBatchForecasts batchForecasts;

            if (!batchFilePath.IsEmpty()) {
                if (!batchForecasts.Load(batchFilePath)) {
                    wxLogWarning(_("Failed to open the batch file ") + batchFilePath);
                    wxPrintf(_("Failed to open the batch file %s\n"), batchFilePath);
                    return 1;
                }
            } else {
                wxLogError(_("Please run 'atmoswing-forecaster -c' first in order to configure."));
                wxPrintf(_("Please run 'atmoswing-forecaster -c' first in order to configure.\n"));
                return 1;
            }

            double now = asTime::NowMJD();
            double startDate = floor(now) + 23.0 / 24.0;
            double endDate = floor(now) - m_forecastPastDays;
            double increment = 1.0 / 24.0;

            for (double date = startDate; date >= endDate; date -= increment) {
                if (now < date) // ulterior to present
                    continue;

                // Log message
                wxString forecastDateStr = asTime::GetStringTime(date, "DD.MM.YYYY hh:mm");
                wxLogMessage(_("Forecast started for the %s UTC"), forecastDateStr);
                wxPrintf("Forecast started for the %s UTC\n", forecastDateStr);

                // Launch forecasting
                asMethodForecasting forecaster = asMethodForecasting(&batchForecasts);
                forecaster.SetForecastDate(date);
                if (!forecaster.Manager()) {
                    wxLogError(_("Failed processing the forecast."));
                    wxPrintf(_("Failed processing the forecast.\n"));
                    return 1;
                }
                double realForecastDate = forecaster.GetForecastDate();

                // Log message
                wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
                wxLogMessage(_("Forecast processed for the date %s UTC"), realForecastDateStr);
                wxPrintf("Forecast processed for the date %s UTC\n", realForecastDateStr);

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

    // Delete images
    cleanup_images();
#endif

    // Config file (from wxWidgets samples)
    delete wxFileConfig::Set((wxFileConfig *) nullptr);

    // Delete threads manager and log
    DeleteThreadsManager();
    DeleteLog();

    // Cleanup cURL
    asInternet::Cleanup();

    return 1;
}

