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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#ifdef WX_PRECOMP
#include "wx_pch.h"
#endif

#ifdef __BORLANDC__
#pragma hdrstop
#endif //__BORLANDC__

#include "AtmoswingMainCalibrator.h"

#include "asCatalog.h"
#include "asCatalogPredictands.h"
#include "asCatalogPredictors.h"
#include "asCatalogPredictorsArchive.h"
#include "asCatalogPredictorsRealtime.h"
#include "asGeo.h"
#include "asGeoArea.h"
#include "asGeoAreaCompositeGrid.h"
#include "asGeoPoint.h"
#include "asTime.h"
#include "asTimeArray.h"
#include "asDataPredictand.h"
#include "asDataPredictorArchive.h"
#include "asResults.h"
#include "asResultsAnalogsDates.h"
#include "asResultsAnalogsValues.h"
#include "asResultsAnalogsForecastScores.h"
#include "asResultsAnalogsForecastScoreFinal.h"
#include "asResultsAnalogsScoresMap.h"
#include "asFileNetcdf.h"
#include "asFileXml.h"
#include "asFileAscii.h"
#include "asFileDat.h"
#include "asPredictorCriteria.h"
#include "asProcessor.h"
#include "asConfig.h"


AtmoswingFrameCalibrator::AtmoswingFrameCalibrator(wxFrame *frame)
    : asFrameCalibration(frame)
{
#if wxUSE_STATUSBAR
    wxLogStatus(_("Welcome to Atmoswing %s."), asVersion::GetFullString().c_str());
#endif

    // Config file
    wxConfigBase *pConfig = wxFileConfig::Get();

    // Set default options
    SetDefaultOptions();

    // Create log window and file
    bool displayLogWindow;
    pConfig->Read("/Standard/DisplayLogWindow", &displayLogWindow, true);
    m_LogWindow = new asLogWindow(this, _("Atmoswing log window"), displayLogWindow);
    Log().CreateFile("AtmoswingCalibrator.log");

    // Restore frame position and size
    int minHeight = 600, minWidth = 500;
    int x = pConfig->Read("/MainFrameCalibrator/x", 50),
        y = pConfig->Read("/MainFrameCalibrator/y", 50),
        w = pConfig->Read("/MainFrameCalibrator/w", minWidth),
        h = pConfig->Read("/MainFrameCalibrator/h", minHeight);
    wxRect screen = wxGetClientDisplayRect();
    if (x<screen.x-10) x = screen.x;
    if (x>screen.width) x = screen.x;
    if (y<screen.y-10) y = screen.y;
    if (y>screen.height) y = screen.y;
    if (w+x>screen.width) w = screen.width-x;
    if (w<minWidth) w = minWidth;
    if (w+x>screen.width) x = screen.width-w;
    if (h+y>screen.height) h = screen.height-y;
    if (h<minHeight) h = minHeight;
    if (h+y>screen.height) y = screen.height-h;
    Move(x, y);
    SetClientSize(w, h);
    Fit();

    // Check that the paths to the catalogs are well defined
    bool doConfigure = false;
    wxString CatalogPredictorsArchiveFilePath = pConfig->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", wxEmptyString);
    wxString CatalogPredictorsRealtimeFilePath = pConfig->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", wxEmptyString);
    wxString CatalogPredictandsFilePath = pConfig->Read("/StandardPaths/CatalogPredictandsFilePath", wxEmptyString);

    if (CatalogPredictorsArchiveFilePath.IsEmpty())
    {
        doConfigure = true;
    }
    else
    {
        if (!wxFileName::FileExists(CatalogPredictorsArchiveFilePath))
        {
            asLogError(_("The path to the archive predictors catalog is not correct. Please give the path to the file under Options > Preferences > Paths"));
        }
    }

    if (CatalogPredictorsRealtimeFilePath.IsEmpty())
    {
        doConfigure = true;
    }
    else
    {
        if (!wxFileName::FileExists(CatalogPredictorsRealtimeFilePath))
        {
            asLogError(_("The path to the real-time predictors catalog is not correct. Please give the path to the file under Options > Preferences > Paths"));
        }
    }

    if (CatalogPredictandsFilePath.IsEmpty())
    {
        doConfigure = true;
    }
    else
    {
        if (!wxFileName::FileExists(CatalogPredictandsFilePath))
        {
            asLogError(_("The path to the predictand catalog is not correct. Please give the path to the file under Options > Preferences > Paths"));
        }
    }

    if (doConfigure)
    {
        asLogError(_("The software is not configured. Please go to Options > Preferences and set the minimum required information."));
    }

    // Get the GUI mode -> silent or not
    long guiOptions = pConfig->Read("/Standard/GuiOptions", 0l);
    if (guiOptions==0l)
    {
        g_SilentMode = true;
    }
    else
    {
        g_SilentMode = false;
        g_VerboseMode = false;
        if (guiOptions==2l)
        {
            g_VerboseMode = true;
        }
    }

}

void AtmoswingFrameCalibrator::SetDefaultOptions()
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString pathSep = wxFileName::GetPathSeparator();

    // General
    long guiOptions = pConfig->Read("/Standard/GuiOptions", 1l);
    pConfig->Write("/Standard/GuiOptions", guiOptions);
    bool responsive;
    pConfig->Read("/Standard/Responsive", &responsive, false);
    pConfig->Write("/Standard/Responsive", responsive);
    long defaultLogLevel = 1; // = selection +1
    long logLevel = pConfig->Read("/Standard/LogLevel", defaultLogLevel);
    pConfig->Write("/Standard/LogLevel", logLevel);
    bool displayLogWindow;
    pConfig->Read("/Standard/DisplayLogWindow", &displayLogWindow, false);
    pConfig->Write("/Standard/DisplayLogWindow", displayLogWindow);
    // Multithreading
    bool allowMultithreading;
    pConfig->Read("/Standard/AllowMultithreading", &allowMultithreading, false);
    pConfig->Write("/Standard/AllowMultithreading", allowMultithreading);
    // Set the number of threads
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads==-1) maxThreads = 2;
    wxString maxThreadsStr = wxString::Format("%d", maxThreads);
    wxString ProcessingMaxThreadNb = pConfig->Read("/Standard/ProcessingMaxThreadNb", maxThreadsStr);
    pConfig->Write("/Standard/ProcessingMaxThreadNb", ProcessingMaxThreadNb);

    // Paths
    wxString dirConfig = asConfig::GetDataDir()+DS+"config";
    wxString dirData = asConfig::GetDataDir()+DS+"data";
    wxString CatalogPredictorsArchiveFilePath = pConfig->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", dirConfig+"CatalogPredictorsArchive.xml");
    pConfig->Write("/StandardPaths/CatalogPredictorsArchiveFilePath", CatalogPredictorsArchiveFilePath);
    wxString CatalogPredictorsRealtimeFilePath = pConfig->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", dirConfig+"CatalogPredictorsRealtime.xml");
    pConfig->Write("/StandardPaths/CatalogPredictorsRealtimeFilePath", CatalogPredictorsRealtimeFilePath);
    wxString CatalogPredictandsFilePath = pConfig->Read("/StandardPaths/CatalogPredictandsFilePath", dirConfig+"CatalogPredictands.xml");
    pConfig->Write("/StandardPaths/CatalogPredictandsFilePath", CatalogPredictandsFilePath);
    wxString PredictandDBDir = pConfig->Read("/StandardPaths/DataPredictandDBDir", dirData+"predictands");
    pConfig->Write("/StandardPaths/DataPredictandDBDir", PredictandDBDir);
    wxString IntermediateResultsDir = pConfig->Read("/StandardPaths/IntermediateResultsDir", asConfig::GetTempDir()+"Atmoswing");
    pConfig->Write("/StandardPaths/IntermediateResultsDir", IntermediateResultsDir);
    wxString CalibrationResultsDir = pConfig->Read("/StandardPaths/CalibrationResultsDir", asConfig::GetDocumentsDir()+"Atmoswing"+DS+"Calibration");
    pConfig->Write("/StandardPaths/CalibrationResultsDir", CalibrationResultsDir);
    wxString ForecastResultsDir = pConfig->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDocumentsDir()+"Atmoswing"+DS+"Forecasts");
    pConfig->Write("/StandardPaths/ForecastResultsDir", ForecastResultsDir);
    wxString RealtimePredictorSavingDir = pConfig->Read("/StandardPaths/RealtimePredictorSavingDir", asConfig::GetDocumentsDir()+"Atmoswing"+DS+"Predictors");
    pConfig->Write("/StandardPaths/RealtimePredictorSavingDir", RealtimePredictorSavingDir);
    wxString ForecasterPath = pConfig->Read("/StandardPaths/ForecasterPath", asConfig::GetDataDir()+"AtmoswingForecaster.exe");
    pConfig->Write("/StandardPaths/ForecasterPath", ForecasterPath);
    wxString ViewerPath = pConfig->Read("/StandardPaths/ViewerPath", asConfig::GetDataDir()+"AtmoswingViewer.exe");
    pConfig->Write("/StandardPaths/ViewerPath", ViewerPath);
    wxString ArchivePredictorsDir = pConfig->Read("/StandardPaths/ArchivePredictorsDir", dirData+"predictors");
    pConfig->Write("/StandardPaths/ArchivePredictorsDir", ArchivePredictorsDir);
    wxString ForecastParametersDir = pConfig->Read("/StandardPaths/ForecastParametersDir", dirConfig);
    pConfig->Write("/StandardPaths/ForecastParametersDir", ForecastParametersDir);

    // Processing
    long defaultMethod = (long)asINSERT;
    long ProcessingMethod = pConfig->Read("/ProcessingOptions/ProcessingMethod", defaultMethod);
    if (!allowMultithreading)
    {
        ProcessingMethod = (long)asINSERT;
    }
    pConfig->Write("/ProcessingOptions/ProcessingMethod", ProcessingMethod);
    long defaultLinAlgebra = (long)asCOEFF;
    long ProcessingLinAlgebra = pConfig->Read("/ProcessingOptions/ProcessingLinAlgebra", defaultLinAlgebra);
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", ProcessingLinAlgebra);

    pConfig->Flush();
}

AtmoswingFrameCalibrator::~AtmoswingFrameCalibrator()
{
    // Config file
    wxConfigBase *pConfig = wxFileConfig::Get();
    if ( pConfig == NULL )
        return;

    // Save the frame position
    int x, y, w, h;
    GetClientSize(&w, &h);
    GetPosition(&x, &y);
    pConfig->Write("/MainFrameCalibrator/x", (long) x);
    pConfig->Write("/MainFrameCalibrator/y", (long) y);
    pConfig->Write("/MainFrameCalibrator/w", (long) w);
    pConfig->Write("/MainFrameCalibrator/h", (long) h);

    //wxDELETE(m_LogWindow);
}

void AtmoswingFrameCalibrator::OnClose(wxCloseEvent &event)
{
    Close(true);
}

void AtmoswingFrameCalibrator::OnQuit(wxCommandEvent &event)
{
    Close(true);
}
