/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asFramePreferences.h"

#include "wx/fileconf.h"
#include "wx/thread.h"

asFramePreferences::asFramePreferences( wxWindow* parent, wxWindowID id )
:
asFramePreferencesVirtual( parent, id )
{
    LoadPreferences();
    Fit();

        // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferences::CloseFrame( wxCommandEvent& event )
{
    Close();
}

void asFramePreferences::Update()
{
    LoadPreferences();
}

void asFramePreferences::LoadPreferences()
{
    wxConfigBase *pConfigForecaster;
    wxConfigBase *pConfigViewer;

    if (g_AppViewer)
    {
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir("Atmoswing forecaster"));
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        pConfigForecaster = new wxFileConfig("Atmoswing",wxEmptyString,asConfig::GetUserDataDir("Atmoswing forecaster") + "Atmoswing.ini",asConfig::GetUserDataDir() + "Atmoswing.ini",wxCONFIG_USE_LOCAL_FILE);
        pConfigViewer = wxFileConfig::Get();
    }
    else if (g_AppForecaster)
    {
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir("Atmoswing viewer"));
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        pConfigForecaster = wxFileConfig::Get();
        pConfigViewer = new wxFileConfig("Atmoswing",wxEmptyString,asConfig::GetUserDataDir("Atmoswing viewer") + "Atmoswing.ini",asConfig::GetUserDataDir() + "Atmoswing.ini",wxCONFIG_USE_LOCAL_FILE);
    }
    else
    {
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir("Atmoswing viewer"));
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        pConfigForecaster = wxFileConfig::Get();
        pConfigViewer = new wxFileConfig("Atmoswing",wxEmptyString,asConfig::GetUserDataDir("Atmoswing viewer") + "Atmoswing.ini",asConfig::GetUserDataDir() + "Atmoswing.ini",wxCONFIG_USE_LOCAL_FILE);
    }

    // Fix the color of the file/dir pickers
    wxColour col = m_NotebookBase->GetThemeBackgroundColour();
    if (col.IsOk())
    {
        m_FilePickerCatalogPredictorsArchive->SetBackgroundColour(col);
        m_FilePickerCatalogPredictorsRealtime->SetBackgroundColour(col);
        m_FilePickerCatalogPredictands->SetBackgroundColour(col);
        m_DirPickerPredictandDB->SetBackgroundColour(col);
        m_DirPickerIntermediateResults->SetBackgroundColour(col);
        m_DirPickerForecastResults->SetBackgroundColour(col);
        m_DirPickerParameters->SetBackgroundColour(col);
        m_DirPickerArchivePredictors->SetBackgroundColour(col);
        m_DirPickerRealtimePredictorSaving->SetBackgroundColour(col);
        m_FilePickerGISLayerHillshade->SetBackgroundColour(col);
        m_FilePickerGISLayerCatchments->SetBackgroundColour(col);
        m_FilePickerGISLayerHydro->SetBackgroundColour(col);
        m_FilePickerGISLayerLakes->SetBackgroundColour(col);
        m_FilePickerGISLayerBasemap->SetBackgroundColour(col);
        m_FilePickerForecaster->SetBackgroundColour(col);
        m_FilePickerViewer->SetBackgroundColour(col);
    }

    // General
    long guiOptions = pConfigForecaster->Read("/Standard/GuiOptions", 1l);
    m_RadioBoxGui->SetSelection((int)guiOptions);
    if (guiOptions==0)
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
    bool responsive;
    pConfigForecaster->Read("/Standard/Responsive", &responsive, true);
    m_CheckBoxResponsiveness->SetValue(responsive);
    if (responsive)
    {
        g_Responsive = true;
    }
    else
    {
        g_Responsive = false;
    }

    bool multiForecaster;
    pConfigForecaster->Read("/Standard/MultiInstances", &multiForecaster, false);
    m_CheckBoxMultiInstancesForecaster->SetValue(multiForecaster);
    bool multiViewer;
    pConfigViewer->Read("/Standard/MultiInstances", &multiViewer, false);
    m_CheckBoxMultiInstancesViewer->SetValue(multiViewer);

    long defaultLogLevelForecaster = 1; // = selection +1
    long logLevelForecaster = pConfigForecaster->Read("/Standard/LogLevel", defaultLogLevelForecaster);
    m_RadioBoxLogFLevel->SetSelection((int)logLevelForecaster-1);
    bool displayLogWindowForecaster;
    pConfigForecaster->Read("/Standard/DisplayLogWindow", &displayLogWindowForecaster, false);
    m_CheckBoxDisplayLogFWindow->SetValue(displayLogWindowForecaster);

    long defaultLogLevelViewer = 1; // = selection +1
    long logLevelViewer = pConfigViewer->Read("/Standard/LogLevel", defaultLogLevelViewer);
    m_RadioBoxLogVLevel->SetSelection((int)logLevelViewer-1);
    bool displayLogWindowViewer;
    pConfigViewer->Read("/Standard/DisplayLogWindow", &displayLogWindowViewer, false);
    m_CheckBoxDisplayLogVWindow->SetValue(displayLogWindowViewer);
    // Multithreading
    bool allowMultithreading;
    pConfigForecaster->Read("/Standard/AllowMultithreading", &allowMultithreading, false);
    m_CheckBoxAllowMultithreading->SetValue(allowMultithreading);
    // Set the number of threads
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads==-1) maxThreads = 2;
    wxString maxThreadsStr = wxString::Format("%d", maxThreads);
    wxString ProcessingMaxThreadNb = pConfigForecaster->Read("/Standard/ProcessingMaxThreadNb", maxThreadsStr);
    m_TextCtrlThreadsNb->SetValue(ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = pConfigForecaster->Read("/Standard/ProcessingThreadsPriority", 95l);
    m_SliderThreadsPriority->SetValue((int)ProcessingThreadsPriority);

    // Internet
    int maxPrevStepsNb = 5;
    wxString maxPrevStepsNbStr = wxString::Format("%d", maxPrevStepsNb);
    wxString InternetMaxPrevStepsNb = pConfigForecaster->Read("/Internet/MaxPreviousStepsNb", maxPrevStepsNbStr);
    m_TextCtrlMaxPrevStepsNb->SetValue(InternetMaxPrevStepsNb);
    int maxParallelRequests = 1;
    wxString maxParallelRequestsStr = wxString::Format("%d", maxParallelRequests);
    wxString InternetParallelRequestsNb = pConfigForecaster->Read("/Internet/ParallelRequestsNb", maxParallelRequestsStr);
    m_TextCtrlMaxRequestsNb->SetValue(InternetParallelRequestsNb);
    bool restrictDownloads;
    pConfigForecaster->Read("/Internet/RestrictDownloads", &restrictDownloads, true);
    m_CheckBoxRestrictDownloads->SetValue(restrictDownloads);
    bool checkBoxProxy;
    pConfigForecaster->Read("/Internet/UsesProxy", &checkBoxProxy, false);
    m_CheckBoxProxy->SetValue(checkBoxProxy);
    wxString ProxyAddress = pConfigForecaster->Read("/Internet/ProxyAddress", wxEmptyString);
    m_TextCtrlProxyAddress->SetValue(ProxyAddress);
    wxString ProxyPort = pConfigForecaster->Read("/Internet/ProxyPort", wxEmptyString);
    m_TextCtrlProxyPort->SetValue(ProxyPort);
    wxString ProxyUser = pConfigForecaster->Read("/Internet/ProxyUser", wxEmptyString);
    m_TextCtrlProxyUser->SetValue(ProxyUser);
    wxString ProxyPasswd = pConfigForecaster->Read("/Internet/ProxyPasswd", wxEmptyString);
    m_TextCtrlProxyPasswd->SetValue(ProxyPasswd);

    // Paths
    wxString dirConfig = asConfig::GetDataDir()+"config"+DS;
    wxString dirData = asConfig::GetDataDir()+"data"+DS;
    wxString CatalogPredictorsArchiveFilePath = pConfigForecaster->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", dirConfig+"CatalogPredictorsArchive.xml");
    m_FilePickerCatalogPredictorsArchive->SetPath(CatalogPredictorsArchiveFilePath);
    wxString CatalogPredictorsRealtimeFilePath = pConfigForecaster->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", dirConfig+"CatalogPredictorsRealtime.xml");
    m_FilePickerCatalogPredictorsRealtime->SetPath(CatalogPredictorsRealtimeFilePath);
    wxString CatalogPredictandsFilePath = pConfigForecaster->Read("/StandardPaths/CatalogPredictandsFilePath", dirConfig+"CatalogPredictands.xml");
    m_FilePickerCatalogPredictands->SetPath(CatalogPredictandsFilePath);
    wxString PredictandDBDir = pConfigForecaster->Read("/StandardPaths/DataPredictandDBDir", dirData+"predictands");
    m_DirPickerPredictandDB->SetPath(PredictandDBDir);
    wxString IntermediateResultsDir = pConfigForecaster->Read("/StandardPaths/IntermediateResultsDir", asConfig::GetTempDir()+"Atmoswing");
    m_DirPickerIntermediateResults->SetPath(IntermediateResultsDir);
    wxString ForecastResultsDir = pConfigForecaster->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDocumentsDir()+"Atmoswing"+DS+"Forecasts");
    m_DirPickerForecastResults->SetPath(ForecastResultsDir);
    wxString RealtimePredictorSavingDir = pConfigForecaster->Read("/StandardPaths/RealtimePredictorSavingDir", asConfig::GetDocumentsDir()+"Atmoswing"+DS+"Predictors");
    m_DirPickerRealtimePredictorSaving->SetPath(RealtimePredictorSavingDir);
    wxString ForecasterPath = pConfigForecaster->Read("/StandardPaths/ForecasterPath", asConfig::GetDataDir()+"AtmoswingForecaster.exe");
    m_FilePickerForecaster->SetPath(ForecasterPath);
    wxString ViewerPath = pConfigForecaster->Read("/StandardPaths/ViewerPath", asConfig::GetDataDir()+"AtmoswingViewer.exe");
    m_FilePickerViewer->SetPath(ViewerPath);
    wxString ArchivePredictorsDir = pConfigForecaster->Read("/StandardPaths/ArchivePredictorsDir", dirData+"predictors");
    m_DirPickerArchivePredictors->SetPath(ArchivePredictorsDir);
	wxString ForecastParametersDir = pConfigForecaster->Read("/StandardPaths/ForecastParametersDir", dirConfig);
    m_DirPickerParameters->SetPath(ForecastParametersDir);

    // Processing
    long defaultMethod = (long)asINSERT;
    long ProcessingMethod = pConfigForecaster->Read("/ProcessingOptions/ProcessingMethod", defaultMethod);
    if (!allowMultithreading)
    {
        m_RadioBoxProcessingMethods->Enable(0, false);
        if (ProcessingMethod==(long)asMULTITHREADS)
        {
            ProcessingMethod = (long)asINSERT;
        }
    }
    else
    {
        m_RadioBoxProcessingMethods->Enable(0, true);
    }
    m_RadioBoxProcessingMethods->SetSelection((int)ProcessingMethod);
    long defaultLinAlgebra = (long)asCOEFF;
    #if defined (__WIN32__)
        defaultLinAlgebra = (long)asCOEFF;
    #endif
    long ProcessingLinAlgebra = pConfigForecaster->Read("/ProcessingOptions/ProcessingLinAlgebra", defaultLinAlgebra);
    m_RadioBoxLinearAlgebra->SetSelection((int)ProcessingLinAlgebra);

    // GIS
        // Hillshade
    wxString LayerHillshadeFilePath = pConfigViewer->Read("/GIS/LayerHillshadeFilePath", dirData+"gis"+DS+"Local"+DS+"hillshade"+DS+"hdr.adf");
    m_FilePickerGISLayerHillshade->SetPath(LayerHillshadeFilePath);
    wxString LayerHillshadeTransp = pConfigViewer->Read("/GIS/LayerHillshadeTransp", "0");
    m_TextCtrlGISLayerHillshadeTransp->SetValue(LayerHillshadeTransp);
    bool LayerHillshadeVisibility;
    pConfigViewer->Read("/GIS/LayerHillshadeVisibility", &LayerHillshadeVisibility, true);
    m_CheckBoxGISLayerHillshadeVisibility->SetValue(LayerHillshadeVisibility);
        // Catchments
    wxString LayerCatchmentsFilePath = pConfigViewer->Read("/GIS/LayerCatchmentsFilePath", dirData+"gis"+DS+"Local"+DS+"catchments.shp");
    m_FilePickerGISLayerCatchments->SetPath(LayerCatchmentsFilePath);
    wxString LayerCatchmentsTransp = pConfigViewer->Read("/GIS/LayerCatchmentsTransp", "50");
    m_TextCtrlGISLayerCatchmentsTransp->SetValue(LayerCatchmentsTransp);
    long LayerCatchmentsColor = (long)0x0000fffc;
    LayerCatchmentsColor = pConfigViewer->Read("/GIS/LayerCatchmentsColor", LayerCatchmentsColor);
    wxColour colorCatchments;
    colorCatchments.SetRGB((wxUint32)LayerCatchmentsColor);
    m_ColourPickerGISLayerCatchmentsColor->SetColour(colorCatchments);
    wxString LayerCatchmentsSize = pConfigViewer->Read("/GIS/LayerCatchmentsSize", "1");
    m_TextCtrlGISLayerCatchmentsSize->SetValue(LayerCatchmentsSize);
    bool LayerCatchmentsVisibility;
    pConfigViewer->Read("/GIS/LayerCatchmentsVisibility", &LayerCatchmentsVisibility, false);
    m_CheckBoxGISLayerCatchmentsVisibility->SetValue(LayerCatchmentsVisibility);
        // Hydro
    wxString LayerHydroFilePath = pConfigViewer->Read("/GIS/LayerHydroFilePath", dirData+"gis"+DS+"Local"+DS+"hydrography.shp");
    m_FilePickerGISLayerHydro->SetPath(LayerHydroFilePath);
    wxString LayerHydroTransp = pConfigViewer->Read("/GIS/LayerHydroTransp", "40");
    m_TextCtrlGISLayerHydroTransp->SetValue(LayerHydroTransp);
    long LayerHydroColor = (long)0x00c81616;
    LayerHydroColor = pConfigViewer->Read("/GIS/LayerHydroColor", LayerHydroColor);
    wxColour colorHydro;
    colorHydro.SetRGB((wxUint32)LayerHydroColor);
    m_ColourPickerGISLayerHydroColor->SetColour(colorHydro);
    wxString LayerHydroSize = pConfigViewer->Read("/GIS/LayerHydroSize", "1");
    m_TextCtrlGISLayerHydroSize->SetValue(LayerHydroSize);
    bool LayerHydroVisibility;
    pConfigViewer->Read("/GIS/LayerHydroVisibility", &LayerHydroVisibility, true);
    m_CheckBoxGISLayerHydroVisibility->SetValue(LayerHydroVisibility);
        // Lakes
    wxString LayerLakesFilePath = pConfigViewer->Read("/GIS/LayerLakesFilePath", dirData+"gis"+DS+"Local"+DS+"lakes.shp");
    m_FilePickerGISLayerLakes->SetPath(LayerLakesFilePath);
    wxString LayerLakesTransp = pConfigViewer->Read("/GIS/LayerLakesTransp", "40");
    m_TextCtrlGISLayerLakesTransp->SetValue(LayerLakesTransp);
    long LayerLakesColor = (long)0x00d4c92a;
    LayerLakesColor = pConfigViewer->Read("/GIS/LayerLakesColor", LayerLakesColor);
    wxColour colorLakes;
    colorLakes.SetRGB((wxUint32)LayerLakesColor);
    m_ColourPickerGISLayerLakesColor->SetColour(colorLakes);
    bool LayerLakesVisibility;
    pConfigViewer->Read("/GIS/LayerLakesVisibility", &LayerLakesVisibility, true);
    m_CheckBoxGISLayerLakesVisibility->SetValue(LayerLakesVisibility);
        // Basemap
    wxString LayerBasemapFilePath = pConfigViewer->Read("/GIS/LayerBasemapFilePath", dirData+"gis"+DS+"Local"+DS+"basemap.shp");
    m_FilePickerGISLayerBasemap->SetPath(LayerBasemapFilePath);
    wxString LayerBasemapTransp = pConfigViewer->Read("/GIS/LayerBasemapTransp", "50");
    m_TextCtrlGISLayerBasemapTransp->SetValue(LayerBasemapTransp);
    bool LayerBasemapVisibility;
    pConfigViewer->Read("/GIS/LayerBasemapVisibility", &LayerBasemapVisibility, false);
    m_CheckBoxGISLayerBasemapVisibility->SetValue(LayerBasemapVisibility);
        // Continents
    wxString LayerContinentsFilePath = pConfigViewer->Read("/GIS/LayerContinentsFilePath", dirData+"gis"+DS+"World"+DS+"continents.shp");
    m_FilePickerGISLayerContinents->SetPath(LayerContinentsFilePath);
    wxString LayerContinentsTransp = pConfigViewer->Read("/GIS/LayerContinentsTransp", "50");
    m_TextCtrlGISLayerContinentsTransp->SetValue(LayerContinentsTransp);
    long LayerContinentsColor = (long)0x99999999;
    LayerContinentsColor = pConfigViewer->Read("/GIS/LayerContinentsColor", LayerContinentsColor);
    wxColour colorContinents;
    colorContinents.SetRGB((wxUint32)LayerContinentsColor);
    m_ColourPickerGISLayerContinentsColor->SetColour(colorContinents);
    wxString LayerContinentsSize = pConfigViewer->Read("/GIS/LayerContinentsSize", "1");
    m_TextCtrlGISLayerContinentsSize->SetValue(LayerContinentsSize);
    bool LayerContinentsVisibility;
    pConfigViewer->Read("/GIS/LayerContinentsVisibility", &LayerContinentsVisibility, true);
    m_CheckBoxGISLayerContinentsVisibility->SetValue(LayerContinentsVisibility);
        // Countries
    wxString LayerCountriesFilePath = pConfigViewer->Read("/GIS/LayerCountriesFilePath", dirData+"gis"+DS+"World"+DS+"countries.shp");
    m_FilePickerGISLayerCountries->SetPath(LayerCountriesFilePath);
    wxString LayerCountriesTransp = pConfigViewer->Read("/GIS/LayerCountriesTransp", "0");
    m_TextCtrlGISLayerCountriesTransp->SetValue(LayerCountriesTransp);
    long LayerCountriesColor = (long)0x77999999;
    LayerCountriesColor = pConfigViewer->Read("/GIS/LayerCountriesColor", LayerCountriesColor);
    wxColour colorCountries;
    colorCountries.SetRGB((wxUint32)LayerCountriesColor);
    m_ColourPickerGISLayerCountriesColor->SetColour(colorCountries);
    wxString LayerCountriesSize = pConfigViewer->Read("/GIS/LayerCountriesSize", "1");
    m_TextCtrlGISLayerCountriesSize->SetValue(LayerCountriesSize);
    bool LayerCountriesVisibility;
    pConfigViewer->Read("/GIS/LayerCountriesVisibility", &LayerCountriesVisibility, true);
    m_CheckBoxGISLayerCountriesVisibility->SetValue(LayerCountriesVisibility);
        // Geogrid
    wxString LayerGeogridFilePath = pConfigViewer->Read("/GIS/LayerGeogridFilePath", dirData+"gis"+DS+"World"+DS+"geogrid.shp");
    m_FilePickerGISLayerGeogrid->SetPath(LayerGeogridFilePath);
    wxString LayerGeogridTransp = pConfigViewer->Read("/GIS/LayerGeogridTransp", "50");
    m_TextCtrlGISLayerGeogridTransp->SetValue(LayerGeogridTransp);
    long LayerGeogridColor = (long)0xff999999;
    LayerGeogridColor = pConfigViewer->Read("/GIS/LayerGeogridColor", LayerGeogridColor);
    wxColour colorGeogrid;
    colorGeogrid.SetRGB((wxUint32)LayerGeogridColor);
    m_ColourPickerGISLayerGeogridColor->SetColour(colorGeogrid);
    wxString LayerGeogridSize = pConfigViewer->Read("/GIS/LayerGeogridSize", "2");
    m_TextCtrlGISLayerGeogridSize->SetValue(LayerGeogridSize);
    bool LayerGeogridVisibility;
    pConfigViewer->Read("/GIS/LayerGeogridVisibility", &LayerGeogridVisibility, false);
    m_CheckBoxGISLayerGeogridVisibility->SetValue(LayerGeogridVisibility);
        // LatLong
    wxString LayerLatLongFilePath = pConfigViewer->Read("/GIS/LayerLatLongFilePath", dirData+"gis"+DS+"World"+DS+"latlong.shp");
    m_FilePickerGISLayerLatLong->SetPath(LayerLatLongFilePath);
    wxString LayerLatLongTransp = pConfigViewer->Read("/GIS/LayerLatLongTransp", "80");
    m_TextCtrlGISLayerLatLongTransp->SetValue(LayerLatLongTransp);
    long LayerLatLongColor = (long)0xff999999;
    LayerLatLongColor = pConfigViewer->Read("/GIS/LayerLatLongColor", LayerLatLongColor);
    wxColour colorLatLong;
    colorLatLong.SetRGB((wxUint32)LayerLatLongColor);
    m_ColourPickerGISLayerLatLongColor->SetColour(colorLatLong);
    wxString LayerLatLongSize = pConfigViewer->Read("/GIS/LayerLatLongSize", "1");
    m_TextCtrlGISLayerLatLongSize->SetValue(LayerLatLongSize);
    bool LayerLatLongVisibility;
    pConfigViewer->Read("/GIS/LayerLatLongVisibility", &LayerLatLongVisibility, true);
    m_CheckBoxGISLayerLatLongVisibility->SetValue(LayerLatLongVisibility);

    // Forecast display
    wxString ColorbarMaxValue = pConfigViewer->Read("/GIS/ColorbarMaxValue", "50");
    m_TextCtrlColorbarMaxValue->SetValue(ColorbarMaxValue);
    wxString PastDaysNb = pConfigViewer->Read("/Plot/PastDaysNb", "3");
    m_TextCtrlPastDaysNb->SetValue(PastDaysNb);
    int alarmsReturnPeriod;
    pConfigViewer->Read("/SidebarAlarms/ReturnPeriod", &alarmsReturnPeriod, 10);
    switch (alarmsReturnPeriod)
    {
        case 2:
            m_ChoiceAlarmsReturnPeriod->SetSelection(0);
            break;
        case 5:
            m_ChoiceAlarmsReturnPeriod->SetSelection(1);
            break;
        case 10:
            m_ChoiceAlarmsReturnPeriod->SetSelection(2);
            break;
        case 20:
            m_ChoiceAlarmsReturnPeriod->SetSelection(3);
            break;
        case 50:
            m_ChoiceAlarmsReturnPeriod->SetSelection(4);
            break;
        case 100:
            m_ChoiceAlarmsReturnPeriod->SetSelection(5);
            break;
        default:
            m_ChoiceAlarmsReturnPeriod->SetSelection(2);
            pConfigViewer->Write("/SidebarAlarms/ReturnPeriod", 10);
    }
    wxString alarmsPercentile = pConfigViewer->Read("/SidebarAlarms/Percentile", "0.9");
    m_TextCtrlAlarmsPercentile->SetValue(alarmsPercentile);

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_StaticTextUserDir->SetLabel(userpath);
    wxString logpathForecaster = asConfig::GetLogDir();
    logpathForecaster.Append(DS);
    logpathForecaster.Append("AtmoswingForecaster.log");
    m_StaticTextLogFileForecaster->SetLabel(logpathForecaster);
    wxString logpathViewer = asConfig::GetLogDir();
    logpathViewer.Append(DS);
    logpathViewer.Append("AtmoswingViewer.log");
    m_StaticTextLogFileViewer->SetLabel(logpathViewer);
    m_StaticTextPrefFileForecaster->SetLabel(asConfig::GetUserDataDir("Atmoswing forecaster")+"Atmoswing.ini");
    m_StaticTextPrefFileViewer->SetLabel(asConfig::GetUserDataDir("Atmoswing viewer")+"Atmoswing.ini");

    if (g_AppViewer)
    {
        wxDELETE(pConfigForecaster);
    }
    else if (g_AppForecaster)
    {
        wxDELETE(pConfigViewer);
    }
    else
    {
        wxDELETE(pConfigViewer);
    }
}

void asFramePreferences::SavePreferences( )
{
    wxConfigBase *pConfigForecaster;
    wxConfigBase *pConfigViewer;

    if (g_AppViewer)
    {
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir("Atmoswing forecaster"));
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        pConfigForecaster = new wxFileConfig("Atmoswing",wxEmptyString,asConfig::GetUserDataDir("Atmoswing forecaster")+"Atmoswing.ini",asConfig::GetUserDataDir()+DS+"Atmoswing.ini",wxCONFIG_USE_LOCAL_FILE);
        pConfigViewer = wxFileConfig::Get();
    }
    else if (g_AppForecaster)
    {
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir("Atmoswing viewer"));
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        pConfigForecaster = wxFileConfig::Get();
        pConfigViewer = new wxFileConfig("Atmoswing",wxEmptyString,asConfig::GetUserDataDir("Atmoswing viewer")+"Atmoswing.ini",asConfig::GetUserDataDir()+DS+"Atmoswing.ini",wxCONFIG_USE_LOCAL_FILE);
    }
    else
    {
        wxFileName userDir = wxFileName::DirName(asConfig::GetUserDataDir("Atmoswing viewer"));
        userDir.Mkdir(wxS_DIR_DEFAULT,wxPATH_MKDIR_FULL);

        pConfigForecaster = wxFileConfig::Get();
        pConfigViewer = new wxFileConfig("Atmoswing",wxEmptyString,asConfig::GetUserDataDir("Atmoswing viewer")+"Atmoswing.ini",asConfig::GetUserDataDir()+DS+"Atmoswing.ini",wxCONFIG_USE_LOCAL_FILE);
    }

    // General
    long guiOptions = (long)m_RadioBoxGui->GetSelection();
    pConfigForecaster->Write("/Standard/GuiOptions", guiOptions);
    pConfigViewer->Write("/Standard/GuiOptions", guiOptions);
    if (guiOptions==0)
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
    bool responsive = m_CheckBoxResponsiveness->GetValue();
    pConfigForecaster->Write("/Standard/Responsive", responsive);
    pConfigViewer->Write("/Standard/Responsive", responsive);
    if (responsive)
    {
        g_Responsive = true;
    }
    else
    {
        g_Responsive = false;
    }

    bool multiForecaster = m_CheckBoxMultiInstancesForecaster->GetValue();
    pConfigForecaster->Write("/Standard/MultiInstances", multiForecaster);
    bool multiViewer = m_CheckBoxMultiInstancesViewer->GetValue();
    pConfigViewer->Write("/Standard/MultiInstances", multiViewer);

    long logLevelForecaster = (long)m_RadioBoxLogFLevel->GetSelection();
    pConfigForecaster->Write("/Standard/LogLevel", logLevelForecaster+1); // = selection +1
    bool displayLogWindowForecaster = m_CheckBoxDisplayLogFWindow->GetValue();
    pConfigForecaster->Write("/Standard/DisplayLogWindow", displayLogWindowForecaster);

    long logLevelViewer = (long)m_RadioBoxLogVLevel->GetSelection();
    pConfigViewer->Write("/Standard/LogLevel", logLevelViewer+1); // = selection +1
    bool displayLogWindowViewer = m_CheckBoxDisplayLogVWindow->GetValue();
    pConfigViewer->Write("/Standard/DisplayLogWindow", displayLogWindowViewer);

    // Multithreading
    bool allowMultithreading = m_CheckBoxAllowMultithreading->GetValue();
    pConfigForecaster->Write("/Standard/AllowMultithreading", allowMultithreading);
    pConfigViewer->Write("/Standard/AllowMultithreading", allowMultithreading);
    wxString ProcessingMaxThreadNb = m_TextCtrlThreadsNb->GetValue();
    if (!ProcessingMaxThreadNb.IsNumber()) ProcessingMaxThreadNb = "2";
    pConfigForecaster->Write("/Standard/ProcessingMaxThreadNb", ProcessingMaxThreadNb);
    pConfigViewer->Write("/Standard/ProcessingMaxThreadNb", ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = (long)m_SliderThreadsPriority->GetValue();
    pConfigForecaster->Write("/Standard/ProcessingThreadsPriority", ProcessingThreadsPriority);
    pConfigViewer->Write("/Standard/ProcessingThreadsPriority", ProcessingThreadsPriority);
    // Internet
    wxString InternetMaxPrevStepsNb = m_TextCtrlMaxPrevStepsNb->GetValue();
    if (!InternetMaxPrevStepsNb.IsNumber()) InternetMaxPrevStepsNb = "5";
    pConfigForecaster->Write("/Internet/MaxPreviousStepsNb", InternetMaxPrevStepsNb);
    wxString InternetParallelRequestsNb = m_TextCtrlMaxRequestsNb->GetValue();
    if (!InternetParallelRequestsNb.IsNumber()) InternetParallelRequestsNb = "3";
    pConfigForecaster->Write("/Internet/ParallelRequestsNb", InternetParallelRequestsNb);
    bool restrictDownloads = m_CheckBoxRestrictDownloads->GetValue();
    pConfigForecaster->Write("/Internet/RestrictDownloads", restrictDownloads);
    bool checkBoxProxy = m_CheckBoxProxy->GetValue();
    pConfigForecaster->Write("/Internet/UsesProxy", checkBoxProxy);
    pConfigViewer->Write("/Internet/UsesProxy", checkBoxProxy);
    wxString ProxyAddress = m_TextCtrlProxyAddress->GetValue();
    pConfigForecaster->Write("/Internet/ProxyAddress", ProxyAddress);
    pConfigViewer->Write("/Internet/ProxyAddress", ProxyAddress);
    wxString ProxyPort = m_TextCtrlProxyPort->GetValue();
    pConfigForecaster->Write("/Internet/ProxyPort", ProxyPort);
    pConfigViewer->Write("/Internet/ProxyPort", ProxyPort);
    wxString ProxyUser = m_TextCtrlProxyUser->GetValue();
    pConfigForecaster->Write("/Internet/ProxyUser", ProxyUser);
    pConfigViewer->Write("/Internet/ProxyUser", ProxyUser);
    wxString ProxyPasswd = m_TextCtrlProxyPasswd->GetValue();
    pConfigForecaster->Write("/Internet/ProxyPasswd", ProxyPasswd);
    pConfigViewer->Write("/Internet/ProxyPasswd", ProxyPasswd);

    // Paths
    wxString CatalogPredictorsArchiveFilePath = m_FilePickerCatalogPredictorsArchive->GetPath();
    pConfigForecaster->Write("/StandardPaths/CatalogPredictorsArchiveFilePath", CatalogPredictorsArchiveFilePath);
    pConfigViewer->Write("/StandardPaths/CatalogPredictorsArchiveFilePath", CatalogPredictorsArchiveFilePath);
    wxString CatalogPredictorsRealtimeFilePath = m_FilePickerCatalogPredictorsRealtime->GetPath();
    pConfigForecaster->Write("/StandardPaths/CatalogPredictorsRealtimeFilePath", CatalogPredictorsRealtimeFilePath);
    pConfigViewer->Write("/StandardPaths/CatalogPredictorsRealtimeFilePath", CatalogPredictorsRealtimeFilePath);
    wxString CatalogPredictandsFilePath = m_FilePickerCatalogPredictands->GetPath();
    pConfigForecaster->Write("/StandardPaths/CatalogPredictandsFilePath", CatalogPredictandsFilePath);
    pConfigViewer->Write("/StandardPaths/CatalogPredictandsFilePath", CatalogPredictandsFilePath);
    wxString PredictandDBDir = m_DirPickerPredictandDB->GetPath();
    pConfigForecaster->Write("/StandardPaths/DataPredictandDBDir", PredictandDBDir);
    pConfigViewer->Write("/StandardPaths/DataPredictandDBDir", PredictandDBDir);
    wxString IntermediateResultsDir = m_DirPickerIntermediateResults->GetPath();
    pConfigForecaster->Write("/StandardPaths/IntermediateResultsDir", IntermediateResultsDir);
    wxString ForecastResultsDir = m_DirPickerForecastResults->GetPath();
    pConfigForecaster->Write("/StandardPaths/ForecastResultsDir", ForecastResultsDir);
    pConfigViewer->Write("/StandardPaths/ForecastResultsDir", ForecastResultsDir);
    wxString RealtimePredictorSavingDir = m_DirPickerRealtimePredictorSaving->GetPath();
    pConfigForecaster->Write("/StandardPaths/RealtimePredictorSavingDir", RealtimePredictorSavingDir);
    pConfigViewer->Write("/StandardPaths/RealtimePredictorSavingDir", RealtimePredictorSavingDir);
    wxString ForecasterPath = m_FilePickerForecaster->GetPath();
    pConfigForecaster->Write("/StandardPaths/ForecasterPath", ForecasterPath);
    pConfigViewer->Write("/StandardPaths/ForecasterPath", ForecasterPath);
    wxString ViewerPath = m_FilePickerViewer->GetPath();
    pConfigForecaster->Write("/StandardPaths/ViewerPath", ViewerPath);
    pConfigViewer->Write("/StandardPaths/ViewerPath", ViewerPath);
    wxString ArchivePredictorsDir = m_DirPickerArchivePredictors->GetPath();
    pConfigForecaster->Write("/StandardPaths/ArchivePredictorsDir", ArchivePredictorsDir);
	wxString ForecastParametersDir = m_DirPickerParameters->GetPath();
	pConfigForecaster->Write("/StandardPaths/ForecastParametersDir", ForecastParametersDir);

    // Processing
    long ProcessingMethod = (long)m_RadioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && ProcessingMethod==(long)asMULTITHREADS)
    {
        ProcessingMethod = (long)asINSERT;
    }
    pConfigForecaster->Write("/ProcessingOptions/ProcessingMethod", ProcessingMethod);
    long ProcessingLinAlgebra = (long)m_RadioBoxLinearAlgebra->GetSelection();
    pConfigForecaster->Write("/ProcessingOptions/ProcessingLinAlgebra", ProcessingLinAlgebra);

    // GIS
        // Hillshade
    wxString LayerHillshadeFilePath = m_FilePickerGISLayerHillshade->GetPath();
    pConfigViewer->Write("/GIS/LayerHillshadeFilePath", LayerHillshadeFilePath);
    wxString LayerHillshadeTransp = m_TextCtrlGISLayerHillshadeTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerHillshadeTransp", LayerHillshadeTransp);
    bool LayerHillshadeVisibility = m_CheckBoxGISLayerHillshadeVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerHillshadeVisibility", LayerHillshadeVisibility);
        // Catchments
    wxString LayerCatchmentsFilePath = m_FilePickerGISLayerCatchments->GetPath();
    pConfigViewer->Write("/GIS/LayerCatchmentsFilePath", LayerCatchmentsFilePath);
    wxString LayerCatchmentsTransp = m_TextCtrlGISLayerCatchmentsTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerCatchmentsTransp", LayerCatchmentsTransp);
    wxColour colorCatchments = m_ColourPickerGISLayerCatchmentsColor->GetColour();
    pConfigViewer->Write("/GIS/LayerCatchmentsColor", (long)colorCatchments.GetRGB());
    wxString LayerCatchmentsSize = m_TextCtrlGISLayerCatchmentsSize->GetValue();
    pConfigViewer->Write("/GIS/LayerCatchmentsSize", LayerCatchmentsSize);
    bool LayerCatchmentsVisibility = m_CheckBoxGISLayerCatchmentsVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerCatchmentsVisibility", LayerCatchmentsVisibility);
        // Hydro
    wxString LayerHydroFilePath = m_FilePickerGISLayerHydro->GetPath();
    pConfigViewer->Write("/GIS/LayerHydroFilePath", LayerHydroFilePath);
    wxString LayerHydroTransp = m_TextCtrlGISLayerHydroTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerHydroTransp", LayerHydroTransp);
    wxColour colorHydro = m_ColourPickerGISLayerHydroColor->GetColour();
    pConfigViewer->Write("/GIS/LayerHydroColor", (long)colorHydro.GetRGB());
    wxString LayerHydroSize = m_TextCtrlGISLayerHydroSize->GetValue();
    pConfigViewer->Write("/GIS/LayerHydroSize", LayerHydroSize);
    bool LayerHydroVisibility = m_CheckBoxGISLayerHydroVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerHydroVisibility", LayerHydroVisibility);
        // Lakes
    wxString LayerLakesFilePath = m_FilePickerGISLayerLakes->GetPath();
    pConfigViewer->Write("/GIS/LayerLakesFilePath", LayerLakesFilePath);
    wxString LayerLakesTransp = m_TextCtrlGISLayerLakesTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerLakesTransp", LayerLakesTransp);
    wxColour colorLakes = m_ColourPickerGISLayerLakesColor->GetColour();
    pConfigViewer->Write("/GIS/LayerLakesColor", (long)colorLakes.GetRGB());
    bool LayerLakesVisibility = m_CheckBoxGISLayerLakesVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerLakesVisibility", LayerLakesVisibility);
        // Basemap
    wxString LayerBasemapFilePath = m_FilePickerGISLayerBasemap->GetPath();
    pConfigViewer->Write("/GIS/LayerBasemapFilePath", LayerBasemapFilePath);
    wxString LayerBasemapTransp = m_TextCtrlGISLayerBasemapTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerBasemapTransp", LayerBasemapTransp);
    bool LayerBasemapVisibility = m_CheckBoxGISLayerBasemapVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerBasemapVisibility", LayerBasemapVisibility);
	    // Continents
    wxString LayerContinentsFilePath = m_FilePickerGISLayerContinents->GetPath();
    pConfigViewer->Write("/GIS/LayerContinentsFilePath", LayerContinentsFilePath);
    wxString LayerContinentsTransp = m_TextCtrlGISLayerContinentsTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerContinentsTransp", LayerContinentsTransp);
    wxColour colorContinents = m_ColourPickerGISLayerContinentsColor->GetColour();
    pConfigViewer->Write("/GIS/LayerContinentsColor", (long)colorContinents.GetRGB());
    wxString LayerContinentsSize = m_TextCtrlGISLayerContinentsSize->GetValue();
    pConfigViewer->Write("/GIS/LayerContinentsSize", LayerContinentsSize);
    bool LayerContinentsVisibility = m_CheckBoxGISLayerContinentsVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerContinentsVisibility", LayerContinentsVisibility);
	    // Countries
    wxString LayerCountriesFilePath = m_FilePickerGISLayerCountries->GetPath();
    pConfigViewer->Write("/GIS/LayerCountriesFilePath", LayerCountriesFilePath);
    wxString LayerCountriesTransp = m_TextCtrlGISLayerCountriesTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerCountriesTransp", LayerCountriesTransp);
    wxColour colorCountries = m_ColourPickerGISLayerCountriesColor->GetColour();
    pConfigViewer->Write("/GIS/LayerCountriesColor", (long)colorCountries.GetRGB());
    wxString LayerCountriesSize = m_TextCtrlGISLayerCountriesSize->GetValue();
    pConfigViewer->Write("/GIS/LayerCountriesSize", LayerCountriesSize);
    bool LayerCountriesVisibility = m_CheckBoxGISLayerCountriesVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerCountriesVisibility", LayerCountriesVisibility);
	    // Geogrid
    wxString LayerGeogridFilePath = m_FilePickerGISLayerGeogrid->GetPath();
    pConfigViewer->Write("/GIS/LayerGeogridFilePath", LayerGeogridFilePath);
    wxString LayerGeogridTransp = m_TextCtrlGISLayerGeogridTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerGeogridTransp", LayerGeogridTransp);
    wxColour colorGeogrid = m_ColourPickerGISLayerGeogridColor->GetColour();
    pConfigViewer->Write("/GIS/LayerGeogridColor", (long)colorGeogrid.GetRGB());
    wxString LayerGeogridSize = m_TextCtrlGISLayerGeogridSize->GetValue();
    pConfigViewer->Write("/GIS/LayerGeogridSize", LayerGeogridSize);
    bool LayerGeogridVisibility = m_CheckBoxGISLayerGeogridVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerGeogridVisibility", LayerGeogridVisibility);
	    // LatLong
    wxString LayerLatLongFilePath = m_FilePickerGISLayerLatLong->GetPath();
    pConfigViewer->Write("/GIS/LayerLatLongFilePath", LayerLatLongFilePath);
    wxString LayerLatLongTransp = m_TextCtrlGISLayerLatLongTransp->GetValue();
    pConfigViewer->Write("/GIS/LayerLatLongTransp", LayerLatLongTransp);
    wxColour colorLatLong = m_ColourPickerGISLayerLatLongColor->GetColour();
    pConfigViewer->Write("/GIS/LayerLatLongColor", (long)colorLatLong.GetRGB());
    wxString LayerLatLongSize = m_TextCtrlGISLayerLatLongSize->GetValue();
    pConfigViewer->Write("/GIS/LayerLatLongSize", LayerLatLongSize);
    bool LayerLatLongVisibility = m_CheckBoxGISLayerLatLongVisibility->GetValue();
    pConfigViewer->Write("/GIS/LayerLatLongVisibility", LayerLatLongVisibility);

    // Forecast display
    wxString ColorbarMaxValue = m_TextCtrlColorbarMaxValue->GetValue();
    pConfigViewer->Write("/GIS/ColorbarMaxValue", ColorbarMaxValue);
    wxString PastDaysNb = m_TextCtrlPastDaysNb->GetValue();
    pConfigViewer->Write("/Plot/PastDaysNb", PastDaysNb);

    int alarmsReturnPeriod;
    int alarmsReturnPeriodSlct = m_ChoiceAlarmsReturnPeriod->GetSelection();
    switch (alarmsReturnPeriodSlct)
    {
        case 0:
            alarmsReturnPeriod = 2;
            break;
        case 1:
            alarmsReturnPeriod = 5;
            break;
        case 2:
            alarmsReturnPeriod = 10;
            break;
        case 3:
            alarmsReturnPeriod = 20;
            break;
        case 4:
            alarmsReturnPeriod = 50;
            break;
        case 5:
            alarmsReturnPeriod = 100;
            break;
        default:
            alarmsReturnPeriod = 10;
    }
    pConfigViewer->Write("/SidebarAlarms/ReturnPeriod", alarmsReturnPeriod);

    wxString alarmsPercentile = m_TextCtrlAlarmsPercentile->GetValue();
    double alarmsPercentileVal;
    alarmsPercentile.ToDouble(&alarmsPercentileVal);
    if (alarmsPercentileVal>1)
        alarmsPercentileVal = 0.9;
    if (alarmsPercentileVal<0)
        alarmsPercentileVal = 0.9;
    pConfigViewer->Write("/SidebarAlarms/Percentile", alarmsPercentileVal);

    // Display the paths to the catalogs on the main GUI
    GetParent()->Update();

    if (g_AppViewer)
    {
        wxDELETE(pConfigForecaster);
        pConfigViewer->Flush();
    }
    else if (g_AppForecaster)
    {
        pConfigForecaster->Flush();
        wxDELETE(pConfigViewer);
    }
    else
    {
        pConfigForecaster->Flush();
        wxDELETE(pConfigViewer);
    }
}

void asFramePreferences::OnChangeMultithreadingCheckBox( wxCommandEvent& event )
{
    if (event.GetInt()==0)
    {
        m_RadioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (m_RadioBoxProcessingMethods->GetSelection()==asMULTITHREADS)
        {
            m_RadioBoxProcessingMethods->SetSelection(asINSERT);
        }
    }
    else
    {
        m_RadioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferences::SaveAndClose( wxCommandEvent& event )
{
    SavePreferences();
    Close();
}

void asFramePreferences::ApplyChanges( wxCommandEvent& event )
{
    SavePreferences();
}
