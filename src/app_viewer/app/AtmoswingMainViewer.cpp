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

#include "AtmoswingMainViewer.h"

#include "asGeo.h"
#include "asGeoArea.h"
#include "asGeoAreaRegularGrid.h"
#include "asGeoAreaComposite.h"
#include "asGeoAreaCompositeGrid.h"
#include "asGeoPoint.h"
#include "asTime.h"
#include "asTimeArray.h"
#include "asFileNetcdf.h"
#include "asFileXml.h"
#include "asFileAscii.h"
#include "asFileDat.h"
#include "asConfig.h"


AtmoswingFrameViewer::AtmoswingFrameViewer(wxFrame *frame)
    : asFrameForecastRings(frame)
{
#if wxUSE_STATUSBAR
    wxLogStatus(_("Welcome to AtmoSwing %s."), asVersion::GetFullString().c_str());
#endif

    // Config file
    wxConfigBase *pConfig = wxFileConfig::Get();

    // Set default options
    SetDefaultOptions();

    // Create log window and file
    bool displayLogWindow;
    pConfig->Read("/Standard/DisplayLogWindow", &displayLogWindow, false);
    m_LogWindow = new asLogWindow(this, _("AtmoSwing log window"), displayLogWindow);
    Log().CreateFile("AtmoSwingViewer.log");
}

void AtmoswingFrameViewer::SetDefaultOptions()
{
    wxConfigBase *pConfig = wxFileConfig::Get();

    // General
    long guiOptions = pConfig->Read("/Standard/GuiOptions", 1l);
    pConfig->Write("/Standard/GuiOptions", guiOptions);
    bool responsive;
    pConfig->Read("/Standard/Responsive", &responsive, true);
    pConfig->Write("/Standard/Responsive", responsive);
    long defaultLogLevel = 1; // = selection +1
    long logLevel = pConfig->Read("/Standard/LogLevelViewer", defaultLogLevel);
    pConfig->Write("/Standard/LogLevelViewer", logLevel);
    bool displayLogWindow;
    pConfig->Read("/Standard/DisplayLogWindowViewer", &displayLogWindow, false);
    pConfig->Write("/Standard/DisplayLogWindowViewer", displayLogWindow);
    // Multithreading
    bool allowMultithreading;
    pConfig->Read("/Standard/AllowMultithreading", &allowMultithreading, true);
    pConfig->Write("/Standard/AllowMultithreading", allowMultithreading);
    // Set the number of threads
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads==-1) maxThreads = 2;
    wxString maxThreadsStr = wxString::Format("%d", maxThreads);
    wxString ProcessingMaxThreadNb = pConfig->Read("/Standard/ProcessingMaxThreadNb", maxThreadsStr);
    pConfig->Write("/Standard/ProcessingMaxThreadNb", ProcessingMaxThreadNb);

    // Paths
    wxString dirConfig = asConfig::GetDataDir()+"config"+DS;
    wxString dirData = asConfig::GetDataDir()+"data"+DS;
    wxString CatalogPredictorsArchiveFilePath = pConfig->Read("/StandardPaths/CatalogPredictorsArchiveFilePath", dirConfig+"CatalogPredictorsArchive.xml");
    pConfig->Write("/StandardPaths/CatalogPredictorsArchiveFilePath", CatalogPredictorsArchiveFilePath);
    wxString CatalogPredictorsRealtimeFilePath = pConfig->Read("/StandardPaths/CatalogPredictorsRealtimeFilePath", dirConfig+"CatalogPredictorsRealtime.xml");
    pConfig->Write("/StandardPaths/CatalogPredictorsRealtimeFilePath", CatalogPredictorsRealtimeFilePath);
    wxString CatalogPredictandsFilePath = pConfig->Read("/StandardPaths/CatalogPredictandsFilePath", dirConfig+"CatalogPredictands.xml");
    pConfig->Write("/StandardPaths/CatalogPredictandsFilePath", CatalogPredictandsFilePath);
    wxString ForecastResultsDir = pConfig->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDocumentsDir()+DS+"AtmoSwing"+DS+"Forecasts");
    pConfig->Write("/StandardPaths/ForecastResultsDir", ForecastResultsDir);
    wxString RealtimePredictorSavingDir = pConfig->Read("/StandardPaths/RealtimePredictorSavingDir", asConfig::GetDocumentsDir()+"AtmoSwing"+DS+"Predictors");
    pConfig->Write("/StandardPaths/RealtimePredictorSavingDir", RealtimePredictorSavingDir);
    wxString ForecasterPath = pConfig->Read("/StandardPaths/ForecasterPath", asConfig::GetDataDir()+"AtmoSwingForecaster.exe");
    pConfig->Write("/StandardPaths/ForecasterPath", ForecasterPath);
    wxString ViewerPath = pConfig->Read("/StandardPaths/ViewerPath", asConfig::GetDataDir()+"AtmoSwingViewer.exe");
    pConfig->Write("/StandardPaths/ViewerPath", ViewerPath);

    // GIS
        // Hillshade
    wxString LayerHillshadeFilePath = pConfig->Read("/GIS/LayerHillshadeFilePath", dirData+"gis"+DS+"Local"+DS+"hillshade"+DS+"hdr.adf");
    pConfig->Write("/GIS/LayerHillshadeFilePath", LayerHillshadeFilePath);
    wxString LayerHillshadeTransp = pConfig->Read("/GIS/LayerHillshadeTransp", "0");
    pConfig->Write("/GIS/LayerHillshadeTransp", LayerHillshadeTransp);
    bool LayerHillshadeVisibility;
    pConfig->Read("/GIS/LayerHillshadeVisibility", &LayerHillshadeVisibility, true);
    pConfig->Write("/GIS/LayerHillshadeVisibility", LayerHillshadeVisibility);
        // Catchments
    wxString LayerCatchmentsFilePath = pConfig->Read("/GIS/LayerCatchmentsFilePath", dirData+"gis"+DS+"Local"+DS+"catchments.shp");
    pConfig->Write("/GIS/LayerCatchmentsFilePath", LayerCatchmentsFilePath);
    wxString LayerCatchmentsTransp = pConfig->Read("/GIS/LayerCatchmentsTransp", "50");
    pConfig->Write("/GIS/LayerCatchmentsTransp", LayerCatchmentsTransp);
    long LayerCatchmentsColor = (long)0x0000fffc;
    LayerCatchmentsColor = pConfig->Read("/GIS/LayerCatchmentsColor", LayerCatchmentsColor);
    pConfig->Write("/GIS/LayerCatchmentsColor", LayerCatchmentsColor);
    wxColour colorCatchments;
    colorCatchments.SetRGB((wxUint32)LayerCatchmentsColor);
    wxString LayerCatchmentsSize = pConfig->Read("/GIS/LayerCatchmentsSize", "1");
    pConfig->Write("/GIS/LayerCatchmentsSize",LayerCatchmentsSize);
    bool LayerCatchmentsVisibility;
    pConfig->Read("/GIS/LayerCatchmentsVisibility", &LayerCatchmentsVisibility, false);
    pConfig->Write("/GIS/LayerCatchmentsVisibility",LayerCatchmentsVisibility);
        // Hydro
    wxString LayerHydroFilePath = pConfig->Read("/GIS/LayerHydroFilePath", dirData+"gis"+DS+"Local"+DS+"hydrography.shp");
    pConfig->Write("/GIS/LayerHydroFilePath", LayerHydroFilePath);
    wxString LayerHydroTransp = pConfig->Read("/GIS/LayerHydroTransp", "40");
    pConfig->Write("/GIS/LayerHydroTransp", LayerHydroTransp);
    long LayerHydroColor = (long)0x00c81616;
    LayerHydroColor = pConfig->Read("/GIS/LayerHydroColor", LayerHydroColor);
    pConfig->Write("/GIS/LayerHydroColor", LayerHydroColor);
    wxColour colorHydro;
    colorHydro.SetRGB((wxUint32)LayerHydroColor);
    wxString LayerHydroSize = pConfig->Read("/GIS/LayerHydroSize", "1");
    pConfig->Write("/GIS/LayerHydroSize", LayerHydroSize);
    bool LayerHydroVisibility;
    pConfig->Read("/GIS/LayerHydroVisibility", &LayerHydroVisibility, true);
    pConfig->Write("/GIS/LayerHydroVisibility", LayerHydroVisibility);
        // Lakes
    wxString LayerLakesFilePath = pConfig->Read("/GIS/LayerLakesFilePath", dirData+"gis"+DS+"Local"+DS+"lakes.shp");
    pConfig->Write("/GIS/LayerLakesFilePath", LayerLakesFilePath);
    wxString LayerLakesTransp = pConfig->Read("/GIS/LayerLakesTransp", "40");
    pConfig->Write("/GIS/LayerLakesTransp",LayerLakesTransp );
    long LayerLakesColor = (long)0x00d4c92a;
    LayerLakesColor = pConfig->Read("/GIS/LayerLakesColor", LayerLakesColor);
    pConfig->Write("/GIS/LayerLakesColor", LayerLakesColor);
    wxColour colorLakes;
    colorLakes.SetRGB((wxUint32)LayerLakesColor);
    bool LayerLakesVisibility;
    pConfig->Read("/GIS/LayerLakesVisibility", &LayerLakesVisibility, true);
    pConfig->Write("/GIS/LayerLakesVisibility", LayerLakesVisibility);
        // Basemap
    wxString LayerBasemapFilePath = pConfig->Read("/GIS/LayerBasemapFilePath", dirData+"gis"+DS+"Local"+DS+"basemap.shp");
    pConfig->Write("/GIS/LayerBasemapFilePath", LayerBasemapFilePath);
    wxString LayerBasemapTransp = pConfig->Read("/GIS/LayerBasemapTransp", "50");
    pConfig->Write("/GIS/LayerBasemapTransp", LayerBasemapTransp);
    bool LayerBasemapVisibility;
    pConfig->Read("/GIS/LayerBasemapVisibility", &LayerBasemapVisibility, false);
    pConfig->Write("/GIS/LayerBasemapVisibility", LayerBasemapVisibility);
        // Continents
    wxString LayerContinentsFilePath = pConfig->Read("/GIS/LayerContinentsFilePath", dirData+"gis"+DS+"World"+DS+"continents.shp");
    pConfig->Write("/GIS/LayerContinentsFilePath", LayerContinentsFilePath);
    wxString LayerContinentsTransp = pConfig->Read("/GIS/LayerContinentsTransp", "50");
    pConfig->Write("/GIS/LayerContinentsTransp", LayerContinentsTransp);
    long LayerContinentsColor = (long)0x99999999;
    LayerContinentsColor = pConfig->Read("/GIS/LayerContinentsColor", LayerContinentsColor);
    pConfig->Write("/GIS/LayerContinentsColor", LayerContinentsColor);
    wxColour colorContinents;
    colorContinents.SetRGB((wxUint32)LayerContinentsColor);
    wxString LayerContinentsSize = pConfig->Read("/GIS/LayerContinentsSize", "1");
    pConfig->Write("/GIS/LayerContinentsSize", LayerContinentsSize);
    bool LayerContinentsVisibility;
    pConfig->Read("/GIS/LayerContinentsVisibility", &LayerContinentsVisibility, true);
    pConfig->Write("/GIS/LayerContinentsVisibility", LayerContinentsVisibility);
        // Countries
    wxString LayerCountriesFilePath = pConfig->Read("/GIS/LayerCountriesFilePath", dirData+"gis"+DS+"World"+DS+"countries.shp");
    pConfig->Write("/GIS/LayerCountriesFilePath", LayerCountriesFilePath);
    wxString LayerCountriesTransp = pConfig->Read("/GIS/LayerCountriesTransp", "0");
    pConfig->Write("/GIS/LayerCountriesTransp", LayerCountriesTransp);
    long LayerCountriesColor = (long)0x77999999;
    LayerCountriesColor = pConfig->Read("/GIS/LayerCountriesColor", LayerCountriesColor);
    pConfig->Write("/GIS/LayerCountriesColor", LayerCountriesColor);
    wxColour colorCountries;
    colorCountries.SetRGB((wxUint32)LayerCountriesColor);
    wxString LayerCountriesSize = pConfig->Read("/GIS/LayerCountriesSize", "1");
    pConfig->Write("/GIS/LayerCountriesSize", LayerCountriesSize);
    bool LayerCountriesVisibility;
    pConfig->Read("/GIS/LayerCountriesVisibility", &LayerCountriesVisibility, true);
    pConfig->Write("/GIS/LayerCountriesVisibility", LayerCountriesVisibility);
        // Geogrid
    wxString LayerGeogridFilePath = pConfig->Read("/GIS/LayerGeogridFilePath", dirData+"gis"+DS+"World"+DS+"geogrid.shp");
    pConfig->Write("/GIS/LayerGeogridFilePath", LayerGeogridFilePath);
    wxString LayerGeogridTransp = pConfig->Read("/GIS/LayerGeogridTransp", "50");
    pConfig->Write("/GIS/LayerGeogridTransp", LayerGeogridTransp);
    long LayerGeogridColor = (long)0xff999999;
    LayerGeogridColor = pConfig->Read("/GIS/LayerGeogridColor", LayerGeogridColor);
    pConfig->Write("/GIS/LayerGeogridColor", LayerGeogridColor);
    wxColour colorGeogrid;
    colorGeogrid.SetRGB((wxUint32)LayerGeogridColor);
    wxString LayerGeogridSize = pConfig->Read("/GIS/LayerGeogridSize", "2");
    pConfig->Write("/GIS/LayerGeogridSize", LayerGeogridSize);
    bool LayerGeogridVisibility;
    pConfig->Read("/GIS/LayerGeogridVisibility", &LayerGeogridVisibility, false);
    pConfig->Write("/GIS/LayerGeogridVisibility", LayerGeogridVisibility);
        // LatLong
    wxString LayerLatLongFilePath = pConfig->Read("/GIS/LayerLatLongFilePath", dirData+"gis"+DS+"World"+DS+"latlong.shp");
    pConfig->Write("/GIS/LayerLatLongFilePath", LayerLatLongFilePath);
    wxString LayerLatLongTransp = pConfig->Read("/GIS/LayerLatLongTransp", "80");
    pConfig->Write("/GIS/LayerLatLongTransp", LayerLatLongTransp);
    long LayerLatLongColor = (long)0xff999999;
    LayerLatLongColor = pConfig->Read("/GIS/LayerLatLongColor", LayerLatLongColor);
    pConfig->Write("/GIS/LayerLatLongColor", LayerLatLongColor);
    wxColour colorLatLong;
    colorLatLong.SetRGB((wxUint32)LayerLatLongColor);
    wxString LayerLatLongSize = pConfig->Read("/GIS/LayerLatLongSize", "1");
    pConfig->Write("/GIS/LayerLatLongSize", LayerLatLongSize);
    bool LayerLatLongVisibility;
    pConfig->Read("/GIS/LayerLatLongVisibility", &LayerLatLongVisibility, true);
    pConfig->Write("/GIS/LayerLatLongVisibility", LayerLatLongVisibility);

    // Forecast display
    wxString ColorbarMaxValue = pConfig->Read("/GIS/ColorbarMaxValue", "50");
    pConfig->Write("/GIS/ColorbarMaxValue", ColorbarMaxValue);
    wxString PastDaysNb = pConfig->Read("/Plot/PastDaysNb", "3");
    pConfig->Write("/GIS/PastDaysNb", PastDaysNb);

    pConfig->Flush();
}

AtmoswingFrameViewer::~AtmoswingFrameViewer()
{
    //wxDELETE(m_LogWindow);
}

void AtmoswingFrameViewer::OnClose(wxCloseEvent &event)
{
    Close(true);
}

void AtmoswingFrameViewer::OnQuit(wxCommandEvent &event)
{
    Close(true);
}

void AtmoswingFrameViewer::OnShowLog( wxCommandEvent& event )
{
    wxASSERT(m_LogWindow);
    m_LogWindow->Show();
}

