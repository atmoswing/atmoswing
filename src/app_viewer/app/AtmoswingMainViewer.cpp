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
    : asFrameForecast(frame)
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

