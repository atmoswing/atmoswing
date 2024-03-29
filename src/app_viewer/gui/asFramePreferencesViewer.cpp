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

#include "asFramePreferencesViewer.h"

asFramePreferencesViewer::asFramePreferencesViewer(wxWindow* parent, asWorkspace* workspace, wxWindowID id)
    : asFramePreferencesViewerVirtual(parent, id),
      m_workspace(workspace) {
    SetLabel(_("Preferences"));

    LoadPreferences();
    Fit();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesViewer::CloseFrame(wxCommandEvent& event) {
    Close();
}

void asFramePreferencesViewer::Update() {
    LoadPreferences();
}

void asFramePreferencesViewer::LoadPreferences() {
    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = m_notebookBase->GetThemeBackgroundColour();
    if (col.IsOk()) {
        m_dirPickerForecastResults->SetBackgroundColour(col);
    }

    /*
     * Workspace
     */

    // Directories
    m_dirPickerForecastResults->SetPath(m_workspace->GetForecastsDirectory());

    // Forecast display
    wxString colorbarMaxValue = asStrF("%g", m_workspace->GetColorbarMaxValue());
    m_textCtrlColorbarMaxValue->SetValue(colorbarMaxValue);
    wxString pastDaysNb = asStrF("%d", m_workspace->GetTimeSeriesPlotPastDaysNb());
    m_textCtrlPastDaysNb->SetValue(pastDaysNb);

    // Alarms panel
    int alarmsReturnPeriod = m_workspace->GetAlarmsPanelReturnPeriod();
    switch (alarmsReturnPeriod) {
        case 2:
            m_choiceAlarmsReturnPeriod->SetSelection(0);
            break;
        case 5:
            m_choiceAlarmsReturnPeriod->SetSelection(1);
            break;
        case 10:
            m_choiceAlarmsReturnPeriod->SetSelection(2);
            break;
        case 20:
            m_choiceAlarmsReturnPeriod->SetSelection(3);
            break;
        case 50:
            m_choiceAlarmsReturnPeriod->SetSelection(4);
            break;
        case 100:
            m_choiceAlarmsReturnPeriod->SetSelection(5);
            break;
        default:
            m_choiceAlarmsReturnPeriod->SetSelection(2);
    }
    wxString alarmsQuantile = asStrF("%g", m_workspace->GetAlarmsPanelQuantile());
    m_textCtrlAlarmsQuantile->SetValue(alarmsQuantile);

    // Max length
    int maxLengthDailyVal = m_workspace->GetTimeSeriesMaxLengthDaily();
    wxString maxLengthDaily = wxEmptyString;
    if (maxLengthDailyVal > 0) {
        maxLengthDaily = asStrF("%d", maxLengthDailyVal);
    }
    m_textCtrlMaxLengthDaily->SetValue(maxLengthDaily);

    int maxLengthSubDailyVal = m_workspace->GetTimeSeriesMaxLengthSubDaily();
    wxString maxLengthSubDaily = wxEmptyString;
    if (maxLengthSubDailyVal > 0) {
        maxLengthSubDaily = asStrF("%d", maxLengthSubDailyVal);
    }
    m_textCtrlMaxLengthSubDaily->SetValue(maxLengthSubDaily);

    /*
     * Paths
     */

    m_textCtrlDatasetId1->SetValue(m_workspace->GetPredictorId(1, "Generic_ECMWF_ERA5"));
    m_dirPickerDataset1->SetPath(m_workspace->GetPredictorDir(1));
    m_textCtrlDatasetId2->SetValue(m_workspace->GetPredictorId(2, "Generic_NCEP_R1"));
    m_dirPickerDataset2->SetPath(m_workspace->GetPredictorDir(2));
    m_textCtrlDatasetId3->SetValue(m_workspace->GetPredictorId(3, "NWS_GFS"));
    m_dirPickerDataset3->SetPath(m_workspace->GetPredictorDir(3));
    m_textCtrlDatasetId4->SetValue(m_workspace->GetPredictorId(4, "ECMWF_IFS"));
    m_dirPickerDataset4->SetPath(m_workspace->GetPredictorDir(4));
    m_textCtrlDatasetId5->SetValue(m_workspace->GetPredictorId(5));
    m_dirPickerDataset5->SetPath(m_workspace->GetPredictorDir(5));
    m_textCtrlDatasetId6->SetValue(m_workspace->GetPredictorId(6));
    m_dirPickerDataset6->SetPath(m_workspace->GetPredictorDir(6));
    m_textCtrlDatasetId7->SetValue(m_workspace->GetPredictorId(7));
    m_dirPickerDataset7->SetPath(m_workspace->GetPredictorDir(7));

    /*
     * Colors
     */

    wxString dirData = asConfig::GetShareDir();
    wxString colorDir = dirData + DS + "atmoswing" + DS + "color_tables";

    m_filePickerColorZ->SetPath(pConfig->Read("/ColorTable/GeopotentialHeight", colorDir + DS + "NEO_grav_anom.act"));
    m_filePickerColorPwat->SetPath(pConfig->Read("/ColorTable/PrecipitableWater", colorDir + DS + "NEO_soil_moisture.act"));
    m_filePickerColorRh->SetPath(pConfig->Read("/ColorTable/RelativeHumidity", colorDir + DS + "NEO_soil_moisture.act"));
    m_filePickerColorSh->SetPath(pConfig->Read("/ColorTable/SpecificHumidity", colorDir + DS + "NEO_soil_moisture.act"));

    /*
     * General
     */

    // Locale
    long locale = pConfig->ReadLong("/General/Locale", (long)wxLANGUAGE_ENGLISH);
    switch (locale) {
        case (long)wxLANGUAGE_ENGLISH:
            m_choiceLocale->SetSelection(0);
            break;
        case (long)wxLANGUAGE_FRENCH:
            m_choiceLocale->SetSelection(1);
            break;
        default:
            m_choiceLocale->SetSelection(0);
    }

    // Log
    long logLevel = pConfig->ReadLong("/General/LogLevel", 1);
    if (logLevel == 1) {
        m_radioBtnLogLevel1->SetValue(true);
    } else if (logLevel == 2) {
        m_radioBtnLogLevel2->SetValue(true);
    } else if (logLevel == 3) {
        m_radioBtnLogLevel3->SetValue(true);
    } else {
        m_radioBtnLogLevel1->SetValue(true);
    }
    m_checkBoxDisplayLogWindow->SetValue(pConfig->ReadBool("/General/DisplayLogWindow", false));

    // Proxy
    m_checkBoxProxy->SetValue(pConfig->ReadBool("/Internet/UsesProxy", false));
    m_textCtrlProxyAddress->SetValue(pConfig->Read("/Internet/ProxyAddress", wxEmptyString));
    m_textCtrlProxyPort->SetValue(pConfig->Read("/Internet/ProxyPort", wxEmptyString));
    m_textCtrlProxyUser->SetValue(pConfig->Read("/Internet/ProxyUser", wxEmptyString));
    m_textCtrlProxyPasswd->SetValue(pConfig->Read("/Internet/ProxyPasswd", wxEmptyString));

    /*
     * Advanced
     */

    // Advanced options
    m_checkBoxMultiInstancesViewer->SetValue(pConfig->ReadBool("/General/MultiInstances", false));

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_staticTextUserDir->SetLabel(userpath);
    wxString logpathViewer = asConfig::GetLogDir();
    logpathViewer.Append("AtmoSwingViewer.log");
    m_staticTextLogFile->SetLabel(logpathViewer);
    m_staticTextPrefFile->SetLabel(asConfig::GetConfigFilePath("AtmoSwingViewer.ini"));
}

void asFramePreferencesViewer::SavePreferences() {
    wxBusyCursor wait;

    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * Workspace
     */

    // Directories
    wxString forecastResultsDir = m_dirPickerForecastResults->GetPath();
    m_workspace->SetForecastsDirectory(forecastResultsDir);

    // Forecast display
    wxString colorbarMaxValue = m_textCtrlColorbarMaxValue->GetValue();
    double colorbarMaxValueDouble;
    colorbarMaxValue.ToDouble(&colorbarMaxValueDouble);
    m_workspace->SetColorbarMaxValue(colorbarMaxValueDouble);
    wxString pastDaysNb = m_textCtrlPastDaysNb->GetValue();
    long pastDaysNbLong;
    if (!pastDaysNb.ToLong(&pastDaysNbLong)) {
        m_workspace->SetTimeSeriesPlotPastDaysNb(int(pastDaysNbLong));
    } else {
        m_workspace->SetTimeSeriesPlotPastDaysNb(5);
    }

    // Alarms panel
    int alarmsReturnPeriod;
    int alarmsReturnPeriodSlct = m_choiceAlarmsReturnPeriod->GetSelection();
    switch (alarmsReturnPeriodSlct) {
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
    m_workspace->SetAlarmsPanelReturnPeriod(alarmsReturnPeriod);
    wxString alarmsQuantile = m_textCtrlAlarmsQuantile->GetValue();
    double alarmsQuantileVal;
    alarmsQuantile.ToDouble(&alarmsQuantileVal);
    if (alarmsQuantileVal > 1) alarmsQuantileVal = 0.9;
    if (alarmsQuantileVal < 0) alarmsQuantileVal = 0.9;
    m_workspace->SetAlarmsPanelQuantile(alarmsQuantileVal);

    // Max length
    wxString maxLengthDaily = m_textCtrlMaxLengthDaily->GetValue();
    long maxLengthDailyLong;
    if (!maxLengthDaily.IsEmpty() && maxLengthDaily.ToLong(&maxLengthDailyLong)) {
        m_workspace->SetTimeSeriesMaxLengthDaily(int(maxLengthDailyLong));
    } else {
        m_workspace->SetTimeSeriesMaxLengthDaily(-1);
    }

    wxString maxLengthSubDaily = m_textCtrlMaxLengthSubDaily->GetValue();
    long maxLengthSubDailyLong;
    if (!maxLengthSubDaily.IsEmpty() && maxLengthSubDaily.ToLong(&maxLengthSubDailyLong)) {
        m_workspace->SetTimeSeriesMaxLengthSubDaily(int(maxLengthSubDailyLong));
    } else {
        m_workspace->SetTimeSeriesMaxLengthSubDaily(-1);
    }

    /*
     * Paths
     */

    m_workspace->ClearPredictorDirs();
    m_workspace->AddPredictorDir(m_textCtrlDatasetId1->GetValue(), m_dirPickerDataset1->GetPath());
    m_workspace->AddPredictorDir(m_textCtrlDatasetId2->GetValue(), m_dirPickerDataset2->GetPath());
    m_workspace->AddPredictorDir(m_textCtrlDatasetId3->GetValue(), m_dirPickerDataset3->GetPath());
    m_workspace->AddPredictorDir(m_textCtrlDatasetId4->GetValue(), m_dirPickerDataset4->GetPath());
    m_workspace->AddPredictorDir(m_textCtrlDatasetId5->GetValue(), m_dirPickerDataset5->GetPath());
    m_workspace->AddPredictorDir(m_textCtrlDatasetId6->GetValue(), m_dirPickerDataset6->GetPath());
    m_workspace->AddPredictorDir(m_textCtrlDatasetId7->GetValue(), m_dirPickerDataset7->GetPath());

    /*
     * Colors
     */

    pConfig->Write("/ColorTable/GeopotentialHeight", m_filePickerColorZ->GetPath());
    pConfig->Write("/ColorTable/PrecipitableWater", m_filePickerColorPwat->GetPath());
    pConfig->Write("/ColorTable/RelativeHumidity", m_filePickerColorRh->GetPath());
    pConfig->Write("/ColorTable/SpecificHumidity", m_filePickerColorSh->GetPath());

    /*
     * General
     */

    // Locale
    switch (m_choiceLocale->GetSelection()) {
        case 0:
            pConfig->Write("/General/Locale", (long)wxLANGUAGE_ENGLISH);
            break;
        case 1:
            pConfig->Write("/General/Locale", (long)wxLANGUAGE_FRENCH);
            break;
        default:
            pConfig->Write("/General/Locale", (long)wxLANGUAGE_ENGLISH);
    }

    // Log
    long logLevel = 1;
    if (m_radioBtnLogLevel1->GetValue()) {
        logLevel = 1;
    } else if (m_radioBtnLogLevel2->GetValue()) {
        logLevel = 2;
    } else if (m_radioBtnLogLevel3->GetValue()) {
        logLevel = 3;
    }
    pConfig->Write("/General/LogLevel", logLevel);
    bool displayLogWindowViewer = m_checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindowViewer);

    // Proxy
    bool checkBoxProxy = m_checkBoxProxy->GetValue();
    pConfig->Write("/Internet/UsesProxy", checkBoxProxy);
    wxString proxyAddress = m_textCtrlProxyAddress->GetValue();
    pConfig->Write("/Internet/ProxyAddress", proxyAddress);
    wxString proxyPort = m_textCtrlProxyPort->GetValue();
    pConfig->Write("/Internet/ProxyPort", proxyPort);
    wxString proxyUser = m_textCtrlProxyUser->GetValue();
    pConfig->Write("/Internet/ProxyUser", proxyUser);
    wxString proxyPasswd = m_textCtrlProxyPasswd->GetValue();
    pConfig->Write("/Internet/ProxyPasswd", proxyPasswd);

    /*
     * Advanced
     */

    // Advanced options
    bool multiViewer = m_checkBoxMultiInstancesViewer->GetValue();
    pConfig->Write("/General/MultiInstances", multiViewer);

    GetParent()->Update();
    pConfig->Flush();
    m_workspace->Save();
}

void asFramePreferencesViewer::SaveAndClose(wxCommandEvent& event) {
    SavePreferences();
    Close();
}

void asFramePreferencesViewer::ApplyChanges(wxCommandEvent& event) {
    SavePreferences();
}
