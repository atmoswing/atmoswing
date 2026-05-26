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

#include <wx/fileconf.h>


asFramePreferencesViewer::asFramePreferencesViewer(wxWindow* parent, asWorkspace* workspace, wxWindowID id)
    : asFramePreferencesViewerVirtual(parent, id),
      _workspace(workspace) {
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
    wxColour col = _notebookBase->GetThemeBackgroundColour();
    if (col.IsOk()) {
        _dirPickerForecastResults->SetBackgroundColour(col);
    }

    /*
     * Workspace
     */

    // Directories
    _dirPickerForecastResults->SetPath(_workspace->GetForecastsDirectory());

    // Forecast display
    wxString colorbarMaxValue = asStrF("%g", _workspace->GetColorbarMaxValue());
    _textCtrlColorbarMaxValue->SetValue(colorbarMaxValue);
    wxString pastDaysNb = asStrF("%d", _workspace->GetTimeSeriesPlotPastDaysNb());
    _textCtrlPastDaysNb->SetValue(pastDaysNb);

    // Alarms panel
    int alarmsReturnPeriod = _workspace->GetAlarmsPanelReturnPeriod();
    switch (alarmsReturnPeriod) {
        case 2:
            _choiceAlarmsReturnPeriod->SetSelection(0);
            break;
        case 5:
            _choiceAlarmsReturnPeriod->SetSelection(1);
            break;
        case 10:
            _choiceAlarmsReturnPeriod->SetSelection(2);
            break;
        case 20:
            _choiceAlarmsReturnPeriod->SetSelection(3);
            break;
        case 50:
            _choiceAlarmsReturnPeriod->SetSelection(4);
            break;
        case 100:
            _choiceAlarmsReturnPeriod->SetSelection(5);
            break;
        default:
            _choiceAlarmsReturnPeriod->SetSelection(2);
    }
    wxString alarmsQuantile = asStrF("%g", _workspace->GetAlarmsPanelQuantile());
    _textCtrlAlarmsQuantile->SetValue(alarmsQuantile);

    // Max length
    int maxLengthDailyVal = _workspace->GetTimeSeriesMaxLengthDaily();
    wxString maxLengthDaily = wxEmptyString;
    if (maxLengthDailyVal > 0) {
        maxLengthDaily = asStrF("%d", maxLengthDailyVal);
    }
    _textCtrlMaxLengthDaily->SetValue(maxLengthDaily);

    int maxLengthSubDailyVal = _workspace->GetTimeSeriesMaxLengthSubDaily();
    wxString maxLengthSubDaily = wxEmptyString;
    if (maxLengthSubDailyVal > 0) {
        maxLengthSubDaily = asStrF("%d", maxLengthSubDailyVal);
    }
    _textCtrlMaxLengthSubDaily->SetValue(maxLengthSubDaily);

    /*
     * Paths
     */

    _textCtrlDatasetId1->SetValue(_workspace->GetPredictorId(1, "Generic_ECMWF_ERA5"));
    _dirPickerDataset1->SetPath(_workspace->GetPredictorDir(1));
    _textCtrlDatasetId2->SetValue(_workspace->GetPredictorId(2, "Generic_NCEP_R1"));
    _dirPickerDataset2->SetPath(_workspace->GetPredictorDir(2));
    _textCtrlDatasetId3->SetValue(_workspace->GetPredictorId(3, "NWS_GFS"));
    _dirPickerDataset3->SetPath(_workspace->GetPredictorDir(3));
    _textCtrlDatasetId4->SetValue(_workspace->GetPredictorId(4, "ECMWF_IFS"));
    _dirPickerDataset4->SetPath(_workspace->GetPredictorDir(4));
    _textCtrlDatasetId5->SetValue(_workspace->GetPredictorId(5));
    _dirPickerDataset5->SetPath(_workspace->GetPredictorDir(5));
    _textCtrlDatasetId6->SetValue(_workspace->GetPredictorId(6));
    _dirPickerDataset6->SetPath(_workspace->GetPredictorDir(6));
    _textCtrlDatasetId7->SetValue(_workspace->GetPredictorId(7));
    _dirPickerDataset7->SetPath(_workspace->GetPredictorDir(7));

    /*
     * Colors
     */

    wxString dirData = asConfig::GetShareDir();
    wxString colorDir = dirData + DS + "atmoswing" + DS + "color_tables";

    _filePickerColorZ->SetPath(pConfig->Read("/ColorTable/GeopotentialHeight", colorDir + DS + "NEO_grav_anom.act"));
    _filePickerColorPwat->SetPath(
        pConfig->Read("/ColorTable/PrecipitableWater", colorDir + DS + "NEO_soil_moisture.act"));
    _filePickerColorRh->SetPath(pConfig->Read("/ColorTable/RelativeHumidity", colorDir + DS + "NEO_soil_moisture.act"));
    _filePickerColorSh->SetPath(pConfig->Read("/ColorTable/SpecificHumidity", colorDir + DS + "NEO_soil_moisture.act"));

    /*
     * General
     */

    // Locale
    long locale = pConfig->ReadLong("/General/Locale", (long)wxLANGUAGE_ENGLISH);
    switch (locale) {
        case (long)wxLANGUAGE_ENGLISH:
            _choiceLocale->SetSelection(0);
            break;
        case (long)wxLANGUAGE_FRENCH:
            _choiceLocale->SetSelection(1);
            break;
        default:
            _choiceLocale->SetSelection(0);
    }

    // Log
    long logLevel = pConfig->ReadLong("/General/LogLevel", 1);
    if (logLevel == 1) {
        _radioBtnLogLevel1->SetValue(true);
    } else if (logLevel == 2) {
        _radioBtnLogLevel2->SetValue(true);
    } else if (logLevel == 3) {
        _radioBtnLogLevel3->SetValue(true);
    } else {
        _radioBtnLogLevel1->SetValue(true);
    }
    _checkBoxDisplayLogWindow->SetValue(pConfig->ReadBool("/General/DisplayLogWindow", false));

    // Proxy
    _checkBoxProxy->SetValue(pConfig->ReadBool("/Internet/UsesProxy", false));
    _textCtrlProxyAddress->SetValue(pConfig->Read("/Internet/ProxyAddress", wxEmptyString));
    _textCtrlProxyPort->SetValue(pConfig->Read("/Internet/ProxyPort", wxEmptyString));
    _textCtrlProxyUser->SetValue(pConfig->Read("/Internet/ProxyUser", wxEmptyString));
    _textCtrlProxyPasswd->SetValue(pConfig->Read("/Internet/ProxyPasswd", wxEmptyString));

    /*
     * Advanced
     */

    // Advanced options
    _checkBoxMultiInstancesViewer->SetValue(pConfig->ReadBool("/General/MultiInstances", false));

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    _staticTextUserDir->SetLabel(userpath);
    wxString logpathViewer = asConfig::GetLogDir();
    logpathViewer.Append("AtmoSwingViewer.log");
    _staticTextLogFile->SetLabel(logpathViewer);
    _staticTextPrefFile->SetLabel(asConfig::GetConfigFilePath("AtmoSwingViewer.ini"));
}

void asFramePreferencesViewer::SavePreferences() {
    wxBusyCursor wait;

    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * Workspace
     */

    // Directories
    wxString forecastResultsDir = _dirPickerForecastResults->GetPath();
    _workspace->SetForecastsDirectory(forecastResultsDir);

    // Forecast display
    wxString colorbarMaxValue = _textCtrlColorbarMaxValue->GetValue();
    double colorbarMaxValueDouble;
    colorbarMaxValue.ToDouble(&colorbarMaxValueDouble);
    _workspace->SetColorbarMaxValue(colorbarMaxValueDouble);
    wxString pastDaysNb = _textCtrlPastDaysNb->GetValue();
    long pastDaysNbLong;
    if (!pastDaysNb.ToLong(&pastDaysNbLong)) {
        _workspace->SetTimeSeriesPlotPastDaysNb(int(pastDaysNbLong));
    } else {
        _workspace->SetTimeSeriesPlotPastDaysNb(5);
    }

    // Alarms panel
    int alarmsReturnPeriod;
    int alarmsReturnPeriodSlct = _choiceAlarmsReturnPeriod->GetSelection();
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
    _workspace->SetAlarmsPanelReturnPeriod(alarmsReturnPeriod);
    wxString alarmsQuantile = _textCtrlAlarmsQuantile->GetValue();
    double alarmsQuantileVal;
    alarmsQuantile.ToDouble(&alarmsQuantileVal);
    if (alarmsQuantileVal > 1) alarmsQuantileVal = 0.9;
    if (alarmsQuantileVal < 0) alarmsQuantileVal = 0.9;
    _workspace->SetAlarmsPanelQuantile(alarmsQuantileVal);

    // Max length
    wxString maxLengthDaily = _textCtrlMaxLengthDaily->GetValue();
    long maxLengthDailyLong;
    if (!maxLengthDaily.IsEmpty() && maxLengthDaily.ToLong(&maxLengthDailyLong)) {
        _workspace->SetTimeSeriesMaxLengthDaily(int(maxLengthDailyLong));
    } else {
        _workspace->SetTimeSeriesMaxLengthDaily(-1);
    }

    wxString maxLengthSubDaily = _textCtrlMaxLengthSubDaily->GetValue();
    long maxLengthSubDailyLong;
    if (!maxLengthSubDaily.IsEmpty() && maxLengthSubDaily.ToLong(&maxLengthSubDailyLong)) {
        _workspace->SetTimeSeriesMaxLengthSubDaily(int(maxLengthSubDailyLong));
    } else {
        _workspace->SetTimeSeriesMaxLengthSubDaily(-1);
    }

    /*
     * Paths
     */

    _workspace->ClearPredictorDirs();
    _workspace->AddPredictorDir(_textCtrlDatasetId1->GetValue(), _dirPickerDataset1->GetPath());
    _workspace->AddPredictorDir(_textCtrlDatasetId2->GetValue(), _dirPickerDataset2->GetPath());
    _workspace->AddPredictorDir(_textCtrlDatasetId3->GetValue(), _dirPickerDataset3->GetPath());
    _workspace->AddPredictorDir(_textCtrlDatasetId4->GetValue(), _dirPickerDataset4->GetPath());
    _workspace->AddPredictorDir(_textCtrlDatasetId5->GetValue(), _dirPickerDataset5->GetPath());
    _workspace->AddPredictorDir(_textCtrlDatasetId6->GetValue(), _dirPickerDataset6->GetPath());
    _workspace->AddPredictorDir(_textCtrlDatasetId7->GetValue(), _dirPickerDataset7->GetPath());

    /*
     * Colors
     */

    pConfig->Write("/ColorTable/GeopotentialHeight", _filePickerColorZ->GetPath());
    pConfig->Write("/ColorTable/PrecipitableWater", _filePickerColorPwat->GetPath());
    pConfig->Write("/ColorTable/RelativeHumidity", _filePickerColorRh->GetPath());
    pConfig->Write("/ColorTable/SpecificHumidity", _filePickerColorSh->GetPath());

    /*
     * General
     */

    // Locale
    switch (_choiceLocale->GetSelection()) {
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
    if (_radioBtnLogLevel1->GetValue()) {
        logLevel = 1;
    } else if (_radioBtnLogLevel2->GetValue()) {
        logLevel = 2;
    } else if (_radioBtnLogLevel3->GetValue()) {
        logLevel = 3;
    }
    pConfig->Write("/General/LogLevel", logLevel);
    bool displayLogWindowViewer = _checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindowViewer);

    // Proxy
    bool checkBoxProxy = _checkBoxProxy->GetValue();
    pConfig->Write("/Internet/UsesProxy", checkBoxProxy);
    wxString proxyAddress = _textCtrlProxyAddress->GetValue();
    pConfig->Write("/Internet/ProxyAddress", proxyAddress);
    wxString proxyPort = _textCtrlProxyPort->GetValue();
    pConfig->Write("/Internet/ProxyPort", proxyPort);
    wxString proxyUser = _textCtrlProxyUser->GetValue();
    pConfig->Write("/Internet/ProxyUser", proxyUser);
    wxString proxyPasswd = _textCtrlProxyPasswd->GetValue();
    pConfig->Write("/Internet/ProxyPasswd", proxyPasswd);

    /*
     * Advanced
     */

    // Advanced options
    bool multiViewer = _checkBoxMultiInstancesViewer->GetValue();
    pConfig->Write("/General/MultiInstances", multiViewer);

    GetParent()->Update();
    pConfig->Flush();
    _workspace->Save();
}

void asFramePreferencesViewer::SaveAndClose(wxCommandEvent& event) {
    SavePreferences();
    Close();
}

void asFramePreferencesViewer::ApplyChanges(wxCommandEvent& event) {
    SavePreferences();
}
