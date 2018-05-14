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

asFramePreferencesViewer::asFramePreferencesViewer(wxWindow *parent, asWorkspace *workspace, wxWindowID id)
        : asFramePreferencesViewerVirtual(parent, id),
          m_workspace(workspace)
{
    LoadPreferences();
    Fit();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesViewer::CloseFrame(wxCommandEvent &event)
{
    Close();
}

void asFramePreferencesViewer::Update()
{
    LoadPreferences();
}

void asFramePreferencesViewer::LoadPreferences()
{
    wxConfigBase *pConfig;
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
    wxString colorbarMaxValue = wxString::Format("%g", m_workspace->GetColorbarMaxValue());
    m_textCtrlColorbarMaxValue->SetValue(colorbarMaxValue);
    wxString pastDaysNb = wxString::Format("%d", m_workspace->GetTimeSeriesPlotPastDaysNb());
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
    wxString alarmsQuantile = wxString::Format("%g", m_workspace->GetAlarmsPanelQuantile());
    m_textCtrlAlarmsQuantile->SetValue(alarmsQuantile);

    /*
     * General
     */

    // Log
    long defaultlogLevel = 1;
    long logLevel = pConfig->Read("/General/LogLevel", defaultlogLevel);
    if (logLevel == 1) {
        m_radioBtnLogLevel1->SetValue(true);
    } else if (logLevel == 2) {
        m_radioBtnLogLevel2->SetValue(true);
    } else if (logLevel == 3) {
        m_radioBtnLogLevel3->SetValue(true);
    } else {
        m_radioBtnLogLevel1->SetValue(true);
    }
    bool displayLogWindowViewer;
    pConfig->Read("/General/DisplayLogWindow", &displayLogWindowViewer, false);
    m_checkBoxDisplayLogWindow->SetValue(displayLogWindowViewer);

    // Proxy
    bool checkBoxProxy;
    pConfig->Read("/Internet/UsesProxy", &checkBoxProxy, false);
    m_checkBoxProxy->SetValue(checkBoxProxy);
    wxString proxyAddress = pConfig->Read("/Internet/ProxyAddress", wxEmptyString);
    m_textCtrlProxyAddress->SetValue(proxyAddress);
    wxString proxyPort = pConfig->Read("/Internet/ProxyPort", wxEmptyString);
    m_textCtrlProxyPort->SetValue(proxyPort);
    wxString proxyUser = pConfig->Read("/Internet/ProxyUser", wxEmptyString);
    m_textCtrlProxyUser->SetValue(proxyUser);
    wxString proxyPasswd = pConfig->Read("/Internet/ProxyPasswd", wxEmptyString);
    m_textCtrlProxyPasswd->SetValue(proxyPasswd);

    /*
     * Advanced
     */

    // Advanced options
    bool multiViewer;
    pConfig->Read("/General/MultiInstances", &multiViewer, false);
    m_checkBoxMultiInstancesViewer->SetValue(multiViewer);

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_staticTextUserDir->SetLabel(userpath);
    wxString logpathViewer = asConfig::GetLogDir();
    logpathViewer.Append("AtmoSwingForecaster.log");
    m_staticTextLogFile->SetLabel(logpathViewer);
    m_staticTextPrefFile->SetLabel(asConfig::GetUserDataDir() + "AtmoSwingViewer.ini");
}

void asFramePreferencesViewer::SavePreferences()
{
    wxBusyCursor wait;

    wxConfigBase *pConfig;
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
        m_workspace->SetTimeSeriesPlotPastDaysNb(static_cast<int>(pastDaysNbLong));
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
    if (alarmsQuantileVal > 1)
        alarmsQuantileVal = 0.9;
    if (alarmsQuantileVal < 0)
        alarmsQuantileVal = 0.9;
    m_workspace->SetAlarmsPanelQuantile(alarmsQuantileVal);

    /*
     * General
     */

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

void asFramePreferencesViewer::SaveAndClose(wxCommandEvent &event)
{
    SavePreferences();
    Close();
}

void asFramePreferencesViewer::ApplyChanges(wxCommandEvent &event)
{
    SavePreferences();
}
