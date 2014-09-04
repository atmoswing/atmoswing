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

#include "asFramePreferencesViewer.h"

#include "wx/fileconf.h"
#include "wx/thread.h"

asFramePreferencesViewer::asFramePreferencesViewer( wxWindow* parent, wxWindowID id )
:
asFramePreferencesViewerVirtual( parent, id )
{
    LoadPreferences();
    Fit();

        // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesViewer::CloseFrame( wxCommandEvent& event )
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
    wxColour col = m_NotebookBase->GetThemeBackgroundColour();
    if (col.IsOk())
    {
        m_DirPickerForecastResults->SetBackgroundColour(col);
    }

    /*
     * Workspace
     */

    // Directories
    wxString ForecastResultsDir = pConfig->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDocumentsDir()+"AtmoSwing"+DS+"Forecasts");
    m_DirPickerForecastResults->SetPath(ForecastResultsDir);

    // Forecast display
    wxString ColorbarMaxValue = pConfig->Read("/GIS/ColorbarMaxValue", "50");
    m_TextCtrlColorbarMaxValue->SetValue(ColorbarMaxValue);
    wxString PastDaysNb = pConfig->Read("/Plot/PastDaysNb", "3");
    m_TextCtrlPastDaysNb->SetValue(PastDaysNb);

    // Alarms panel
    int alarmsReturnPeriod;
    pConfig->Read("/SidebarAlarms/ReturnPeriod", &alarmsReturnPeriod, 10);
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
            pConfig->Write("/SidebarAlarms/ReturnPeriod", 10);
    }
    wxString alarmsPercentile = pConfig->Read("/SidebarAlarms/Percentile", "0.9");
    m_TextCtrlAlarmsPercentile->SetValue(alarmsPercentile);

    /*
     * General
     */

    // Log
    long defaultLogLevelViewer = 1; // = selection +1
    long logLevelViewer = pConfig->Read("/Standard/LogLevel", defaultLogLevelViewer);
    m_RadioBoxLogLevel->SetSelection((int)logLevelViewer-1);
    bool displayLogWindowViewer;
    pConfig->Read("/Standard/DisplayLogWindow", &displayLogWindowViewer, false);
    m_CheckBoxDisplayLogWindow->SetValue(displayLogWindowViewer);

    // Proxy
    bool checkBoxProxy;
    pConfig->Read("/Internet/UsesProxy", &checkBoxProxy, false);
    m_CheckBoxProxy->SetValue(checkBoxProxy);
    wxString ProxyAddress = pConfig->Read("/Internet/ProxyAddress", wxEmptyString);
    m_TextCtrlProxyAddress->SetValue(ProxyAddress);
    wxString ProxyPort = pConfig->Read("/Internet/ProxyPort", wxEmptyString);
    m_TextCtrlProxyPort->SetValue(ProxyPort);
    wxString ProxyUser = pConfig->Read("/Internet/ProxyUser", wxEmptyString);
    m_TextCtrlProxyUser->SetValue(ProxyUser);
    wxString ProxyPasswd = pConfig->Read("/Internet/ProxyPasswd", wxEmptyString);
    m_TextCtrlProxyPasswd->SetValue(ProxyPasswd);

    /*
     * Advanced
     */

    // Advanced options
    bool multiViewer;
    pConfig->Read("/Standard/MultiInstances", &multiViewer, false);
    m_CheckBoxMultiInstancesViewer->SetValue(multiViewer);
    
    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_StaticTextUserDir->SetLabel(userpath);
    wxString logpathViewer = asConfig::GetLogDir();
    logpathViewer.Append("AtmoSwingForecaster.log");
    m_StaticTextLogFile->SetLabel(logpathViewer);
    m_StaticTextPrefFile->SetLabel(asConfig::GetUserDataDir("AtmoSwing viewer")+"AtmoSwing.ini");
}

void asFramePreferencesViewer::SavePreferences( )
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * Workspace
     */
    
    // Directories
    wxString ForecastResultsDir = m_DirPickerForecastResults->GetPath();
    pConfig->Write("/StandardPaths/ForecastResultsDir", ForecastResultsDir);

    // Forecast display
    wxString ColorbarMaxValue = m_TextCtrlColorbarMaxValue->GetValue();
    pConfig->Write("/GIS/ColorbarMaxValue", ColorbarMaxValue);
    wxString PastDaysNb = m_TextCtrlPastDaysNb->GetValue();
    pConfig->Write("/Plot/PastDaysNb", PastDaysNb);

    // Alarms panel
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
    pConfig->Write("/SidebarAlarms/ReturnPeriod", alarmsReturnPeriod);
    wxString alarmsPercentile = m_TextCtrlAlarmsPercentile->GetValue();
    double alarmsPercentileVal;
    alarmsPercentile.ToDouble(&alarmsPercentileVal);
    if (alarmsPercentileVal>1)
        alarmsPercentileVal = 0.9;
    if (alarmsPercentileVal<0)
        alarmsPercentileVal = 0.9;
    pConfig->Write("/SidebarAlarms/Percentile", alarmsPercentileVal);

    /*
     * General
     */

    // Log
    long logLevelViewer = (long)m_RadioBoxLogLevel->GetSelection();
    pConfig->Write("/Standard/LogLevel", logLevelViewer+1); // = selection +1
    bool displayLogWindowViewer = m_CheckBoxDisplayLogWindow->GetValue();
    pConfig->Write("/Standard/DisplayLogWindow", displayLogWindowViewer);

    // Proxy
    bool checkBoxProxy = m_CheckBoxProxy->GetValue();
    pConfig->Write("/Internet/UsesProxy", checkBoxProxy);
    wxString ProxyAddress = m_TextCtrlProxyAddress->GetValue();
    pConfig->Write("/Internet/ProxyAddress", ProxyAddress);
    wxString ProxyPort = m_TextCtrlProxyPort->GetValue();
    pConfig->Write("/Internet/ProxyPort", ProxyPort);
    wxString ProxyUser = m_TextCtrlProxyUser->GetValue();
    pConfig->Write("/Internet/ProxyUser", ProxyUser);
    wxString ProxyPasswd = m_TextCtrlProxyPasswd->GetValue();
    pConfig->Write("/Internet/ProxyPasswd", ProxyPasswd);

    /*
     * Advanced
     */

    // Advanced options
    bool multiViewer = m_CheckBoxMultiInstancesViewer->GetValue();
    pConfig->Write("/Standard/MultiInstances", multiViewer);


    GetParent()->Update();
    pConfig->Flush();
}

void asFramePreferencesViewer::SaveAndClose( wxCommandEvent& event )
{
    SavePreferences();
    Close();
}

void asFramePreferencesViewer::ApplyChanges( wxCommandEvent& event )
{
    SavePreferences();
}
