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

#include "asFramePreferencesForecaster.h"


asFramePreferencesForecaster::asFramePreferencesForecaster(wxWindow *parent, asBatchForecasts *batchForecasts,
                                                           wxWindowID id)
        : asFramePreferencesForecasterVirtual(parent, id),
          m_batchForecasts(batchForecasts)
{
    LoadPreferences();
    Fit();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesForecaster::CloseFrame(wxCommandEvent &event)
{
    Close();
}

void asFramePreferencesForecaster::Update()
{
    LoadPreferences();
}

void asFramePreferencesForecaster::LoadPreferences()
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = m_notebookBase->GetThemeBackgroundColour();
    if (col.IsOk()) {
        m_dirPickerPredictandDB->SetBackgroundColour(col);
        m_dirPickerForecastResults->SetBackgroundColour(col);
        m_dirPickerForecastResultsExports->SetBackgroundColour(col);
        m_dirPickerParameters->SetBackgroundColour(col);
        m_dirPickerArchivePredictors->SetBackgroundColour(col);
        m_dirPickerRealtimePredictorSaving->SetBackgroundColour(col);
    }

    /*
     * Batch file properties
     */

    // Paths
    m_dirPickerPredictandDB->SetPath(m_batchForecasts->GetPredictandDBDirectory());
    m_dirPickerForecastResults->SetPath(m_batchForecasts->GetForecastsOutputDirectory());
    m_dirPickerForecastResultsExports->SetPath(m_batchForecasts->GetExportsOutputDirectory());
    m_dirPickerRealtimePredictorSaving->SetPath(m_batchForecasts->GetPredictorsRealtimeDirectory());
    m_dirPickerArchivePredictors->SetPath(m_batchForecasts->GetPredictorsArchiveDirectory());
    m_dirPickerParameters->SetPath(m_batchForecasts->GetParametersFileDirectory());

    // Exports
    m_checkBoxExportSyntheticXml->SetValue(m_batchForecasts->ExportSyntheticXml());

    /*
     * General
     */

    // Log
    long logLevelForecaster = pConfig->ReadLong("/General/LogLevel", 1);
    if (logLevelForecaster == 1) {
        m_radioBtnLogLevel1->SetValue(true);
    } else if (logLevelForecaster == 2) {
        m_radioBtnLogLevel2->SetValue(true);
    } else if (logLevelForecaster == 3) {
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

    // GUI options
    long guiOptions = pConfig->ReadLong("/General/GuiOptions", 1l);
    m_radioBoxGui->SetSelection((int) guiOptions);
    if (guiOptions == 0) {
        g_silentMode = true;
    } else {
        g_silentMode = false;
        g_verboseMode = false;
        if (guiOptions == 2l) {
            g_verboseMode = true;
        }
    }

    // Downloads
    m_textCtrlMaxPrevStepsNb->SetValue(pConfig->Read("/Internet/MaxPreviousStepsNb", "5"));
    m_textCtrlMaxRequestsNb->SetValue(pConfig->Read("/Internet/ParallelRequestsNb", "5"));
    m_checkBoxRestrictDownloads->SetValue(pConfig->ReadBool("/Internet/RestrictDownloads", true));

    // Advanced options
    g_responsive = pConfig->ReadBool("/General/Responsive", true);
    m_checkBoxResponsiveness->SetValue(g_responsive);
    m_checkBoxMultiInstancesForecaster->SetValue(pConfig->ReadBool("/General/MultiInstances", false));

    // Multithreading
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    m_checkBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads == -1)
        maxThreads = 2;
    m_textCtrlThreadsNb->SetValue(pConfig->Read("/Processing/ThreadsNb", wxString::Format("%d", maxThreads)));
    m_sliderThreadsPriority->SetValue(pConfig->ReadLong("/Processing/ThreadsPriority", 95l));

    // Processing
    long processingMethod = pConfig->ReadLong("/Processing/Method", (long) asMULTITHREADS);
    if (!allowMultithreading) {
        m_radioBoxProcessingMethods->Enable(0, false);
        if (processingMethod == (long) asMULTITHREADS) {
            processingMethod = (long) asSTANDARD;
        }
    } else {
        m_radioBoxProcessingMethods->Enable(0, true);
    }
    m_radioBoxProcessingMethods->SetSelection((int) processingMethod);

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_staticTextUserDir->SetLabel(userpath);
    wxString logpathForecaster = asConfig::GetLogDir();
    logpathForecaster.Append("AtmoSwingForecaster.log");
    m_staticTextLogFile->SetLabel(logpathForecaster);
    m_staticTextPrefFile->SetLabel(asConfig::GetUserDataDir() + "AtmoSwingForecaster.ini");

}

void asFramePreferencesForecaster::SavePreferences()
{
    wxBusyCursor wait;

    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * Batch file properties
     */

    // Paths
    m_batchForecasts->SetPredictandDBDirectory(m_dirPickerPredictandDB->GetPath());
    m_batchForecasts->SetForecastsOutputDirectory(m_dirPickerForecastResults->GetPath());
    m_batchForecasts->SetExportsOutputDirectory(m_dirPickerForecastResultsExports->GetPath());
    m_batchForecasts->SetPredictorsRealtimeDirectory(m_dirPickerRealtimePredictorSaving->GetPath());
    m_batchForecasts->SetPredictorsArchiveDirectory(m_dirPickerArchivePredictors->GetPath());
    m_batchForecasts->SetParametersFileDirectory(m_dirPickerParameters->GetPath());

    // Exports
    m_batchForecasts->SetExportSyntheticXml(m_checkBoxExportSyntheticXml->GetValue());

    /*
     * General
     */

    // Log
    long logLevelForecaster = 1;
    if (m_radioBtnLogLevel1->GetValue()) {
        logLevelForecaster = 1;
    } else if (m_radioBtnLogLevel2->GetValue()) {
        logLevelForecaster = 2;
    } else if (m_radioBtnLogLevel3->GetValue()) {
        logLevelForecaster = 3;
    }
    pConfig->Write("/General/LogLevel", logLevelForecaster);
    bool displayLogWindowForecaster = m_checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindowForecaster);

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

    // GUI options
    auto guiOptions = (long) m_radioBoxGui->GetSelection();
    pConfig->Write("/General/GuiOptions", guiOptions);
    if (guiOptions == 0) {
        g_silentMode = true;
    } else {
        g_silentMode = false;
        g_verboseMode = false;
        if (guiOptions == 2l) {
            g_verboseMode = true;
        }
    }

    // Downloads
    wxString internetMaxPrevStepsNb = m_textCtrlMaxPrevStepsNb->GetValue();
    if (!internetMaxPrevStepsNb.IsNumber())
        internetMaxPrevStepsNb = "5";
    pConfig->Write("/Internet/MaxPreviousStepsNb", internetMaxPrevStepsNb);
    wxString internetParallelRequestsNb = m_textCtrlMaxRequestsNb->GetValue();
    if (!internetParallelRequestsNb.IsNumber())
        internetParallelRequestsNb = "5";
    pConfig->Write("/Internet/ParallelRequestsNb", internetParallelRequestsNb);
    bool restrictDownloads = m_checkBoxRestrictDownloads->GetValue();
    pConfig->Write("/Internet/RestrictDownloads", restrictDownloads);

    // Advanced options
    bool responsive = m_checkBoxResponsiveness->GetValue();
    pConfig->Write("/General/Responsive", responsive);
    g_responsive = responsive;

    bool multiForecaster = m_checkBoxMultiInstancesForecaster->GetValue();
    pConfig->Write("/General/MultiInstances", multiForecaster);

    // Multithreading
    bool allowMultithreading = m_checkBoxAllowMultithreading->GetValue();
    pConfig->Write("/Processing/AllowMultithreading", allowMultithreading);
    wxString processingMaxThreadNb = m_textCtrlThreadsNb->GetValue();
    if (!processingMaxThreadNb.IsNumber())
        processingMaxThreadNb = "2";
    pConfig->Write("/Processing/ThreadsNb", processingMaxThreadNb);
    auto processingThreadsPriority = (long) m_sliderThreadsPriority->GetValue();
    pConfig->Write("/Processing/ThreadsPriority", processingThreadsPriority);

    // Processing
    auto processingMethod = (long) m_radioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && processingMethod == (long) asMULTITHREADS) {
        processingMethod = (long) asSTANDARD;
    }
    pConfig->Write("/Processing/Method", processingMethod);

    if (GetParent() != nullptr) {
        GetParent()->Update();
    }

    pConfig->Flush();
    m_batchForecasts->Save();
}

void asFramePreferencesForecaster::OnChangeMultithreadingCheckBox(wxCommandEvent &event)
{
    if (event.GetInt() == 0) {
        m_radioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (m_radioBoxProcessingMethods->GetSelection() == asMULTITHREADS) {
            m_radioBoxProcessingMethods->SetSelection(asSTANDARD);
        }
    } else {
        m_radioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferencesForecaster::SaveAndClose(wxCommandEvent &event)
{
    SavePreferences();
    Close();
}

void asFramePreferencesForecaster::ApplyChanges(wxCommandEvent &event)
{
    SavePreferences();
}
