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
 * Portions Copyright 2017 Pascal Horton, University of Bern.
 */

#include "asFramePreferencesDownscaler.h"

asFramePreferencesDownscaler::asFramePreferencesDownscaler(wxWindow* parent, wxWindowID id)
    : asFramePreferencesDownscalerVirtual(parent, id) {
    SetLabel(_("Preferences"));
    LoadPreferences();
    Fit();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesDownscaler::CloseFrame(wxCommandEvent& event) {
    Close();
}

void asFramePreferencesDownscaler::Update() {
    LoadPreferences();
}

void asFramePreferencesDownscaler::LoadPreferences() {
    wxBusyCursor wait;

    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = m_notebookBase->GetThemeBackgroundColour();
    if (col.IsOk()) {
        m_dirPickerPredictandDB->SetBackgroundColour(col);
        m_dirPickerIntermediateResults->SetBackgroundColour(col);
        m_dirPickerArchivePredictors->SetBackgroundColour(col);
        m_dirPickerScenarioPredictors->SetBackgroundColour(col);
    }

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
    long logLevel = pConfig->ReadLong("/General/LogLevel", 1L);
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

    // Paths
    wxString dirData = asConfig::GetDataDir() + "data" + DS;
    m_dirPickerPredictandDB->SetPath(pConfig->Read("/Paths/DataPredictandDBDir", dirData + "predictands"));
    m_dirPickerArchivePredictors->SetPath(pConfig->Read("/Paths/ArchivePredictorsDir", dirData + "predictors"));
    m_dirPickerScenarioPredictors->SetPath(pConfig->Read("/Paths/ScenarioPredictorsDir", dirData + "predictors"));

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = pConfig->ReadLong("/General/GuiOptions", 1l);
    m_radioBoxGui->SetSelection(int(guiOptions));
    if (guiOptions == 0) {
        g_silentMode = true;
    } else {
        g_silentMode = false;
        g_verboseMode = false;
        if (guiOptions == 2l) {
            g_verboseMode = true;
        }
    }

    // Advanced options
    bool responsive = pConfig->ReadBool("/General/Responsive", true);
    m_checkBoxResponsiveness->SetValue(responsive);
    g_responsive = responsive;

    // Multithreading
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    m_checkBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads == -1) maxThreads = 2;
    wxString maxThreadsStr = asStrF("%d", maxThreads);
    m_textCtrlThreadsNb->SetValue(pConfig->Read("/Processing/ThreadsNb", maxThreadsStr));
    m_sliderThreadsPriority->SetValue((int)pConfig->ReadLong("/Processing/ThreadsPriority", 95l));

    // Processing
    long processingMethod = pConfig->ReadLong("/Processing/Method", (long)asMULTITHREADS);
    if (!allowMultithreading) {
        m_radioBoxProcessingMethods->Enable(0, false);
        if (processingMethod == (long)asMULTITHREADS) {
            processingMethod = (long)asSTANDARD;
        }
    } else {
        m_radioBoxProcessingMethods->Enable(0, true);
    }
    m_radioBoxProcessingMethods->SetSelection((int)processingMethod);

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_staticTextUserDir->SetLabel(userpath);
    wxString logpath = asConfig::GetLogDir();
    logpath.Append("AtmoSwingDownscaler.log");
    m_staticTextLogFile->SetLabel(logpath);
    m_staticTextPrefFile->SetLabel(asConfig::GetConfigFilePath("AtmoSwingDownscaler.ini"));
}

void asFramePreferencesDownscaler::SavePreferences() const {
    wxBusyCursor wait;

    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

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
    bool displayLogWindow = m_checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindow);

    // Paths
    wxString predictandDBDir = m_dirPickerPredictandDB->GetPath();
    pConfig->Write("/Paths/DataPredictandDBDir", predictandDBDir);
    wxString archivePredictorsDir = m_dirPickerArchivePredictors->GetPath();
    pConfig->Write("/Paths/ArchivePredictorsDir", archivePredictorsDir);
    wxString scenarioPredictorsDir = m_dirPickerScenarioPredictors->GetPath();
    pConfig->Write("/Paths/ScenarioPredictorsDir", scenarioPredictorsDir);

    /*
     * Advanced
     */

    // GUI options
    auto guiOptions = (long)m_radioBoxGui->GetSelection();
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

    // Advanced options
    bool responsive = m_checkBoxResponsiveness->GetValue();
    pConfig->Write("/General/Responsive", responsive);
    g_responsive = responsive;

    // Multithreading
    bool allowMultithreading = m_checkBoxAllowMultithreading->GetValue();
    pConfig->Write("/Processing/AllowMultithreading", allowMultithreading);
    wxString processingMaxThreadNb = m_textCtrlThreadsNb->GetValue();
    if (!processingMaxThreadNb.IsNumber()) processingMaxThreadNb = "2";
    pConfig->Write("/Processing/ThreadsNb", processingMaxThreadNb);
    auto processingThreadsPriority = (long)m_sliderThreadsPriority->GetValue();
    pConfig->Write("/Processing/ThreadsPriority", processingThreadsPriority);

    // Processing
    auto processingMethod = (long)m_radioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && processingMethod == (long)asMULTITHREADS) {
        processingMethod = (long)asSTANDARD;
    }
    pConfig->Write("/Processing/Method", processingMethod);

    GetParent()->Update();
    pConfig->Flush();
}

void asFramePreferencesDownscaler::OnChangeMultithreadingCheckBox(wxCommandEvent& event) {
    if (event.GetInt() == 0) {
        m_radioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (m_radioBoxProcessingMethods->GetSelection() == asMULTITHREADS) {
            m_radioBoxProcessingMethods->SetSelection(asSTANDARD);
        }
    } else {
        m_radioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferencesDownscaler::SaveAndClose(wxCommandEvent& event) {
    SavePreferences();
    Close();
}

void asFramePreferencesDownscaler::ApplyChanges(wxCommandEvent& event) {
    SavePreferences();
}
