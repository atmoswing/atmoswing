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

#include "asIncludes.h"

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
    wxColour col = _notebookBase->GetThemeBackgroundColour();
    if (col.IsOk()) {
        _dirPickerPredictandDB->SetBackgroundColour(col);
        _dirPickerIntermediateResults->SetBackgroundColour(col);
        _dirPickerArchivePredictors->SetBackgroundColour(col);
        _dirPickerScenarioPredictors->SetBackgroundColour(col);
    }

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
    long logLevel = pConfig->ReadLong("/General/LogLevel", 1L);
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

    // Paths
    wxString dirData = asConfig::GetDataDir() + "data" + DS;
    _dirPickerPredictandDB->SetPath(pConfig->Read("/Paths/DataPredictandDBDir", dirData + "predictands"));
    _dirPickerArchivePredictors->SetPath(pConfig->Read("/Paths/ArchivePredictorsDir", dirData + "predictors"));
    _dirPickerScenarioPredictors->SetPath(pConfig->Read("/Paths/ScenarioPredictorsDir", dirData + "predictors"));

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = pConfig->ReadLong("/General/GuiOptions", 1l);
    _radioBoxGui->SetSelection(int(guiOptions));
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
    _checkBoxResponsiveness->SetValue(responsive);
    g_responsive = responsive;

    // Multithreading
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    _checkBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads == -1) maxThreads = 2;
    wxString maxThreadsStr = asStrF("%d", maxThreads);
    _textCtrlThreadsNb->SetValue(pConfig->Read("/Processing/ThreadsNb", maxThreadsStr));
    _sliderThreadsPriority->SetValue((int)pConfig->ReadLong("/Processing/ThreadsPriority", 95l));

    // Processing
    long processingMethod = pConfig->ReadLong("/Processing/Method", (long)asMULTITHREADS);
    if (!allowMultithreading) {
        _radioBoxProcessingMethods->Enable(0, false);
        if (processingMethod == (long)asMULTITHREADS) {
            processingMethod = (long)asSTANDARD;
        }
    } else {
        _radioBoxProcessingMethods->Enable(0, true);
    }
    _radioBoxProcessingMethods->SetSelection((int)processingMethod);

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    _staticTextUserDir->SetLabel(userpath);
    wxString logpath = asConfig::GetLogDir();
    logpath.Append("AtmoSwingDownscaler.log");
    _staticTextLogFile->SetLabel(logpath);
    _staticTextPrefFile->SetLabel(asConfig::GetConfigFilePath("AtmoSwingDownscaler.ini"));
}

void asFramePreferencesDownscaler::SavePreferences() const {
    wxBusyCursor wait;

    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

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
    bool displayLogWindow = _checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindow);

    // Paths
    wxString predictandDBDir = _dirPickerPredictandDB->GetPath();
    pConfig->Write("/Paths/DataPredictandDBDir", predictandDBDir);
    wxString archivePredictorsDir = _dirPickerArchivePredictors->GetPath();
    pConfig->Write("/Paths/ArchivePredictorsDir", archivePredictorsDir);
    wxString scenarioPredictorsDir = _dirPickerScenarioPredictors->GetPath();
    pConfig->Write("/Paths/ScenarioPredictorsDir", scenarioPredictorsDir);

    /*
     * Advanced
     */

    // GUI options
    auto guiOptions = (long)_radioBoxGui->GetSelection();
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
    bool responsive = _checkBoxResponsiveness->GetValue();
    pConfig->Write("/General/Responsive", responsive);
    g_responsive = responsive;

    // Multithreading
    bool allowMultithreading = _checkBoxAllowMultithreading->GetValue();
    pConfig->Write("/Processing/AllowMultithreading", allowMultithreading);
    wxString processingMaxThreadNb = _textCtrlThreadsNb->GetValue();
    if (!processingMaxThreadNb.IsNumber()) processingMaxThreadNb = "2";
    pConfig->Write("/Processing/ThreadsNb", processingMaxThreadNb);
    auto processingThreadsPriority = (long)_sliderThreadsPriority->GetValue();
    pConfig->Write("/Processing/ThreadsPriority", processingThreadsPriority);

    // Processing
    auto processingMethod = (long)_radioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && processingMethod == (long)asMULTITHREADS) {
        processingMethod = (long)asSTANDARD;
    }
    pConfig->Write("/Processing/Method", processingMethod);

    GetParent()->Update();
    pConfig->Flush();
}

void asFramePreferencesDownscaler::OnChangeMultithreadingCheckBox(wxCommandEvent& event) {
    if (event.GetInt() == 0) {
        _radioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (_radioBoxProcessingMethods->GetSelection() == asMULTITHREADS) {
            _radioBoxProcessingMethods->SetSelection(asSTANDARD);
        }
    } else {
        _radioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferencesDownscaler::SaveAndClose(wxCommandEvent& event) {
    SavePreferences();
    Close();
}

void asFramePreferencesDownscaler::ApplyChanges(wxCommandEvent& event) {
    SavePreferences();
}
