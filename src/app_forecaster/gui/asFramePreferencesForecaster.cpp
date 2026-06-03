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

#include "asIncludes.h"

#include "asFileGrib.h"

asFramePreferencesForecaster::asFramePreferencesForecaster(wxWindow* parent, asBatchForecasts* batchForecasts,
                                                           wxWindowID id)
    : asFramePreferencesForecasterVirtual(parent, id),
      _batchForecasts(batchForecasts) {
    SetLabel(_("Preferences"));

    LoadPreferences();
    Fit();

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesForecaster::CloseFrame(wxCommandEvent& event) {
    Close();
}

void asFramePreferencesForecaster::Update() {
    LoadPreferences();
}

void asFramePreferencesForecaster::LoadPreferences() {
    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = _notebookBase->GetThemeBackgroundColour();
    if (col.IsOk()) {
        _dirPickerPredictandDB->SetBackgroundColour(col);
        _dirPickerForecastResults->SetBackgroundColour(col);
        _dirPickerForecastResultsExports->SetBackgroundColour(col);
        _dirPickerParameters->SetBackgroundColour(col);
        _dirPickerArchivePredictors->SetBackgroundColour(col);
        _dirPickerRealtimePredictorSaving->SetBackgroundColour(col);
    }

    /*
     * Batch file properties
     */

    // Paths
    _dirPickerPredictandDB->SetPath(_batchForecasts->GetPredictandDBDirectory());
    _dirPickerForecastResults->SetPath(_batchForecasts->GetForecastsOutputDirectory());
    _dirPickerForecastResultsExports->SetPath(_batchForecasts->GetExportsOutputDirectory());
    _dirPickerRealtimePredictorSaving->SetPath(_batchForecasts->GetPredictorsRealtimeDirectory());
    _dirPickerArchivePredictors->SetPath(_batchForecasts->GetPredictorsArchiveDirectory());
    _dirPickerParameters->SetPath(_batchForecasts->GetParametersFileDirectory());

    // Exports
    switch (_batchForecasts->GetExport()) {
        case asBatchForecasts::None:
            _choiceExports->SetSelection(0);
            break;
        case asBatchForecasts::FullXml:
            _choiceExports->SetSelection(1);
            break;
        case asBatchForecasts::SmallCsv:
            _choiceExports->SetSelection(2);
            break;
        case asBatchForecasts::CustomCsvFVG:
            _choiceExports->SetSelection(3);
            break;
        default:
            _choiceExports->SetSelection(0);
            break;
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
    long logLevelForecaster = pConfig->ReadLong("/General/LogLevel", 1);
    if (logLevelForecaster == 1) {
        _radioBtnLogLevel1->SetValue(true);
    } else if (logLevelForecaster == 2) {
        _radioBtnLogLevel2->SetValue(true);
    } else if (logLevelForecaster == 3) {
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

    // Libraries
    _textCtrlEcCodesDefs->SetValue(pConfig->Read("/Libraries/EcCodesDefinitions", asFileGrib::GetDefinitionsPath()));

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = pConfig->ReadLong("/General/GuiOptions", 1l);
    _radioBoxGui->SetSelection((int)guiOptions);
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
    _textCtrlMaxPrevStepsNb->SetValue(pConfig->Read("/Internet/MaxPreviousStepsNb", "5"));
    _checkBoxRestrictDownloads->SetValue(pConfig->ReadBool("/Internet/RestrictDownloads", true));

    // Advanced options
    g_responsive = pConfig->ReadBool("/General/Responsive", true);
    _checkBoxResponsiveness->SetValue(g_responsive);
    _checkBoxMultiInstancesForecaster->SetValue(pConfig->ReadBool("/General/MultiInstances", false));

    // Multithreading
    bool allowMultithreading = pConfig->ReadBool("/Processing/AllowMultithreading", true);
    _checkBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads == -1) maxThreads = 2;
    _textCtrlThreadsNb->SetValue(pConfig->Read("/Processing/ThreadsNb", asStrF("%d", maxThreads)));
    _sliderThreadsPriority->SetValue(pConfig->ReadLong("/Processing/ThreadsPriority", 95l));

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
    wxString logpathForecaster = asConfig::GetLogDir();
    logpathForecaster.Append("AtmoSwingForecaster.log");
    _staticTextLogFile->SetLabel(logpathForecaster);
    _staticTextPrefFile->SetLabel(asConfig::GetConfigFilePath("AtmoSwingForecaster.ini"));
}

void asFramePreferencesForecaster::SavePreferences() {
    wxBusyCursor wait;

    wxConfigBase* pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * Batch file properties
     */

    // Paths
    _batchForecasts->SetPredictandDBDirectory(_dirPickerPredictandDB->GetPath());
    _batchForecasts->SetForecastsOutputDirectory(_dirPickerForecastResults->GetPath());
    _batchForecasts->SetExportsOutputDirectory(_dirPickerForecastResultsExports->GetPath());
    _batchForecasts->SetPredictorsRealtimeDirectory(_dirPickerRealtimePredictorSaving->GetPath());
    _batchForecasts->SetPredictorsArchiveDirectory(_dirPickerArchivePredictors->GetPath());
    _batchForecasts->SetParametersFileDirectory(_dirPickerParameters->GetPath());

    // Exports
    switch (_choiceExports->GetSelection()) {
        case 0:
            _batchForecasts->SetExport(asBatchForecasts::None);
            break;
        case 1:
            _batchForecasts->SetExport(asBatchForecasts::FullXml);
            break;
        case 2:
            _batchForecasts->SetExport(asBatchForecasts::SmallCsv);
            break;
        case 3:
            _batchForecasts->SetExport(asBatchForecasts::CustomCsvFVG);
            break;
        default:
            _batchForecasts->SetExport(asBatchForecasts::None);
    }

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
    long logLevelForecaster = 1;
    if (_radioBtnLogLevel1->GetValue()) {
        logLevelForecaster = 1;
    } else if (_radioBtnLogLevel2->GetValue()) {
        logLevelForecaster = 2;
    } else if (_radioBtnLogLevel3->GetValue()) {
        logLevelForecaster = 3;
    }
    pConfig->Write("/General/LogLevel", logLevelForecaster);
    bool displayLogWindowForecaster = _checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindowForecaster);

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

    // Libraries
    wxString ecCodesDefs = _textCtrlEcCodesDefs->GetValue();
    pConfig->Write("/Libraries/EcCodesDefinitions", ecCodesDefs);

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

    // Downloads
    wxString internetMaxPrevStepsNb = _textCtrlMaxPrevStepsNb->GetValue();
    if (!internetMaxPrevStepsNb.IsNumber()) internetMaxPrevStepsNb = "5";
    pConfig->Write("/Internet/MaxPreviousStepsNb", internetMaxPrevStepsNb);
    bool restrictDownloads = _checkBoxRestrictDownloads->GetValue();
    pConfig->Write("/Internet/RestrictDownloads", restrictDownloads);

    // Advanced options
    bool responsive = _checkBoxResponsiveness->GetValue();
    pConfig->Write("/General/Responsive", responsive);
    g_responsive = responsive;

    bool multiForecaster = _checkBoxMultiInstancesForecaster->GetValue();
    pConfig->Write("/General/MultiInstances", multiForecaster);

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

    if (GetParent() != nullptr) {
        GetParent()->Update();
    }

    pConfig->Flush();
    _batchForecasts->Save();
}

void asFramePreferencesForecaster::OnChangeMultithreadingCheckBox(wxCommandEvent& event) {
    if (event.GetInt() == 0) {
        _radioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (_radioBoxProcessingMethods->GetSelection() == asMULTITHREADS) {
            _radioBoxProcessingMethods->SetSelection(asSTANDARD);
        }
    } else {
        _radioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferencesForecaster::SaveAndClose(wxCommandEvent& event) {
    SavePreferences();
    Close();
}

void asFramePreferencesForecaster::ApplyChanges(wxCommandEvent& event) {
    SavePreferences();
}
