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

#include "asFrameDownscaler.h"

#include "asIncludes.h"

#include "asBitmaps.h"
#include "asFrameAbout.h"
#include "asFramePredictandDB.h"
#include "asFramePreferencesDownscaler.h"
#include "asMethodDownscalerClassic.h"
#include "wx/fileconf.h"

asFrameDownscaler::asFrameDownscaler(wxWindow* parent)
    : asFrameDownscalerVirtual(parent),
      _logWindow(nullptr),
      _methodDownscaler(nullptr) {
    // Toolbar
    _toolBar->AddTool(asID_RUN, wxT("Run"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::RUN), wxNullBitmap, wxITEM_NORMAL,
                      _("Run downscaler"), _("Run downscaler now"), nullptr);
    _toolBar->AddTool(asID_CANCEL, wxT("Cancel"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::STOP), wxNullBitmap,
                      wxITEM_NORMAL, _("Cancel downscaling"), _("Cancel current downscaling"), nullptr);
    _toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::PREFERENCES),
                      wxNullBitmap, wxITEM_NORMAL, _("Preferences"), _("Preferences"), nullptr);
    _toolBar->Realize();

    // Connect events
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::Launch, this, asID_RUN);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::Cancel, this, asID_CANCEL);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::OpenFramePreferences, this, asID_PREFERENCES);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::OpenFramePredictandDB, this, asID_DB_CREATE);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameDownscaler::~asFrameDownscaler() {
    // Disconnect events
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::Launch, this, asID_RUN);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::Cancel, this, asID_CANCEL);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::OpenFramePreferences, this, asID_PREFERENCES);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameDownscaler::OpenFramePredictandDB, this, asID_DB_CREATE);
}

void asFrameDownscaler::OnInit() {
    wxBusyCursor wait;

    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameDownscaler::Update() {
    DisplayLogLevelMenu();
}

void asFrameDownscaler::OpenFramePredictandDB(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesDownscaler(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OpenFrameAbout(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OnShowLog(wxCommandEvent& event) {
    wxBusyCursor wait;

    wxASSERT(_logWindow);
    _logWindow->DoShow(true);
}

void asFrameDownscaler::OnLogLevel1(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(1);
    _menuLogLevel->FindItemByPosition(0)->Check(true);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameDownscaler::OnLogLevel2(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(2);
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(true);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameDownscaler::OnLogLevel3(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(3);
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(true);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameDownscaler::DisplayLogLevelMenu() {
    // Set log level in the menu
    ThreadsManager().CritSectionConfig().Enter();
    int logLevel = int(wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l));
    ThreadsManager().CritSectionConfig().Leave();
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel) {
        case 1:
            _menuLogLevel->FindItemByPosition(0)->Check(true);
            Log()->SetLevel(1);
            break;
        case 2:
            _menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
            break;
        case 3:
            _menuLogLevel->FindItemByPosition(2)->Check(true);
            Log()->SetLevel(3);
            break;
        default:
            _menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
    }
}

void asFrameDownscaler::Cancel(wxCommandEvent& event) {
    if (_methodDownscaler) {
        _methodDownscaler->Cancel();
    }
}

void asFrameDownscaler::LoadOptions() {
    wxConfigBase* pConfig = wxFileConfig::Get();
    _choiceMethod->SetSelection(pConfig->ReadLong("/MethodSelection", 0l));
    _filePickerParameters->SetPath(pConfig->Read("/ParametersFilePath", wxEmptyString));
    _filePickerPredictand->SetPath(pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString));
    _dirPickerArchivePredictor->SetPath(pConfig->Read("/Paths/ArchivePredictorsDir", wxEmptyString));
    _dirPickerScenarioPredictor->SetPath(pConfig->Read("/Paths/ScenarioPredictorsDir", wxEmptyString));
    _dirPickerDownscalingResults->SetPath(
        pConfig->Read("/Paths/DownscalerResultsDir", asConfig::GetDocumentsDir() + "AtmoSwing" + DS + "Downscaler"));
    _checkBoxParallelEvaluations->SetValue(pConfig->ReadBool("/ParallelEvaluations", false));
}

void asFrameDownscaler::OnSaveDefault(wxCommandEvent& event) {
    SaveOptions();
}

void asFrameDownscaler::SaveOptions() const {
    wxBusyCursor wait;

    wxConfigBase* pConfig = wxFileConfig::Get();
    auto methodSelection = (long)_choiceMethod->GetSelection();
    pConfig->Write("/MethodSelection", methodSelection);
    wxString parametersFilePath = _filePickerParameters->GetPath();
    pConfig->Write("/ParametersFilePath", parametersFilePath);
    wxString predictandDBFilePath = _filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", predictandDBFilePath);
    wxString archivePredictorDir = _dirPickerArchivePredictor->GetPath();
    pConfig->Write("/Paths/ArchivePredictorDir", archivePredictorDir);
    wxString scenarioPredictorDir = _dirPickerScenarioPredictor->GetPath();
    pConfig->Write("/Paths/ScenarioPredictorDir", scenarioPredictorDir);
    wxString downscalerResultsDir = _dirPickerDownscalingResults->GetPath();
    pConfig->Write("/Paths/DownscalerResultsDir", downscalerResultsDir);
    bool parallelEvaluations = _checkBoxParallelEvaluations->GetValue();
    pConfig->Write("/ParallelEvaluations", parallelEvaluations);

    pConfig->Flush();
}

/*
void asFrameDownscaler::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    _staticTextState->SetLabel(state);
}
*/
void asFrameDownscaler::Launch(wxCommandEvent& event) {
    wxBusyCursor wait;

    SaveOptions();

    try {
        switch (_choiceMethod->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong method selection."));
                break;
            }
            case 0:  // Classic
            {
                wxLogVerbose(_("Proceeding to classic downscaling."));
                _methodDownscaler = new asMethodDownscalerClassic();
                break;
            }
            default:
                wxLogError(_("Chosen method not defined yet."));
        }

        if (_methodDownscaler) {
            _methodDownscaler->SetParamsFilePath(_filePickerParameters->GetPath());
            _methodDownscaler->SetPredictandDBFilePath(_filePickerPredictand->GetPath());
            _methodDownscaler->SetPredictorDataDir(_dirPickerArchivePredictor->GetPath());
            _methodDownscaler->SetPredictorProjectionDataDir(_dirPickerScenarioPredictor->GetPath());
            _methodDownscaler->Manager();
        }
    } catch (std::bad_alloc& ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught: %s"), msg);
        wxLogError(_("Failed to process the downscaling."));
    } catch (runtime_error& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        wxLogError(_("Failed to process the downscaling."));
    }

    wxDELETE(_methodDownscaler);

    wxMessageBox(_("Downscaler over."));
}
