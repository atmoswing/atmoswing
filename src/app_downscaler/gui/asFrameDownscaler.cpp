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

#include "wx/fileconf.h"

#include "asMethodDownscalerClassic.h"
#include "images.h"
#include "asFramePreferencesDownscaler.h"
#include "asFrameAbout.h"
#include "asFramePredictandDB.h"


asFrameDownscaler::asFrameDownscaler(wxWindow *parent)
        : asFrameDownscalerVirtual(parent),
          m_logWindow(nullptr),
          m_methodDownscaler(nullptr)
{
    // Toolbar
    m_toolBar->AddTool(asID_RUN, wxT("Run"), *_img_run, *_img_run, wxITEM_NORMAL, _("Run downscaler"),
                       _("Run downscaler now"), nullptr);
    m_toolBar->AddTool(asID_CANCEL, wxT("Cancel"), *_img_stop, *_img_stop, wxITEM_NORMAL, _("Cancel downscaling"),
                       _("Cancel current downscaling"), nullptr);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), *_img_preferences, *_img_preferences, wxITEM_NORMAL,
                       _("Preferences"), _("Preferences"), nullptr);
    m_toolBar->Realize();

    // Connect events
    this->Connect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Launch));
    this->Connect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Cancel));
    this->Connect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFrameDownscaler::OpenFramePreferences));
    this->Connect(asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED,
                  wxCommandEventHandler(asFrameDownscaler::OpenFramePredictandDB));

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameDownscaler::~asFrameDownscaler()
{
    // Disconnect events
    this->Disconnect(asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Launch));
    this->Disconnect(asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler(asFrameDownscaler::Cancel));
    this->Disconnect(asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED,
                     wxCommandEventHandler(asFrameDownscaler::OpenFramePreferences));
    this->Disconnect(asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED,
                     wxCommandEventHandler(asFrameDownscaler::OpenFramePredictandDB));
}

void asFrameDownscaler::OnInit()
{
    wxBusyCursor wait;

    // Set the defaults
    LoadOptions();
    DisplayLogLevelMenu();
}

void asFrameDownscaler::Update()
{
    DisplayLogLevelMenu();
}

void asFrameDownscaler::OpenFramePredictandDB(wxCommandEvent &event)
{
    wxBusyCursor wait;

    auto *frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OpenFramePreferences(wxCommandEvent &event)
{
    wxBusyCursor wait;

    auto *frame = new asFramePreferencesDownscaler(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OpenFrameAbout(wxCommandEvent &event)
{
    wxBusyCursor wait;

    auto *frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameDownscaler::OnShowLog(wxCommandEvent &event)
{
    wxBusyCursor wait;

    wxASSERT(m_logWindow);
    m_logWindow->DoShow(true);
}

void asFrameDownscaler::OnLogLevel1(wxCommandEvent &event)
{
    wxBusyCursor wait;

    Log().SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame)
        prefFrame->Update();
}

void asFrameDownscaler::OnLogLevel2(wxCommandEvent &event)
{
    wxBusyCursor wait;

    Log().SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame)
        prefFrame->Update();
}

void asFrameDownscaler::OnLogLevel3(wxCommandEvent &event)
{
    wxBusyCursor wait;

    Log().SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    ThreadsManager().CritSectionConfig().Enter();
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    ThreadsManager().CritSectionConfig().Leave();
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame)
        prefFrame->Update();
}

void asFrameDownscaler::DisplayLogLevelMenu()
{
    // Set log level in the menu
    ThreadsManager().CritSectionConfig().Enter();
    int logLevel = static_cast<int>(wxFileConfig::Get()->Read("/General/LogLevel", 2l));
    ThreadsManager().CritSectionConfig().Leave();
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel) {
        case 1:
            m_menuLogLevel->FindItemByPosition(0)->Check(true);
            Log().SetLevel(1);
            break;
        case 2:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log().SetLevel(2);
            break;
        case 3:
            m_menuLogLevel->FindItemByPosition(2)->Check(true);
            Log().SetLevel(3);
            break;
        default:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log().SetLevel(2);
    }
}

void asFrameDownscaler::Cancel(wxCommandEvent &event)
{
    if (m_methodDownscaler) {
        m_methodDownscaler->Cancel();
    }
}

void asFrameDownscaler::LoadOptions()
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    long methodSelection = pConfig->Read("/Downscaler/MethodSelection", 0l);
    m_choiceMethod->SetSelection(static_cast<int>(methodSelection));
    wxString parametersFilePath = pConfig->Read("/Downscaler/ParametersFilePath", wxEmptyString);
    m_filePickerParameters->SetPath(parametersFilePath);
    wxString predictandDBFilePath = pConfig->Read("/Paths/PredictandDBFilePath", wxEmptyString);
    m_filePickerPredictand->SetPath(predictandDBFilePath);
    wxString predictorArchiveDir = pConfig->Read("/Paths/ArchivePredictorsDir", wxEmptyString);
    m_dirPickerArchivePredictor->SetPath(predictorArchiveDir);
    wxString predictorScenarioDir = pConfig->Read("/Paths/ScenarioPredictorsDir", wxEmptyString);
    m_dirPickerScenarioPredictor->SetPath(predictorScenarioDir);
    wxString downscalerResultsDir = pConfig->Read("/Paths/DownscalerResultsDir",
                                                 asConfig::GetDocumentsDir() + "AtmoSwing/Downscaler");
    m_dirPickerDownscalingResults->SetPath(downscalerResultsDir);
    bool parallelEvaluations;
    pConfig->Read("/Downscaler/ParallelEvaluations", &parallelEvaluations, false);
    m_checkBoxParallelEvaluations->SetValue(parallelEvaluations);
}

void asFrameDownscaler::OnSaveDefault(wxCommandEvent &event)
{
    SaveOptions();
}

void asFrameDownscaler::SaveOptions() const
{
    wxBusyCursor wait;

    wxConfigBase *pConfig = wxFileConfig::Get();
    auto methodSelection = (long) m_choiceMethod->GetSelection();
    pConfig->Write("/Downscaler/MethodSelection", methodSelection);
    wxString parametersFilePath = m_filePickerParameters->GetPath();
    pConfig->Write("/Downscaler/ParametersFilePath", parametersFilePath);
    wxString predictandDBFilePath = m_filePickerPredictand->GetPath();
    pConfig->Write("/Paths/PredictandDBFilePath", predictandDBFilePath);
    wxString archivePredictorDir = m_dirPickerArchivePredictor->GetPath();
    pConfig->Write("/Paths/ArchivePredictorDir", archivePredictorDir);
    wxString scenarioPredictorDir = m_dirPickerScenarioPredictor->GetPath();
    pConfig->Write("/Paths/ScenarioPredictorDir", scenarioPredictorDir);
    wxString downscalerResultsDir = m_dirPickerDownscalingResults->GetPath();
    pConfig->Write("/Paths/DownscalerResultsDir", downscalerResultsDir);
    bool parallelEvaluations = m_checkBoxParallelEvaluations->GetValue();
    pConfig->Write("/Downscaler/ParallelEvaluations", parallelEvaluations);

    pConfig->Flush();
}

/*
void asFrameDownscaler::OnIdle( wxCommandEvent& event )
{
    wxString state = asGetState();
    m_staticTextState->SetLabel(state);
}
*/
void asFrameDownscaler::Launch(wxCommandEvent &event)
{
    wxBusyCursor wait;

    SaveOptions();

    try {
        switch (m_choiceMethod->GetSelection()) {
            case wxNOT_FOUND: {
                wxLogError(_("Wrong method selection."));
                break;
            }
            case 0: // Classic
            {
                wxLogVerbose(_("Proceeding to classic downscaling."));
                m_methodDownscaler = new asMethodDownscalerClassic();
                break;
            }
            default:
                wxLogError(_("Chosen method not defined yet."));
        }

        if (m_methodDownscaler) {
            m_methodDownscaler->SetParamsFilePath(m_filePickerParameters->GetPath());
            m_methodDownscaler->SetPredictandDBFilePath(m_filePickerPredictand->GetPath());
            m_methodDownscaler->SetPredictorDataDir(m_dirPickerArchivePredictor->GetPath());
            m_methodDownscaler->SetPredictorProjectionDataDir(m_dirPickerScenarioPredictor->GetPath());
            m_methodDownscaler->Manager();
        }
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught: %s"), msg);
        wxLogError(_("Failed to process the downscaling."));
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to process the downscaling."));
    }

    wxDELETE(m_methodDownscaler);

    wxMessageBox(_("Downscaler over."));
}
