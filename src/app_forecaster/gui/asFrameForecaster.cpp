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

#include "asFrameForecaster.h"

#include "asBitmaps.h"
#include "asFrameAbout.h"
#include "asFramePredictandDB.h"
#include "asFramePreferencesForecaster.h"
#include "asPanelForecast.h"
#include "asWizardBatchForecasts.h"

BEGIN_EVENT_TABLE(asFrameForecaster, wxFrame)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_STARTING, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_RUNNING, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_FAILED, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_SUCCESS, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_DOWNLOADING, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_DOWNLOADED, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_LOADING, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_LOADED, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_SAVING, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_SAVED, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_PROCESSING, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_STATUS_PROCESSED, asFrameForecaster::OnStatusMethodUpdate)
EVT_COMMAND(wxID_ANY, asEVT_ACTION_OPEN_BATCHFORECASTS, asFrameForecaster::OnOpenBatchForecasts)
END_EVENT_TABLE()

asFrameForecaster::asFrameForecaster(wxWindow* parent)
    : asFrameForecasterVirtual(parent) {
    m_forecaster = nullptr;
    m_logWindow = nullptr;
    m_fileHistory = new wxFileHistory(9);

    // Fix colors
    // m_panelMain->SetBackgroundColour(asConfig::GetFrameBgColour());

    // Menu recent
    auto menuOpenRecent = new wxMenu();
    m_menuFile->Insert(1, asID_MENU_RECENT, _("Open recent"), menuOpenRecent);

    // Toolbar
    m_toolBar->AddTool(asID_RUN, wxT("Run"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::RUN), wxNullBitmap, wxITEM_NORMAL,
                       _("Run forecast"), _("Run forecast now"), nullptr);
    m_toolBar->AddTool(asID_CANCEL, wxT("Cancel"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::STOP), wxNullBitmap,
                       wxITEM_NORMAL, _("Cancel forecast"), _("Cancel current forecast"), nullptr);
    m_toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::PREFERENCES),
                       wxNullBitmap, wxITEM_NORMAL, _("Preferences"), _("Preferences"), nullptr);
    m_toolBar->Realize();

    // Leds
    m_ledDownloading = new awxLed(m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    m_ledDownloading->SetState(awxLED_OFF);
    m_sizerLeds->Add(m_ledDownloading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textDownloading = new wxStaticText(m_panelMain, wxID_ANY, _("Downloading predictors"));
    textDownloading->Wrap(-1);
    m_sizerLeds->Add(textDownloading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    m_ledLoading = new awxLed(m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    m_ledLoading->SetState(awxLED_OFF);
    m_sizerLeds->Add(m_ledLoading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textLoading = new wxStaticText(m_panelMain, wxID_ANY, _("Loading data"));
    textLoading->Wrap(-1);
    m_sizerLeds->Add(textLoading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    m_ledProcessing = new awxLed(m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    m_ledProcessing->SetState(awxLED_OFF);
    m_sizerLeds->Add(m_ledProcessing, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textProcessing = new wxStaticText(m_panelMain, wxID_ANY, _("Processing"));
    textProcessing->Wrap(-1);
    m_sizerLeds->Add(textProcessing, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    m_ledSaving = new awxLed(m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    m_ledSaving->SetState(awxLED_OFF);
    m_sizerLeds->Add(m_ledSaving, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textSaving = new wxStaticText(m_panelMain, wxID_ANY, _("Saving results"));
    textSaving->Wrap(-1);
    m_sizerLeds->Add(textSaving, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    // Buttons
    m_bpButtonNow->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::UPDATE));
    m_bpButtonAdd->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::PLUS));

    // Create panels manager
    m_panelsManager = new asPanelsManagerForecasts();

    // Connect events
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::LaunchForecasting, this, asID_RUN);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::CancelForecasting, this, asID_CANCEL);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::OpenFramePredictandDB, this, asID_DB_CREATE);
    Bind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::OpenFramePreferences, this, asID_PREFERENCES);
    Bind(wxEVT_COMMAND_MENU_SELECTED, &asFrameForecaster::OnFileHistory, this, wxID_FILE1, wxID_FILE9);

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif

    SetRecentFiles();
}

asFrameForecaster::~asFrameForecaster() {
    wxDELETE(m_panelsManager);

    SaveRecentFiles();

    // Disconnect events
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::LaunchForecasting, this, asID_RUN);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::CancelForecasting, this, asID_CANCEL);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::OpenFramePredictandDB, this, asID_DB_CREATE);
    Unbind(wxEVT_COMMAND_TOOL_CLICKED, &asFrameForecaster::OpenFramePreferences, this, asID_PREFERENCES);
    Unbind(wxEVT_COMMAND_MENU_SELECTED, &asFrameForecaster::OnFileHistory, this, wxID_FILE1, wxID_FILE9);
}

void asFrameForecaster::OnInit() {
    wxBusyCursor wait;

    DisplayLogLevelMenu();
    SetPresentDate();

    // Open last batch file
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString batchFilePath = pConfig->Read("/BatchForecasts/LastOpened", wxEmptyString);

    // Check provided files
    if (!g_cmdFileName.IsEmpty()) {
        long strSize = g_cmdFileName.size();
        long strExt = g_cmdFileName.size() - 4;
        wxString ext = g_cmdFileName.SubString(strExt - 1, strSize - 1);
        if (ext.IsSameAs(".asfb", false)) {
            batchFilePath = g_cmdFileName;
        }
    }

    if (!batchFilePath.IsEmpty()) {
        if (!m_batchForecasts.Load(batchFilePath)) {
            wxLogWarning(_("Failed to open the batch file ") + batchFilePath);
        }

        OpenBatchForecasts();
    } else {
        asWizardBatchForecasts wizard(this, &m_batchForecasts);
        wizard.RunWizard(wizard.GetFirstPage());

        OpenBatchForecasts();
    }
}

void asFrameForecaster::OnOpenBatchForecasts(wxCommandEvent& event) {
    // Ask for a batch file
    wxFileDialog openFileDialog(this, _("Select a batch file"), wxEmptyString, wxEmptyString,
                                "AtmoSwing forecaster batch (*.xml)|*.xml",
                                wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR);

    // If canceled
    if (openFileDialog.ShowModal() == wxID_CANCEL) return;

    wxBusyCursor wait;

    wxString batchFilePath = openFileDialog.GetPath();

    // Save last opened
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);

    // Do open the batch file
    if (!m_batchForecasts.Load(batchFilePath)) {
        wxLogError(_("Failed to open the batch file ") + batchFilePath);
    }

    OpenBatchForecasts();

    m_fileHistory->AddFileToHistory(batchFilePath);
}

void asFrameForecaster::OnFileHistory(wxCommandEvent& event) {
    int id = event.GetId() - wxID_FILE1;
    wxString batchFilePath = m_fileHistory->GetHistoryFile(id);

    wxBusyCursor wait;

    // Save last opened
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);

    // Do open the batch file
    if (!m_batchForecasts.Load(batchFilePath)) {
        wxLogError(_("Failed to open the batch file ") + batchFilePath);
    }

    OpenBatchForecasts();
}

void asFrameForecaster::OnSaveBatchForecasts(wxCommandEvent& event) {
    SaveBatchForecasts();
}

void asFrameForecaster::OnSaveBatchForecastsAs(wxCommandEvent& event) {
    // Ask for a batch file
    wxFileDialog openFileDialog(this, _("Select a path to save the batch file"), wxEmptyString, wxEmptyString,
                                "AtmoSwing forecaster batch (*.xml)|*.xml", wxFD_SAVE | wxFD_CHANGE_DIR);

    // If canceled
    if (openFileDialog.ShowModal() == wxID_CANCEL) return;

    wxBusyCursor wait;

    wxString batchFilePath = openFileDialog.GetPath();
    m_batchForecasts.SetFilePath(batchFilePath);

    if (SaveBatchForecasts()) {
        // Save preferences
        wxConfigBase* pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);
    }
}

bool asFrameForecaster::SaveBatchForecasts() {
    wxBusyCursor wait;

    UpdateBatchForecasts();

    if (!m_batchForecasts.Save()) {
        wxLogError(_("Could not save the batch file."));
        return false;
    }

    m_batchForecasts.SetHasChanged(false);

    return true;
}

bool asFrameForecaster::UpdateBatchForecasts() {
    m_batchForecasts.ClearForecasts();

    for (int i = 0; i < m_panelsManager->GetPanelsNb(); i++) {
        asPanelForecast* panel = m_panelsManager->GetPanel(i);

        m_batchForecasts.AddForecast();

        m_batchForecasts.SetForecastFileName(i, panel->GetParametersFileName());
    }

    return true;
}

void asFrameForecaster::OnNewBatchForecasts(wxCommandEvent& event) {
    asWizardBatchForecasts wizard(this, &m_batchForecasts);
    wizard.RunWizard(wizard.GetSecondPage());
}

bool asFrameForecaster::OpenBatchForecasts() {
    wxBusyCursor wait;

    Freeze();

    wxFileName batchFileName = wxFileName(m_batchForecasts.GetFilePath());
    m_staticTextbatchFile->SetLabel(batchFileName.GetFullName());

    // Cleanup the actual panels
    m_panelsManager->Clear();

    // Create the panels
    for (int i = 0; i < m_batchForecasts.GetForecastsNb(); i++) {
        auto panel = new asPanelForecast(m_scrolledWindowForecasts, &m_batchForecasts);
        panel->SetParametersFileName(m_batchForecasts.GetForecastFileName(i));
        panel->Layout();
        m_sizerForecasts->Add(panel, 0, wxALL | wxEXPAND, 5);
        // Add to the array
        m_panelsManager->AddPanel(panel);
    }

    InitOverallProgress();

    Layout();  // For the scrollbar
    Thaw();

    return true;
}

void asFrameForecaster::Update() {
    DisplayLogLevelMenu();
}

void asFrameForecaster::OpenFramePredictandDB(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameForecaster::OnConfigureDirectories(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesForecaster(this, &m_batchForecasts);
    frame->Fit();
    frame->Show();
}

void asFrameForecaster::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesForecaster(this, &m_batchForecasts);
    frame->Fit();
    frame->Show();
}

void asFrameForecaster::OpenFrameAbout(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameForecaster::OnShowLog(wxCommandEvent& event) {
    wxBusyCursor wait;

    wxASSERT(m_logWindow);
    m_logWindow->DoShow(true);
}

void asFrameForecaster::OnLogLevel1(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecaster::OnLogLevel2(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecaster::OnLogLevel3(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecaster::OnStatusMethodUpdate(wxCommandEvent& event) {
    int eventInt = event.GetInt();
    wxEventType eventType = event.GetEventType();

    if (eventType == asEVT_STATUS_STARTING) {
        m_panelsManager->SetForecastsAllLedsOff();
    } else if (eventType == asEVT_STATUS_FAILED) {
        m_panelsManager->SetForecastLedError(eventInt);
        IncrementOverallProgress();
    } else if (eventType == asEVT_STATUS_SUCCESS) {
        m_panelsManager->SetForecastLedDone(eventInt);
        IncrementOverallProgress();
    } else if (eventType == asEVT_STATUS_DOWNLOADING) {
        m_ledDownloading->SetColour(awxLED_YELLOW);
        m_ledDownloading->SetState(awxLED_ON);
        m_ledDownloading->Refresh();
    } else if (eventType == asEVT_STATUS_DOWNLOADED) {
        m_ledDownloading->SetColour(awxLED_GREEN);
        m_ledDownloading->SetState(awxLED_ON);
        m_ledDownloading->Refresh();
    } else if (eventType == asEVT_STATUS_LOADING) {
        m_ledLoading->SetColour(awxLED_YELLOW);
        m_ledLoading->SetState(awxLED_ON);
        m_ledLoading->Refresh();
    } else if (eventType == asEVT_STATUS_LOADED) {
        m_ledLoading->SetColour(awxLED_GREEN);
        m_ledLoading->SetState(awxLED_ON);
        m_ledLoading->Refresh();
    } else if (eventType == asEVT_STATUS_SAVING) {
        m_ledSaving->SetColour(awxLED_YELLOW);
        m_ledSaving->SetState(awxLED_ON);
        m_ledSaving->Refresh();
    } else if (eventType == asEVT_STATUS_SAVED) {
        m_ledSaving->SetColour(awxLED_GREEN);
        m_ledSaving->SetState(awxLED_ON);
        m_ledSaving->Refresh();
    } else if (eventType == asEVT_STATUS_PROCESSING) {
        m_ledProcessing->SetColour(awxLED_YELLOW);
        m_ledProcessing->SetState(awxLED_ON);
        m_ledProcessing->Refresh();
    } else if (eventType == asEVT_STATUS_PROCESSED) {
        m_ledProcessing->SetColour(awxLED_GREEN);
        m_ledProcessing->SetState(awxLED_ON);
        m_ledProcessing->Refresh();
    } else if ((eventType == asEVT_STATUS_RUNNING)) {
        m_panelsManager->SetForecastLedRunning(eventInt);
        m_ledDownloading->SetState(awxLED_OFF);
        m_ledLoading->SetState(awxLED_OFF);
        m_ledProcessing->SetState(awxLED_OFF);
        m_ledSaving->SetState(awxLED_OFF);
    } else {
        wxLogError(_("Event not identified."));
    }
}

void asFrameForecaster::DisplayLogLevelMenu() {
    // Set log level in the menu
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l)) {
        case 1:
            m_menuLogLevel->FindItemByPosition(0)->Check(true);
            Log()->SetLevel(1);
            break;
        case 2:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
            break;
        case 3:
            m_menuLogLevel->FindItemByPosition(2)->Check(true);
            Log()->SetLevel(3);
            break;
        default:
            m_menuLogLevel->FindItemByPosition(1)->Check(true);
            Log()->SetLevel(2);
    }
}

void asFrameForecaster::LaunchForecasting(wxCommandEvent& event) {
    wxBusyCursor wait;

    UpdateBatchForecasts();
    InitOverallProgress();

    // Get date
    double forecastDate = GetForecastDate();
    wxString forecastDateStr = asTime::GetStringTime(forecastDate, "DD.MM.YYYY hh:mm");
    wxLogVerbose(_("Trying to run the forecast for the date %s"), forecastDateStr);

    if (m_forecaster) {
        wxLogError(_("The forecaster is already processing."));
        return;
    }

    // Launch forecasting
    m_forecaster = new asMethodForecasting(&m_batchForecasts, this);
    m_forecaster->SetForecastDate(forecastDate);
    if (!m_forecaster->Manager()) {
        wxLogError(_("Failed processing the forecast."));

        wxDELETE(m_forecaster);

        return;
    }

    double realForecastDate = m_forecaster->GetForecastDate();
    SetForecastDate(realForecastDate);

    // Log message
    wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
    wxLogVerbose(_("Forecast processed for the date %s"), realForecastDateStr);

    wxDELETE(m_forecaster);

    InitOverallProgress();
}

void asFrameForecaster::CancelForecasting(wxCommandEvent& event) {
    if (m_forecaster) {
        m_forecaster->Cancel();
    }
}

void asFrameForecaster::AddForecast(wxCommandEvent& event) {
    Freeze();
    auto panel = new asPanelForecast(m_scrolledWindowForecasts, &m_batchForecasts);
    panel->Layout();
    m_sizerForecasts->Add(panel, 0, wxALL | wxEXPAND, 5);
    Layout();  // For the scrollbar
    Thaw();

    // Add to the array
    m_panelsManager->AddPanel(panel);
}

void asFrameForecaster::OnSetPresentDate(wxCommandEvent& event) {
    SetPresentDate();
}

void asFrameForecaster::SetPresentDate() {
    // Set the present date in the calendar and the hour field
    wxDateTime nowWx = asTime::NowWxDateTime(asUTM);
    Time nowStruct = asTime::NowTimeStruct(asUTM);
    wxString hourStr = asStrF("%d", nowStruct.hour);
    m_calendarForecastDate->SetDate(nowWx);
    m_textCtrlForecastHour->SetValue(hourStr);
}

double asFrameForecaster::GetForecastDate() const {
    // Date
    wxDateTime forecastDateWx = m_calendarForecastDate->GetDate();
    double forecastDate = asTime::GetMJD(forecastDateWx);

    // Hour
    wxString forecastHourStr = m_textCtrlForecastHour->GetValue();
    double forecastHour = 0;
    forecastHourStr.ToDouble(&forecastHour);

    // Sum
    double total = forecastDate + forecastHour / (double)24;

    return total;
}

void asFrameForecaster::SetForecastDate(double date) {
    // Calendar
    wxDateTime forecastDateWx = asTime::GetWxDateTime(date);
    m_calendarForecastDate->SetDate(forecastDateWx);
    // Hour
    Time forecastDateStruct = asTime::GetTimeStruct(date);
    wxString hourStr = asStrF("%d", forecastDateStruct.hour);
    m_textCtrlForecastHour->SetValue(hourStr);
}

void asFrameForecaster::UpdateRecentFiles() {
    wxASSERT(m_fileHistory);

    for (int i = 0; i < m_fileHistory->GetCount(); ++i) {
        wxString filePath = m_fileHistory->GetHistoryFile(i);
        if (!wxFileExists(filePath)) {
            m_fileHistory->RemoveFileFromHistory(i);
            --i;
        }
    }
}

void asFrameForecaster::SetRecentFiles() {
    wxConfigBase* config = wxFileConfig::Get();
    config->SetPath("/Recent");

    wxMenuItem* menuItem = m_menuBar->FindItem(asID_MENU_RECENT);
    if (menuItem->IsSubMenu()) {
        wxMenu* menu = menuItem->GetSubMenu();
        if (menu) {
            m_fileHistory->Load(*config);
            UpdateRecentFiles();
            m_fileHistory->UseMenu(menu);
            m_fileHistory->AddFilesToMenu(menu);
        }
    }

    config->SetPath("..");
}

void asFrameForecaster::SaveRecentFiles() {
    wxASSERT(m_fileHistory);
    wxConfigBase* config = wxFileConfig::Get();
    config->SetPath("/Recent");

    m_fileHistory->Save(*config);

    config->SetPath("..");
}

void asFrameForecaster::InitOverallProgress() {
    m_gauge->SetRange(m_batchForecasts.GetForecastsNb());
    m_gauge->SetValue(0);

    m_staticTextProgressActual->SetLabel('0');
    wxString totForecastsNb;
    totForecastsNb << m_batchForecasts.GetForecastsNb();
    m_staticTextProgressTot->SetLabel(totForecastsNb);
}

void asFrameForecaster::IncrementOverallProgress() {
    int gaugeValue = m_gauge->GetValue() + 1;
    m_gauge->SetValue(gaugeValue);

    wxString forecastsNb;
    forecastsNb << gaugeValue;
    m_staticTextProgressActual->SetLabel(forecastsNb);

    m_staticTextProgressActual->GetParent()->Layout();

#if USE_GUI
    wxYield();
#endif
}
