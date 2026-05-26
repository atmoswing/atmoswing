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

#include "asIncludes.h"

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
    _forecaster = nullptr;
    _logWindow = nullptr;
    _fileHistory = new wxFileHistory(9);

    // Fix colors
    // _panelMain->SetBackgroundColour(asConfig::GetFrameBgColour());

    // Menu recent
    auto menuOpenRecent = new wxMenu();
    _menuFile->Insert(1, asID_MENU_RECENT, _("Open recent"), menuOpenRecent);

    // Toolbar
    _toolBar->AddTool(asID_RUN, wxT("Run"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::RUN), wxNullBitmap, wxITEM_NORMAL,
                      _("Run forecast"), _("Run forecast now"), nullptr);
    _toolBar->AddTool(asID_CANCEL, wxT("Cancel"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::STOP), wxNullBitmap,
                      wxITEM_NORMAL, _("Cancel forecast"), _("Cancel current forecast"), nullptr);
    _toolBar->AddTool(asID_PREFERENCES, wxT("Preferences"), asBitmaps::Get(asBitmaps::ID_TOOLBAR::PREFERENCES),
                      wxNullBitmap, wxITEM_NORMAL, _("Preferences"), _("Preferences"), nullptr);
    _toolBar->Realize();

    // Leds
    _ledDownloading = new awxLed(_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    _ledDownloading->SetState(awxLED_OFF);
    _sizerLeds->Add(_ledDownloading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textDownloading = new wxStaticText(_panelMain, wxID_ANY, _("Downloading predictors"));
    textDownloading->Wrap(-1);
    _sizerLeds->Add(textDownloading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    _ledLoading = new awxLed(_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    _ledLoading->SetState(awxLED_OFF);
    _sizerLeds->Add(_ledLoading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textLoading = new wxStaticText(_panelMain, wxID_ANY, _("Loading data"));
    textLoading->Wrap(-1);
    _sizerLeds->Add(textLoading, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    _ledProcessing = new awxLed(_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    _ledProcessing->SetState(awxLED_OFF);
    _sizerLeds->Add(_ledProcessing, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textProcessing = new wxStaticText(_panelMain, wxID_ANY, _("Processing"));
    textProcessing->Wrap(-1);
    _sizerLeds->Add(textProcessing, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    _ledSaving = new awxLed(_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0);
    _ledSaving->SetState(awxLED_OFF);
    _sizerLeds->Add(_ledSaving, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    auto textSaving = new wxStaticText(_panelMain, wxID_ANY, _("Saving results"));
    textSaving->Wrap(-1);
    _sizerLeds->Add(textSaving, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    // Buttons
    _bpButtonNow->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::UPDATE));
    _bpButtonAdd->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::PLUS));

    // Create panels manager
    _panelsManager = new asPanelsManagerForecasts();

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
    wxDELETE(_panelsManager);

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
        if (!_batchForecasts.Load(batchFilePath)) {
            wxLogWarning(_("Failed to open the batch file ") + batchFilePath);
        }

        OpenBatchForecasts();
    } else {
        asWizardBatchForecasts wizard(this, &_batchForecasts);
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
    if (!_batchForecasts.Load(batchFilePath)) {
        wxLogError(_("Failed to open the batch file ") + batchFilePath);
    }

    OpenBatchForecasts();

    _fileHistory->AddFileToHistory(batchFilePath);
}

void asFrameForecaster::OnFileHistory(wxCommandEvent& event) {
    int id = event.GetId() - wxID_FILE1;
    wxString batchFilePath = _fileHistory->GetHistoryFile(id);

    wxBusyCursor wait;

    // Save last opened
    wxConfigBase* pConfig = wxFileConfig::Get();
    pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);

    // Do open the batch file
    if (!_batchForecasts.Load(batchFilePath)) {
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
    _batchForecasts.SetFilePath(batchFilePath);

    if (SaveBatchForecasts()) {
        // Save preferences
        wxConfigBase* pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);
    }
}

bool asFrameForecaster::SaveBatchForecasts() {
    wxBusyCursor wait;

    UpdateBatchForecasts();

    if (!_batchForecasts.Save()) {
        wxLogError(_("Could not save the batch file."));
        return false;
    }

    _batchForecasts.SetHasChanged(false);

    return true;
}

bool asFrameForecaster::UpdateBatchForecasts() {
    _batchForecasts.ClearForecasts();

    for (int i = 0; i < _panelsManager->GetPanelsNb(); i++) {
        asPanelForecast* panel = _panelsManager->GetPanel(i);

        _batchForecasts.AddForecast();

        _batchForecasts.SetForecastFileName(i, panel->GetParametersFileName());
    }

    return true;
}

void asFrameForecaster::OnNewBatchForecasts(wxCommandEvent& event) {
    asWizardBatchForecasts wizard(this, &_batchForecasts);
    wizard.RunWizard(wizard.GetSecondPage());
}

bool asFrameForecaster::OpenBatchForecasts() {
    wxBusyCursor wait;

    Freeze();

    wxFileName batchFileName = wxFileName(_batchForecasts.GetFilePath());
    _staticTextbatchFile->SetLabel(batchFileName.GetFullName());

    // Cleanup the actual panels
    _panelsManager->Clear();

    // Create the panels
    for (int i = 0; i < _batchForecasts.GetForecastsNb(); i++) {
        auto panel = new asPanelForecast(_scrolledWindowForecasts, &_batchForecasts);
        panel->SetParametersFileName(_batchForecasts.GetForecastFileName(i));
        panel->Layout();
        _sizerForecasts->Add(panel, 0, wxALL | wxEXPAND, 5);
        // Add to the array
        _panelsManager->AddPanel(panel);
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

    auto frame = new asFramePreferencesForecaster(this, &_batchForecasts);
    frame->Fit();
    frame->Show();
}

void asFrameForecaster::OpenFramePreferences(wxCommandEvent& event) {
    wxBusyCursor wait;

    auto frame = new asFramePreferencesForecaster(this, &_batchForecasts);
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

    wxASSERT(_logWindow);
    _logWindow->DoShow(true);
}

void asFrameForecaster::OnLogLevel1(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(1);
    _menuLogLevel->FindItemByPosition(0)->Check(true);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecaster::OnLogLevel2(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(2);
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(true);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecaster::OnLogLevel3(wxCommandEvent& event) {
    wxBusyCursor wait;

    Log()->SetLevel(3);
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    wxWindow* prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameForecaster::OnStatusMethodUpdate(wxCommandEvent& event) {
    int eventInt = event.GetInt();
    wxEventType eventType = event.GetEventType();

    if (eventType == asEVT_STATUS_STARTING) {
        _panelsManager->SetForecastsAllLedsOff();
    } else if (eventType == asEVT_STATUS_FAILED) {
        _panelsManager->SetForecastLedError(eventInt);
        IncrementOverallProgress();
    } else if (eventType == asEVT_STATUS_SUCCESS) {
        _panelsManager->SetForecastLedDone(eventInt);
        IncrementOverallProgress();
    } else if (eventType == asEVT_STATUS_DOWNLOADING) {
        _ledDownloading->SetColour(awxLED_YELLOW);
        _ledDownloading->SetState(awxLED_ON);
        _ledDownloading->Refresh();
    } else if (eventType == asEVT_STATUS_DOWNLOADED) {
        _ledDownloading->SetColour(awxLED_GREEN);
        _ledDownloading->SetState(awxLED_ON);
        _ledDownloading->Refresh();
    } else if (eventType == asEVT_STATUS_LOADING) {
        _ledLoading->SetColour(awxLED_YELLOW);
        _ledLoading->SetState(awxLED_ON);
        _ledLoading->Refresh();
    } else if (eventType == asEVT_STATUS_LOADED) {
        _ledLoading->SetColour(awxLED_GREEN);
        _ledLoading->SetState(awxLED_ON);
        _ledLoading->Refresh();
    } else if (eventType == asEVT_STATUS_SAVING) {
        _ledSaving->SetColour(awxLED_YELLOW);
        _ledSaving->SetState(awxLED_ON);
        _ledSaving->Refresh();
    } else if (eventType == asEVT_STATUS_SAVED) {
        _ledSaving->SetColour(awxLED_GREEN);
        _ledSaving->SetState(awxLED_ON);
        _ledSaving->Refresh();
    } else if (eventType == asEVT_STATUS_PROCESSING) {
        _ledProcessing->SetColour(awxLED_YELLOW);
        _ledProcessing->SetState(awxLED_ON);
        _ledProcessing->Refresh();
    } else if (eventType == asEVT_STATUS_PROCESSED) {
        _ledProcessing->SetColour(awxLED_GREEN);
        _ledProcessing->SetState(awxLED_ON);
        _ledProcessing->Refresh();
    } else if ((eventType == asEVT_STATUS_RUNNING)) {
        _panelsManager->SetForecastLedRunning(eventInt);
        _ledDownloading->SetState(awxLED_OFF);
        _ledLoading->SetState(awxLED_OFF);
        _ledProcessing->SetState(awxLED_OFF);
        _ledSaving->SetState(awxLED_OFF);
    } else {
        wxLogError(_("Event not identified."));
    }
}

void asFrameForecaster::DisplayLogLevelMenu() {
    // Set log level in the menu
    _menuLogLevel->FindItemByPosition(0)->Check(false);
    _menuLogLevel->FindItemByPosition(1)->Check(false);
    _menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (wxFileConfig::Get()->ReadLong("/General/LogLevel", 2l)) {
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

void asFrameForecaster::LaunchForecasting(wxCommandEvent& event) {
    wxBusyCursor wait;

    UpdateBatchForecasts();
    InitOverallProgress();

    // Get date
    double forecastDate = GetForecastDate();
    wxString forecastDateStr = asTime::GetStringTime(forecastDate, "DD.MM.YYYY hh:mm");
    wxLogVerbose(_("Trying to run the forecast for the date %s"), forecastDateStr);

    if (_forecaster) {
        wxLogError(_("The forecaster is already processing."));
        return;
    }

    // Launch forecasting
    _forecaster = new asMethodForecasting(&_batchForecasts, this);
    _forecaster->SetForecastDate(forecastDate);
    if (!_forecaster->Manager()) {
        wxLogError(_("Failed processing the forecast."));

        wxDELETE(_forecaster);

        return;
    }

    double realForecastDate = _forecaster->GetForecastDate();
    SetForecastDate(realForecastDate);

    // Log message
    wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
    wxLogVerbose(_("Forecast processed for the date %s"), realForecastDateStr);

    wxDELETE(_forecaster);

    InitOverallProgress();
}

void asFrameForecaster::CancelForecasting(wxCommandEvent& event) {
    if (_forecaster) {
        _forecaster->Cancel();
    }
}

void asFrameForecaster::AddForecast(wxCommandEvent& event) {
    Freeze();
    auto panel = new asPanelForecast(_scrolledWindowForecasts, &_batchForecasts);
    panel->Layout();
    _sizerForecasts->Add(panel, 0, wxALL | wxEXPAND, 5);
    Layout();  // For the scrollbar
    Thaw();

    // Add to the array
    _panelsManager->AddPanel(panel);
}

void asFrameForecaster::OnSetPresentDate(wxCommandEvent& event) {
    SetPresentDate();
}

void asFrameForecaster::SetPresentDate() {
    // Set the present date in the calendar and the hour field
    wxDateTime nowWx = asTime::NowWxDateTime(asUTM);
    Time nowStruct = asTime::NowTimeStruct(asUTM);
    wxString hourStr = asStrF("%d", nowStruct.hour);
    _calendarForecastDate->SetDate(nowWx);
    _textCtrlForecastHour->SetValue(hourStr);
}

double asFrameForecaster::GetForecastDate() const {
    // Date
    wxDateTime forecastDateWx = _calendarForecastDate->GetDate();
    double forecastDate = asTime::GetMJD(forecastDateWx);

    // Hour
    wxString forecastHourStr = _textCtrlForecastHour->GetValue();
    double forecastHour = 0;
    forecastHourStr.ToDouble(&forecastHour);

    // Sum
    double total = forecastDate + forecastHour / (double)24;

    return total;
}

void asFrameForecaster::SetForecastDate(double date) {
    // Calendar
    wxDateTime forecastDateWx = asTime::GetWxDateTime(date);
    _calendarForecastDate->SetDate(forecastDateWx);
    // Hour
    Time forecastDateStruct = asTime::GetTimeStruct(date);
    wxString hourStr = asStrF("%d", forecastDateStruct.hour);
    _textCtrlForecastHour->SetValue(hourStr);
}

void asFrameForecaster::UpdateRecentFiles() {
    wxASSERT(_fileHistory);

    for (int i = 0; i < _fileHistory->GetCount(); ++i) {
        wxString filePath = _fileHistory->GetHistoryFile(i);
        if (!wxFileExists(filePath)) {
            _fileHistory->RemoveFileFromHistory(i);
            --i;
        }
    }
}

void asFrameForecaster::SetRecentFiles() {
    wxConfigBase* config = wxFileConfig::Get();
    config->SetPath("/Recent");

    wxMenuItem* menuItem = _menuBar->FindItem(asID_MENU_RECENT);
    if (menuItem->IsSubMenu()) {
        wxMenu* menu = menuItem->GetSubMenu();
        if (menu) {
            _fileHistory->Load(*config);
            UpdateRecentFiles();
            _fileHistory->UseMenu(menu);
            _fileHistory->AddFilesToMenu(menu);
        }
    }

    config->SetPath("..");
}

void asFrameForecaster::SaveRecentFiles() {
    wxASSERT(_fileHistory);
    wxConfigBase* config = wxFileConfig::Get();
    config->SetPath("/Recent");

    _fileHistory->Save(*config);

    config->SetPath("..");
}

void asFrameForecaster::InitOverallProgress() {
    _gauge->SetRange(_batchForecasts.GetForecastsNb());
    _gauge->SetValue(0);

    _staticTextProgressActual->SetLabel('0');
    wxString totForecastsNb;
    totForecastsNb << _batchForecasts.GetForecastsNb();
    _staticTextProgressTot->SetLabel(totForecastsNb);
}

void asFrameForecaster::IncrementOverallProgress() {
    int gaugeValue = _gauge->GetValue() + 1;
    _gauge->SetValue(gaugeValue);

    wxString forecastsNb;
    forecastsNb << gaugeValue;
    _staticTextProgressActual->SetLabel(forecastsNb);

    _staticTextProgressActual->GetParent()->Layout();

#if USE_GUI
    wxYield();
#endif
}
