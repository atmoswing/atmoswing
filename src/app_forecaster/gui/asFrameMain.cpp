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
 
#include "asFrameMain.h"

#include "asFramePredictandDB.h"
#include "asFramePreferencesForecaster.h"
#include "asFrameXmlEditor.h"
#include "asFrameAbout.h"
#include "asPanelForecast.h"
#include "asWizardBatchForecasts.h"
#include "img_bullets.h"
#include "img_toolbar.h"
#include "img_logo.h"


BEGIN_EVENT_TABLE(asFrameMain, wxFrame)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_STARTING, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_RUNNING, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_FAILED, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_SUCCESS, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_DOWNLOADING, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_DOWNLOADED, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_LOADING, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_LOADED, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_SAVING, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_SAVED, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_PROCESSING, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_STATUS_PROCESSED, asFrameMain::OnStatusMethodUpdate)
    EVT_COMMAND(wxID_ANY, asEVT_ACTION_OPEN_BATCHFORECASTS, asFrameMain::OnOpenBatchForecasts)
END_EVENT_TABLE()


asFrameMain::asFrameMain( wxWindow* parent )
:
asFrameMainVirtual( parent )
{
    m_forecaster = NULL;
    m_logWindow = NULL;

    // Toolbar
    m_toolBar->AddTool( asID_RUN, wxT("Run"), img_run, img_run, wxITEM_NORMAL, _("Run forecast"), _("Run forecast now"), NULL );
    m_toolBar->AddTool( asID_CANCEL, wxT("Cancel"), img_run_cancel, img_run_cancel, wxITEM_NORMAL, _("Cancel forecast"), _("Cancel current forecast"), NULL );
    m_toolBar->AddTool( asID_DB_CREATE, wxT("Database creation"), img_database_run, img_database_run, wxITEM_NORMAL, _("Database creation"), _("Database creation"), NULL );
    m_toolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );
    m_toolBar->Realize();

    // Leds
    m_ledDownloading = new awxLed( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_ledDownloading->SetState( awxLED_OFF );
	m_sizerLeds->Add( m_ledDownloading, 0, wxALL, 5 );
	
	wxStaticText *textDownloading = new wxStaticText( m_panelMain, wxID_ANY, _("Downloading predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	textDownloading->Wrap( -1 );
	m_sizerLeds->Add( textDownloading, 0, wxALL, 5 );
	
	m_ledLoading = new awxLed( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_ledLoading->SetState( awxLED_OFF );
	m_sizerLeds->Add( m_ledLoading, 0, wxALL, 5 );
	
	wxStaticText *textLoading = new wxStaticText( m_panelMain, wxID_ANY, _("Loading data"), wxDefaultPosition, wxDefaultSize, 0 );
	textLoading->Wrap( -1 );
	m_sizerLeds->Add( textLoading, 0, wxALL, 5 );
	
	m_ledProcessing = new awxLed( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_ledProcessing->SetState( awxLED_OFF );
	m_sizerLeds->Add( m_ledProcessing, 0, wxALL, 5 );
	
	wxStaticText *textProcessing = new wxStaticText( m_panelMain, wxID_ANY, _("Processing"), wxDefaultPosition, wxDefaultSize, 0 );
	textProcessing->Wrap( -1 );
	m_sizerLeds->Add( textProcessing, 0, wxALL, 5 );
	
	m_ledSaving = new awxLed( m_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_ledSaving->SetState( awxLED_OFF );
	m_sizerLeds->Add( m_ledSaving, 0, wxALL, 5 );
	
	wxStaticText *textSaving = new wxStaticText( m_panelMain, wxID_ANY, _("Saving results"), wxDefaultPosition, wxDefaultSize, 0 );
	textSaving->Wrap( -1 );
	m_sizerLeds->Add( textSaving, 0, wxALL, 5 );

    // Buttons
    m_bpButtonNow->SetBitmapLabel(img_clock_now);
    m_bpButtonAdd->SetBitmapLabel(img_plus);

    // Create panels manager
    m_panelsManager = new asPanelsManagerForecasts();

    // Connect events
    this->Connect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::LaunchForecasting ) );
    this->Connect( asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::CancelForecasting ) );
    this->Connect( asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePredictandDB ) );
    this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePreferences ) );

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameMain::~asFrameMain()
{
    wxDELETE(m_panelsManager);

    // Disconnect events
    this->Disconnect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::LaunchForecasting ) );
    this->Disconnect( asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::CancelForecasting ) );
    this->Disconnect( asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePredictandDB ) );
    this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePreferences ) );
}

void asFrameMain::OnInit()
{
    DisplayLogLevelMenu();
    SetPresentDate();
    
    // Open last batch file
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString batchFilePath = wxEmptyString;
    pConfig->Read("/BatchForecasts/LastOpened", &batchFilePath);

    if(!batchFilePath.IsEmpty())
    {
        if (!m_batchForecasts.Load(batchFilePath))
        {
            asLogWarning(_("Failed to open the batch file ") + batchFilePath);
        }

        if (!OpenBatchForecasts())
        {
            asLogWarning(_("Failed to open the batch file ") + batchFilePath);
        }
    }
    else
    {
        asWizardBatchForecasts wizard(this, &m_batchForecasts);
        wizard.RunWizard(wizard.GetFirstPage());

        OpenBatchForecasts();
    }
}

void asFrameMain::OnOpenBatchForecasts(wxCommandEvent & event)
{
    // Ask for a batch file
    wxFileDialog openFileDialog (this, _("Select a batch file"),
                            wxEmptyString,
                            wxEmptyString,
                            "xml files (*.xml)|*.xml",
                            wxFD_OPEN | wxFD_FILE_MUST_EXIST | wxFD_CHANGE_DIR);

    // If canceled
    if(openFileDialog.ShowModal()==wxID_CANCEL)
        return;

    wxString batchFilePath = openFileDialog.GetPath();

    // Save preferences
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);

    // Do open the batch file
    if (!m_batchForecasts.Load(batchFilePath))
    {
        asLogError(_("Failed to open the batch file ") + batchFilePath);
    }

    if (!OpenBatchForecasts())
    {
        asLogError(_("Failed to open the batch file ") + batchFilePath);
    }

}

void asFrameMain::OnSaveBatchForecasts(wxCommandEvent & event)
{
    SaveBatchForecasts();
}

void asFrameMain::OnSaveBatchForecastsAs(wxCommandEvent & event)
{
    // Ask for a batch file
    wxFileDialog openFileDialog (this, _("Select a path to save the batch file"),
                            wxEmptyString,
                            wxEmptyString,
                            "xml files (*.xml)|*.xml",
                            wxFD_SAVE | wxFD_CHANGE_DIR);

    // If canceled
    if(openFileDialog.ShowModal()==wxID_CANCEL)
        return;

    wxString batchFilePath = openFileDialog.GetPath();
    m_batchForecasts.SetFilePath(batchFilePath);

    if(SaveBatchForecasts())
    {
        // Save preferences
        wxConfigBase *pConfig = wxFileConfig::Get();
        pConfig->Write("/BatchForecasts/LastOpened", batchFilePath);
    }
}

bool asFrameMain::SaveBatchForecasts()
{
    // Update the GIS layers
    m_batchForecasts.ClearForecasts();

    for (int i=0; i<m_panelsManager->GetPanelsNb(); i++)
    {
        asPanelForecast* panel = m_panelsManager->GetPanel(i);

        m_batchForecasts.AddForecast();

        m_batchForecasts.SetForecastFileName(i, panel->GetParametersFileName());
    }

    if(!m_batchForecasts.Save())
    {
        asLogError(_("Could not save the batch file."));
        return false;
    }

    m_batchForecasts.SetHasChanged(false);

    return true;
}

void asFrameMain::OnNewBatchForecasts(wxCommandEvent & event)
{
    asWizardBatchForecasts wizard(this, &m_batchForecasts);
    wizard.RunWizard(wizard.GetSecondPage());
}

bool asFrameMain::OpenBatchForecasts()
{
    Freeze();

    // Cleanup the actual panels
    m_panelsManager->Clear();

    // Create the panels
    for (int i=0; i<m_batchForecasts.GetForecastsNb(); i++)
    {
        asPanelForecast *panel = new asPanelForecast( m_scrolledWindowForecasts );
        panel->SetParametersFileName(m_batchForecasts.GetForecastFileName(i));
        panel->Layout();
        m_sizerForecasts->Add( panel, 0, wxALL|wxEXPAND, 5 );
        // Add to the array
        m_panelsManager->AddPanel(panel);
    }
    
    Layout(); // For the scrollbar
    Thaw();

    return true;
}

void asFrameMain::Update()
{
    DisplayLogLevelMenu();
}

void asFrameMain::OpenFrameXmlEditor( wxCommandEvent& event )
{
    //asFrameXmlEditor* frame = new asFrameXmlEditor(this, asWINDOW_XML_EDITOR);
    //frame->Fit();
    //frame->Show();
}

void asFrameMain::OpenFramePredictandDB( wxCommandEvent& event )
{
    asFramePredictandDB* frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameMain::OnConfigureDirectories( wxCommandEvent& event )
{
    asFramePreferencesForecaster* frame = new asFramePreferencesForecaster(this, &m_batchForecasts);
    frame->Fit();
    frame->Show();
}

void asFrameMain::OpenFramePreferences( wxCommandEvent& event )
{
    asFramePreferencesForecaster* frame = new asFramePreferencesForecaster(this, &m_batchForecasts);
    frame->Fit();
    frame->Show();
}

void asFrameMain::OpenFrameAbout( wxCommandEvent& event )
{
    asFrameAbout* frame = new asFrameAbout(this);
    frame->Fit();
    frame->Show();
}

void asFrameMain::OnShowLog( wxCommandEvent& event )
{
    wxASSERT(m_logWindow);
    m_logWindow->DoShow();
}

void asFrameMain::OnLogLevel1( wxCommandEvent& event )
{
    Log().SetLevel(1);
    m_menuLogLevel->FindItemByPosition(0)->Check(true);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameMain::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(true);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameMain::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(true);
    wxFileConfig::Get()->Write("/General/LogLevel", 3l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameMain::OnStatusMethodUpdate( wxCommandEvent& event )
{
    int eventInt = event.GetInt();
    wxEventType eventType = event.GetEventType();

    if(eventType==asEVT_STATUS_STARTING)
    {
        m_panelsManager->SetForecastsAllLedsOff();
    }
    else if(eventType==asEVT_STATUS_FAILED)
    {
        m_panelsManager->SetForecastLedError(eventInt);
    }
    else if(eventType==asEVT_STATUS_SUCCESS)
    {
        m_panelsManager->SetForecastLedDone(eventInt);
    }
    else if(eventType==asEVT_STATUS_DOWNLOADING)
    {
        m_ledDownloading->SetColour(awxLED_YELLOW);
        m_ledDownloading->SetState(awxLED_ON);
        m_ledDownloading->Refresh();
    }
    else if(eventType==asEVT_STATUS_DOWNLOADED)
    {
        m_ledDownloading->SetColour(awxLED_GREEN);
        m_ledDownloading->SetState(awxLED_ON);
        m_ledDownloading->Refresh();
    }
    else if(eventType==asEVT_STATUS_LOADING)
    {
        m_ledLoading->SetColour(awxLED_YELLOW);
        m_ledLoading->SetState(awxLED_ON);
        m_ledLoading->Refresh();
    }
    else if(eventType==asEVT_STATUS_LOADED)
    {
        m_ledLoading->SetColour(awxLED_GREEN);
        m_ledLoading->SetState(awxLED_ON);
        m_ledLoading->Refresh();
    }
    else if(eventType==asEVT_STATUS_SAVING)
    {
        m_ledSaving->SetColour(awxLED_YELLOW);
        m_ledSaving->SetState(awxLED_ON);
        m_ledSaving->Refresh();
    }
    else if(eventType==asEVT_STATUS_SAVED)
    {
        m_ledSaving->SetColour(awxLED_GREEN);
        m_ledSaving->SetState(awxLED_ON);
        m_ledSaving->Refresh();
    }
    else if(eventType==asEVT_STATUS_PROCESSING)
    {
        m_ledProcessing->SetColour(awxLED_YELLOW);
        m_ledProcessing->SetState(awxLED_ON);
        m_ledProcessing->Refresh();
    }
    else if(eventType==asEVT_STATUS_PROCESSED)
    {
        m_ledProcessing->SetColour(awxLED_GREEN);
        m_ledProcessing->SetState(awxLED_ON);
        m_ledProcessing->Refresh();
    }
    else if( (eventType==asEVT_STATUS_RUNNING) )
    {
        m_panelsManager->SetForecastLedRunning(eventInt);
        m_ledDownloading->SetState(awxLED_OFF);
        m_ledLoading->SetState(awxLED_OFF);
        m_ledProcessing->SetState(awxLED_OFF);
        m_ledSaving->SetState(awxLED_OFF);
    }
    else
    {
        asLogError(_("Event not identified."));
    }
}

void asFrameMain::DisplayLogLevelMenu()
{
    // Set log level in the menu
    int logLevel = (int)wxFileConfig::Get()->Read("/General/LogLevel", 2l);
    m_menuLogLevel->FindItemByPosition(0)->Check(false);
    m_menuLogLevel->FindItemByPosition(1)->Check(false);
    m_menuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel)
    {
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

void asFrameMain::LaunchForecasting( wxCommandEvent& event )
{
    // Get date
    double forecastDate = GetForecastDate();
    wxString forecastDateStr = asTime::GetStringTime(forecastDate, "DD.MM.YYYY hh:mm");
    asLogMessage(wxString::Format(_("Trying to run the forecast for the date %s"), forecastDateStr.c_str()));

    if (m_forecaster)
    {
        asLogError(_("The forecaster is already processing."));
        return;
    }

    // Launch forecasting
    m_forecaster = new asMethodForecasting(&m_batchForecasts, this);
    m_forecaster->SetForecastDate(forecastDate);
    if(!m_forecaster->Manager())
    {
        asLogError(_("Failed processing the forecast."));

// FIXME (Pascal#1#): Send email in case of failure.
        /*
        wxSMTP *smtp = new wxSMTP(NULL);
        smtp->SetHost("smtp.gmail.com");
        wxEmailMessage *msg = new wxEmailMessage("BLABLABLA",
                                                 "Your code really sucks.\n"
                                                 "Fix your code",
                                                 "pascal.horton.job@gmail.com");
        msg->AddAlternative("<html><body><h1>Bug report</h1>\n"
                            "Your code <b>really</b> sucks <p>Fix your code</html>",
                            "text","html");
        msg->AddTo("pascal.horton.job@gmail.com");
        smtp->Send(msg);

        wxSleep(60);
        smtp->Destroy();
        */

        wxDELETE(m_forecaster);

        return;
    }

    double realForecastDate = m_forecaster->GetForecastDate();
    SetForecastDate(realForecastDate);

    // Log message
    wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
    asLogMessage(wxString::Format(_("Forecast processed for the date %s"), realForecastDateStr.c_str()));

    wxDELETE(m_forecaster);
}

void asFrameMain::CancelForecasting( wxCommandEvent& event )
{
    if (m_forecaster)
    {
        m_forecaster->Cancel();
    }
}

void asFrameMain::AddForecast( wxCommandEvent& event )
{
    Freeze();
    asPanelForecast *panel = new asPanelForecast( m_scrolledWindowForecasts );
    panel->Layout();
    m_sizerForecasts->Add( panel, 0, wxALL|wxEXPAND, 5 );
    Layout(); // For the scrollbar
    Thaw();

    // Add to the array
    m_panelsManager->AddPanel(panel);
}

void asFrameMain::OnSetPresentDate( wxCommandEvent& event )
{
    SetPresentDate();
}

void asFrameMain::SetPresentDate( )
{
    // Set the present date in the calendar and the hour field
    wxDateTime nowWx = asTime::NowWxDateTime(asUTM);
    TimeStruct nowStruct = asTime::NowTimeStruct(asUTM);
    wxString hourStr = wxString::Format("%d", nowStruct.hour);
    m_calendarForecastDate->SetDate(nowWx);
    m_textCtrlForecastHour->SetValue(hourStr);
}

double asFrameMain::GetForecastDate( )
{
    // Date
    wxDateTime forecastDateWx = m_calendarForecastDate->GetDate();
    double forecastDate = asTime::GetMJD(forecastDateWx);

    // Hour
    wxString forecastHourStr = m_textCtrlForecastHour->GetValue();
    double forecastHour = 0;
    forecastHourStr.ToDouble(&forecastHour);

    // Sum
    double total = forecastDate + forecastHour/(double)24;

    return total;
}

void asFrameMain::SetForecastDate( double date )
{
    // Calendar
    wxDateTime forecastDateWx = asTime::GetWxDateTime(date);
    m_calendarForecastDate->SetDate(forecastDateWx);
    // Hour
    TimeStruct forecastDateStruct = asTime::GetTimeStruct(date);
    wxString hourStr = wxString::Format("%d", forecastDateStruct.hour);
    m_textCtrlForecastHour->SetValue(hourStr);
}
