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
#include "asPanelForecastingModel.h"
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
    m_Forecaster = NULL;
    m_LogWindow = NULL;

    // Toolbar
    m_ToolBar->AddTool( asID_RUN, wxT("Run"), img_run, img_run, wxITEM_NORMAL, _("Run forecast"), _("Run forecast now"), NULL );
    m_ToolBar->AddTool( asID_CANCEL, wxT("Cancel"), img_run_cancel, img_run_cancel, wxITEM_NORMAL, _("Cancel forecast"), _("Cancel current forecast"), NULL );
    m_ToolBar->AddTool( asID_DB_CREATE, wxT("Database creation"), img_database_run, img_database_run, wxITEM_NORMAL, _("Database creation"), _("Database creation"), NULL );
    m_ToolBar->AddTool( asID_PREFERENCES, wxT("Preferences"), img_preferences, img_preferences, wxITEM_NORMAL, _("Preferences"), _("Preferences"), NULL );
    m_ToolBar->AddSeparator();
    m_ToolBar->AddTool( asID_FRAME_VIEWER, wxT("Open viewer"), img_frame_viewer, img_frame_viewer, wxITEM_NORMAL, _("Go to viewer"), _("Go to viewer"), NULL );
    m_ToolBar->Realize();

    // Leds
    m_LedDownloading = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedDownloading->SetState( awxLED_OFF );
	m_SizerLeds->Add( m_LedDownloading, 0, wxALL, 5 );
	
	wxStaticText *textDownloading = new wxStaticText( m_PanelMain, wxID_ANY, _("Downloading predictors"), wxDefaultPosition, wxDefaultSize, 0 );
	textDownloading->Wrap( -1 );
	m_SizerLeds->Add( textDownloading, 0, wxALL, 5 );
	
	m_LedLoading = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedLoading->SetState( awxLED_OFF );
	m_SizerLeds->Add( m_LedLoading, 0, wxALL, 5 );
	
	wxStaticText *textLoading = new wxStaticText( m_PanelMain, wxID_ANY, _("Loading data"), wxDefaultPosition, wxDefaultSize, 0 );
	textLoading->Wrap( -1 );
	m_SizerLeds->Add( textLoading, 0, wxALL, 5 );
	
	m_LedProcessing = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedProcessing->SetState( awxLED_OFF );
	m_SizerLeds->Add( m_LedProcessing, 0, wxALL, 5 );
	
	wxStaticText *textProcessing = new wxStaticText( m_PanelMain, wxID_ANY, _("Processing"), wxDefaultPosition, wxDefaultSize, 0 );
	textProcessing->Wrap( -1 );
	m_SizerLeds->Add( textProcessing, 0, wxALL, 5 );
	
	m_LedSaving = new awxLed( m_PanelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_YELLOW, 0 );
	m_LedSaving->SetState( awxLED_OFF );
	m_SizerLeds->Add( m_LedSaving, 0, wxALL, 5 );
	
	wxStaticText *textSaving = new wxStaticText( m_PanelMain, wxID_ANY, _("Saving results"), wxDefaultPosition, wxDefaultSize, 0 );
	textSaving->Wrap( -1 );
	m_SizerLeds->Add( textSaving, 0, wxALL, 5 );

    // Buttons
    m_BpButtonNow->SetBitmapLabel(img_clock_now);
    m_BpButtonAdd->SetBitmapLabel(img_plus);

    // Create panels manager
    m_PanelsManager = new asPanelsManagerForecastingModels();

    // Connect events
    this->Connect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::LaunchForecasting ) );
    this->Connect( asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::CancelForecasting ) );
    this->Connect( asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePredictandDB ) );
    this->Connect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePreferences ) );
    this->Connect( asID_FRAME_VIEWER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::GoToViewer ) );

    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

asFrameMain::~asFrameMain()
{
    wxDELETE(m_PanelsManager);

    // Disconnect events
    this->Disconnect( asID_RUN, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::LaunchForecasting ) );
    this->Disconnect( asID_CANCEL, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::CancelForecasting ) );
    this->Disconnect( asID_DB_CREATE, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePredictandDB ) );
    this->Disconnect( asID_PREFERENCES, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::OpenFramePreferences ) );
    this->Disconnect( asID_FRAME_VIEWER, wxEVT_COMMAND_TOOL_CLICKED, wxCommandEventHandler( asFrameMain::GoToViewer ) );
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
        if (!m_BatchForecasts.Load(batchFilePath))
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
        asWizardBatchForecasts wizard(this, &m_BatchForecasts);
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
    if (!m_BatchForecasts.Load(batchFilePath))
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
    m_BatchForecasts.SetFilePath(batchFilePath);

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
    m_BatchForecasts.ClearModels();

    for (int i=0; i<m_PanelsManager->GetPanelsNb(); i++)
    {
        asPanelForecastingModel* panel = m_PanelsManager->GetPanel(i);

        m_BatchForecasts.SetModelName(i, panel->GetModelName());
        m_BatchForecasts.SetModelDescription(i, panel->GetModelDescription());
        m_BatchForecasts.SetModelFileName(i, panel->GetParametersFileName());
        m_BatchForecasts.SetModelPredictandDB(i, panel->GetPredictandDBName());
    }

    if(!m_BatchForecasts.Save())
    {
        asLogError(_("Could not save the batch file."));
        return false;
    }

    m_BatchForecasts.SetHasChanged(false);

    return true;
}

void asFrameMain::OnNewBatchForecasts(wxCommandEvent & event)
{
    asWizardBatchForecasts wizard(this, &m_BatchForecasts);
    wizard.RunWizard(wizard.GetSecondPage());
}

bool asFrameMain::OpenBatchForecasts()
{
    Freeze();

    // Cleanup the actual panels
    m_PanelsManager->Clear();

    // Create the panels
    for (int i=0; i<m_BatchForecasts.GetModelsNb(); i++)
    {
        asPanelForecastingModel *panel = new asPanelForecastingModel( m_ScrolledWindowModels );
        panel->SetModelName(m_BatchForecasts.GetModelName(i));
        panel->SetModelDescription(m_BatchForecasts.GetModelDescription(i));
        panel->SetParametersFileName(m_BatchForecasts.GetModelFileName(i));
        panel->SetPredictandDBName(m_BatchForecasts.GetModelPredictandDB(i));
        panel->ReducePanel();
        panel->Layout();
        m_SizerModels->Add( panel, 0, wxALL|wxEXPAND, 5 );
        // Add to the array
        m_PanelsManager->AddPanel(panel);
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

void asFrameMain::GoToViewer( wxCommandEvent& event )
{
    wxConfigBase *pConfig = wxFileConfig::Get();
    wxString ViewerPath = pConfig->Read("/Paths/ViewerPath", wxEmptyString);

    if(ViewerPath.IsEmpty())
    {
        asLogError(_("Please set the path to the viewer in the preferences."));
        return;
    }

    // Execute
    long processId = wxExecute(ViewerPath, wxEXEC_ASYNC);

    if (processId==0) // if wxEXEC_ASYNC
    {
        asLogError(_("The viewer could not be executed. Please check the path in the preferences."));
    }
}

void asFrameMain::OpenFramePredictandDB( wxCommandEvent& event )
{
    asFramePredictandDB* frame = new asFramePredictandDB(this);
    frame->Fit();
    frame->Show();
}

void asFrameMain::OnConfigureDirectories( wxCommandEvent& event )
{
    asFramePreferencesForecaster* frame = new asFramePreferencesForecaster(this, &m_BatchForecasts);
    frame->Fit();
    frame->Show();
}

void asFrameMain::OpenFramePreferences( wxCommandEvent& event )
{
    asFramePreferencesForecaster* frame = new asFramePreferencesForecaster(this, &m_BatchForecasts);
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
    wxASSERT(m_LogWindow);
    m_LogWindow->DoShow();
}

void asFrameMain::OnLogLevel1( wxCommandEvent& event )
{
    Log().SetLevel(1);
    m_MenuLogLevel->FindItemByPosition(0)->Check(true);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 1l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameMain::OnLogLevel2( wxCommandEvent& event )
{
    Log().SetLevel(2);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(true);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    wxFileConfig::Get()->Write("/General/LogLevel", 2l);
    wxWindow *prefFrame = FindWindowById(asWINDOW_PREFERENCES);
    if (prefFrame) prefFrame->Update();
}

void asFrameMain::OnLogLevel3( wxCommandEvent& event )
{
    Log().SetLevel(3);
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(true);
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
        m_PanelsManager->SetForecastingModelsAllLedsOff();
    }
    else if(eventType==asEVT_STATUS_FAILED)
    {
        m_PanelsManager->SetForecastingModelLedError(eventInt);
    }
    else if(eventType==asEVT_STATUS_SUCCESS)
    {
        m_PanelsManager->SetForecastingModelLedDone(eventInt);
    }
    else if(eventType==asEVT_STATUS_DOWNLOADING)
    {
        m_LedDownloading->SetColour(awxLED_YELLOW);
        m_LedDownloading->SetState(awxLED_ON);
        m_LedDownloading->Refresh();
    }
    else if(eventType==asEVT_STATUS_DOWNLOADED)
    {
        m_LedDownloading->SetColour(awxLED_GREEN);
        m_LedDownloading->SetState(awxLED_ON);
        m_LedDownloading->Refresh();
    }
    else if(eventType==asEVT_STATUS_LOADING)
    {
        m_LedLoading->SetColour(awxLED_YELLOW);
        m_LedLoading->SetState(awxLED_ON);
        m_LedLoading->Refresh();
    }
    else if(eventType==asEVT_STATUS_LOADED)
    {
        m_LedLoading->SetColour(awxLED_GREEN);
        m_LedLoading->SetState(awxLED_ON);
        m_LedLoading->Refresh();
    }
    else if(eventType==asEVT_STATUS_SAVING)
    {
        m_LedSaving->SetColour(awxLED_YELLOW);
        m_LedSaving->SetState(awxLED_ON);
        m_LedSaving->Refresh();
    }
    else if(eventType==asEVT_STATUS_SAVED)
    {
        m_LedSaving->SetColour(awxLED_GREEN);
        m_LedSaving->SetState(awxLED_ON);
        m_LedSaving->Refresh();
    }
    else if(eventType==asEVT_STATUS_PROCESSING)
    {
        m_LedProcessing->SetColour(awxLED_YELLOW);
        m_LedProcessing->SetState(awxLED_ON);
        m_LedProcessing->Refresh();
    }
    else if(eventType==asEVT_STATUS_PROCESSED)
    {
        m_LedProcessing->SetColour(awxLED_GREEN);
        m_LedProcessing->SetState(awxLED_ON);
        m_LedProcessing->Refresh();
    }
    else if( (eventType==asEVT_STATUS_RUNNING) )
    {
        m_PanelsManager->SetForecastingModelLedRunning(eventInt);
        m_LedDownloading->SetState(awxLED_OFF);
        m_LedLoading->SetState(awxLED_OFF);
        m_LedProcessing->SetState(awxLED_OFF);
        m_LedSaving->SetState(awxLED_OFF);
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
    m_MenuLogLevel->FindItemByPosition(0)->Check(false);
    m_MenuLogLevel->FindItemByPosition(1)->Check(false);
    m_MenuLogLevel->FindItemByPosition(2)->Check(false);
    switch (logLevel)
    {
    case 1:
        m_MenuLogLevel->FindItemByPosition(0)->Check(true);
        Log().SetLevel(1);
        break;
    case 2:
        m_MenuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
        break;
    case 3:
        m_MenuLogLevel->FindItemByPosition(2)->Check(true);
        Log().SetLevel(3);
        break;
    default:
        m_MenuLogLevel->FindItemByPosition(1)->Check(true);
        Log().SetLevel(2);
    }
}

void asFrameMain::LaunchForecasting( wxCommandEvent& event )
{
    // Get date
    double forecastDate = GetForecastDate();
    wxString forecastDateStr = asTime::GetStringTime(forecastDate, "DD.MM.YYYY hh:mm");
    asLogMessage(wxString::Format(_("Trying to run the forecast for the date %s"), forecastDateStr.c_str()));

    if (m_Forecaster)
    {
        asLogError(_("The forecaster is already processing."));
        return;
    }

    // Launch forecasting
    m_Forecaster = new asMethodForecasting(&m_BatchForecasts, this);
    m_Forecaster->SetForecastDate(forecastDate);
    if(!m_Forecaster->Manager())
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

        wxDELETE(m_Forecaster);

        return;
    }

    double realForecastDate = m_Forecaster->GetForecastDate();
    SetForecastDate(realForecastDate);

    // Log message
    wxString realForecastDateStr = asTime::GetStringTime(realForecastDate, "DD.MM.YYYY hh:mm");
    asLogMessage(wxString::Format(_("Forecast processed for the date %s"), realForecastDateStr.c_str()));

    wxDELETE(m_Forecaster);
}

void asFrameMain::CancelForecasting( wxCommandEvent& event )
{
    if (m_Forecaster)
    {
        m_Forecaster->Cancel();
    }
}

void asFrameMain::AddForecastingModel( wxCommandEvent& event )
{
    Freeze();
    asPanelForecastingModel *panel = new asPanelForecastingModel( m_ScrolledWindowModels );
    panel->Layout();
    m_SizerModels->Add( panel, 0, wxALL|wxEXPAND, 5 );
    Layout(); // For the scrollbar
    Thaw();

    // Add to the array
    m_PanelsManager->AddPanel(panel);
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
    m_CalendarForecastDate->SetDate(nowWx);
    m_TextCtrlForecastHour->SetValue(hourStr);
}

double asFrameMain::GetForecastDate( )
{
    // Date
    wxDateTime forecastDateWx = m_CalendarForecastDate->GetDate();
    double forecastDate = asTime::GetMJD(forecastDateWx);

    // Hour
    wxString forecastHourStr = m_TextCtrlForecastHour->GetValue();
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
    m_CalendarForecastDate->SetDate(forecastDateWx);
    // Hour
    TimeStruct forecastDateStruct = asTime::GetTimeStruct(date);
    wxString hourStr = wxString::Format("%d", forecastDateStruct.hour);
    m_TextCtrlForecastHour->SetValue(hourStr);
}
