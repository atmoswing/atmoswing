///////////////////////////////////////////////////////////////////////////
// C++ code generated with wxFormBuilder (version 4.2.1-0-g80c4cb6)
// http://www.wxformbuilder.org/
//
// PLEASE DO *NOT* EDIT THIS FILE!
///////////////////////////////////////////////////////////////////////////

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif  //__BORLANDC__

#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif  // WX_PRECOMP

#include "AtmoSwingDownscalerGui.h"

///////////////////////////////////////////////////////////////////////////

asFrameDownscalerVirtual::asFrameDownscalerVirtual(wxWindow* parent, wxWindowID id, const wxString& title,
                                                   const wxPoint& pos, const wxSize& size, long style)
    : wxFrame(parent, id, title, pos, size, style) {
    this->SetSizeHints(wxSize(600, 500), wxDefaultSize);

    wxBoxSizer* bSizer4;
    bSizer4 = new wxBoxSizer(wxVERTICAL);

    _panelMain = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer29;
    bSizer29 = new wxBoxSizer(wxVERTICAL);

    _panelControls = new wxPanel(_panelMain, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer5;
    bSizer5 = new wxBoxSizer(wxVERTICAL);

    _staticTextMethod = new wxStaticText(_panelControls, wxID_ANY, _("Select the downscaling method"),
                                         wxDefaultPosition, wxDefaultSize, 0);
    _staticTextMethod->Wrap(-1);
    _staticTextMethod->Hide();

    bSizer5->Add(_staticTextMethod, 0, wxALL, 5);

    wxString _choiceMethodChoices[] = {_("Classic downscaling")};
    int _choiceMethodNChoices = sizeof(_choiceMethodChoices) / sizeof(wxString);
    _choiceMethod = new wxChoice(_panelControls, wxID_ANY, wxDefaultPosition, wxDefaultSize, _choiceMethodNChoices,
                                 _choiceMethodChoices, 0);
    _choiceMethod->SetSelection(0);
    _choiceMethod->Hide();

    bSizer5->Add(_choiceMethod, 0, wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _staticTextFileParameters = new wxStaticText(_panelControls, wxID_ANY,
                                                 _("Select the parameters file for the downscaling"), wxDefaultPosition,
                                                 wxDefaultSize, 0);
    _staticTextFileParameters->Wrap(-1);
    bSizer5->Add(_staticTextFileParameters, 0, wxALL, 5);

    _filePickerParameters = new wxFilePickerCtrl(_panelControls, wxID_ANY, wxEmptyString, _("Select a file"),
                                                 _("*.xml"), wxDefaultPosition, wxDefaultSize, wxFLP_DEFAULT_STYLE);
    bSizer5->Add(_filePickerParameters, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _staticTextFilePredictand = new wxStaticText(_panelControls, wxID_ANY, _("Select the predictand DB file"),
                                                 wxDefaultPosition, wxDefaultSize, 0);
    _staticTextFilePredictand->Wrap(-1);
    bSizer5->Add(_staticTextFilePredictand, 0, wxALL, 5);

    _filePickerPredictand = new wxFilePickerCtrl(_panelControls, wxID_ANY, wxEmptyString, _("Select a file"), _("*.nc"),
                                                 wxDefaultPosition, wxSize(-1, -1), wxFLP_DEFAULT_STYLE);
    bSizer5->Add(_filePickerPredictand, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _staticTextArchivePredictorDir = new wxStaticText(
        _panelControls, wxID_ANY, _("Select the archive predictors directory"), wxDefaultPosition, wxDefaultSize, 0);
    _staticTextArchivePredictorDir->Wrap(-1);
    bSizer5->Add(_staticTextArchivePredictorDir, 0, wxALL, 5);

    _dirPickerArchivePredictor = new wxDirPickerCtrl(_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"),
                                                     wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE);
    bSizer5->Add(_dirPickerArchivePredictor, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _staticTextScenarioPredictorDir = new wxStaticText(_panelControls, wxID_ANY,
                                                       _("Select the predictors directory for the target period"),
                                                       wxDefaultPosition, wxDefaultSize, 0);
    _staticTextScenarioPredictorDir->Wrap(-1);
    bSizer5->Add(_staticTextScenarioPredictorDir, 0, wxALL, 5);

    _dirPickerScenarioPredictor = new wxDirPickerCtrl(_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"),
                                                      wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE);
    bSizer5->Add(_dirPickerScenarioPredictor, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _staticTextDownscalingResultsDir = new wxStaticText(
        _panelControls, wxID_ANY, _("Directory to save downscaling outputs"), wxDefaultPosition, wxDefaultSize, 0);
    _staticTextDownscalingResultsDir->Wrap(-1);
    bSizer5->Add(_staticTextDownscalingResultsDir, 0, wxALL, 5);

    _dirPickerDownscalingResults = new wxDirPickerCtrl(_panelControls, wxID_ANY, wxEmptyString, _("Select a folder"),
                                                       wxDefaultPosition, wxDefaultSize, wxDIRP_DEFAULT_STYLE);
    bSizer5->Add(_dirPickerDownscalingResults, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _checkBoxParallelEvaluations = new wxCheckBox(
        _panelControls, wxID_ANY,
        _("Parallel evaluations when possible (competes with multithreading in the processor)"), wxDefaultPosition,
        wxDefaultSize, 0);
    bSizer5->Add(_checkBoxParallelEvaluations, 0, wxALL, 5);

    wxBoxSizer* bSizer23;
    bSizer23 = new wxBoxSizer(wxHORIZONTAL);

    _staticTextStateLabel = new wxStaticText(_panelControls, wxID_ANY, _("Downscaling state: "), wxDefaultPosition,
                                             wxDefaultSize, 0);
    _staticTextStateLabel->Wrap(-1);
    _staticTextStateLabel->Hide();

    bSizer23->Add(_staticTextStateLabel, 0, wxALL, 5);

    _staticTextState = new wxStaticText(_panelControls, wxID_ANY, _("Not running."), wxDefaultPosition, wxDefaultSize,
                                        0);
    _staticTextState->Wrap(350);
    _staticTextState->Hide();

    bSizer23->Add(_staticTextState, 1, wxALL, 5);

    bSizer5->Add(bSizer23, 0, wxEXPAND, 5);

    _panelControls->SetSizer(bSizer5);
    _panelControls->Layout();
    bSizer5->Fit(_panelControls);
    bSizer29->Add(_panelControls, 1, wxEXPAND | wxALL, 5);

    wxBoxSizer* bSizer15;
    bSizer15 = new wxBoxSizer(wxHORIZONTAL);

    _buttonSaveDefault = new wxButton(_panelMain, wxID_ANY, _("Save as default"), wxDefaultPosition, wxDefaultSize, 0);
    bSizer15->Add(_buttonSaveDefault, 0, 0, 5);

    bSizer29->Add(bSizer15, 0, wxALIGN_RIGHT | wxTOP | wxBOTTOM | wxRIGHT, 5);

    _panelMain->SetSizer(bSizer29);
    _panelMain->Layout();
    bSizer29->Fit(_panelMain);
    bSizer4->Add(_panelMain, 1, wxEXPAND, 5);

    this->SetSizer(bSizer4);
    this->Layout();
    _menuBar = new wxMenuBar(0);
    _menuOptions = new wxMenu();
    wxMenuItem* _menuItemPreferences;
    _menuItemPreferences = new wxMenuItem(_menuOptions, wxID_ANY, wxString(_("Preferences")), wxEmptyString,
                                          wxITEM_NORMAL);
    _menuOptions->Append(_menuItemPreferences);

    _menuBar->Append(_menuOptions, _("Options"));

    _menuTools = new wxMenu();
    wxMenuItem* menuItemBuildPredictandDB;
    menuItemBuildPredictandDB = new wxMenuItem(_menuTools, wxID_ANY, wxString(_("Build predictand DB")), wxEmptyString,
                                               wxITEM_NORMAL);
    _menuTools->Append(menuItemBuildPredictandDB);

    _menuBar->Append(_menuTools, _("Tools"));

    _menuLog = new wxMenu();
    wxMenuItem* _menuItemShowLog;
    _menuItemShowLog = new wxMenuItem(_menuLog, wxID_ANY, wxString(_("Show Log Window")), wxEmptyString, wxITEM_NORMAL);
    _menuLog->Append(_menuItemShowLog);

    _menuLogLevel = new wxMenu();
    wxMenuItem* _menuLogLevelItem = new wxMenuItem(_menuLog, wxID_ANY, _("Log level"), wxEmptyString, wxITEM_NORMAL,
                                                   _menuLogLevel);
    wxMenuItem* _menuItemLogLevel1;
    _menuItemLogLevel1 = new wxMenuItem(_menuLogLevel, wxID_ANY, wxString(_("Only errors")), wxEmptyString,
                                        wxITEM_CHECK);
    _menuLogLevel->Append(_menuItemLogLevel1);

    wxMenuItem* _menuItemLogLevel2;
    _menuItemLogLevel2 = new wxMenuItem(_menuLogLevel, wxID_ANY, wxString(_("Errors and warnings")), wxEmptyString,
                                        wxITEM_CHECK);
    _menuLogLevel->Append(_menuItemLogLevel2);

    wxMenuItem* _menuItemLogLevel3;
    _menuItemLogLevel3 = new wxMenuItem(_menuLogLevel, wxID_ANY, wxString(_("Verbose")), wxEmptyString, wxITEM_CHECK);
    _menuLogLevel->Append(_menuItemLogLevel3);

    _menuLog->Append(_menuLogLevelItem);

    _menuBar->Append(_menuLog, _("Log"));

    _menuHelp = new wxMenu();
    wxMenuItem* _menuItemAbout;
    _menuItemAbout = new wxMenuItem(_menuHelp, wxID_ANY, wxString(_("About")), wxEmptyString, wxITEM_NORMAL);
    _menuHelp->Append(_menuItemAbout);

    _menuBar->Append(_menuHelp, _("Help"));

    this->SetMenuBar(_menuBar);

    _toolBar = this->CreateToolBar(wxTB_HORIZONTAL, wxID_ANY);
    _toolBar->SetToolBitmapSize(wxSize(32, 32));
    _toolBar->Realize();

    _statusBar1 = this->CreateStatusBar(1, wxSTB_SIZEGRIP, wxID_ANY);

    this->Centre(wxBOTH);

    // Connect Events
    _buttonSaveDefault->Connect(wxEVT_COMMAND_BUTTON_CLICKED,
                                wxCommandEventHandler(asFrameDownscalerVirtual::OnSaveDefault), NULL, this);
    _menuOptions->Bind(wxEVT_COMMAND_MENU_SELECTED,
                       wxCommandEventHandler(asFrameDownscalerVirtual::OpenFramePreferences), this,
                       _menuItemPreferences->GetId());
    _menuTools->Bind(wxEVT_COMMAND_MENU_SELECTED,
                     wxCommandEventHandler(asFrameDownscalerVirtual::OpenFramePredictandDB), this,
                     menuItemBuildPredictandDB->GetId());
    _menuLog->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(asFrameDownscalerVirtual::OnShowLog), this,
                   _menuItemShowLog->GetId());
    _menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(asFrameDownscalerVirtual::OnLogLevel1), this,
                        _menuItemLogLevel1->GetId());
    _menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(asFrameDownscalerVirtual::OnLogLevel2), this,
                        _menuItemLogLevel2->GetId());
    _menuLogLevel->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(asFrameDownscalerVirtual::OnLogLevel3), this,
                        _menuItemLogLevel3->GetId());
    _menuHelp->Bind(wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(asFrameDownscalerVirtual::OpenFrameAbout), this,
                    _menuItemAbout->GetId());
}

asFrameDownscalerVirtual::~asFrameDownscalerVirtual() {
    // Disconnect Events
    _buttonSaveDefault->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED,
                                   wxCommandEventHandler(asFrameDownscalerVirtual::OnSaveDefault), NULL, this);
}

asFramePreferencesDownscalerVirtual::asFramePreferencesDownscalerVirtual(wxWindow* parent, wxWindowID id,
                                                                         const wxString& title, const wxPoint& pos,
                                                                         const wxSize& size, long style)
    : wxFrame(parent, id, title, pos, size, style) {
    this->SetSizeHints(wxSize(400, 400), wxDefaultSize);

    wxBoxSizer* bSizer14;
    bSizer14 = new wxBoxSizer(wxVERTICAL);

    _panelBase = new wxPanel(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer15;
    bSizer15 = new wxBoxSizer(wxVERTICAL);

    _notebookBase = new wxNotebook(_panelBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0);
    _panelGeneralCommon = new wxPanel(_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer16;
    bSizer16 = new wxBoxSizer(wxVERTICAL);

    wxStaticBoxSizer* sbSizer6;
    sbSizer6 = new wxStaticBoxSizer(new wxStaticBox(_panelGeneralCommon, wxID_ANY, _("Language")), wxVERTICAL);

    wxString _choiceLocaleChoices[] = {_("English"), _("French")};
    int _choiceLocaleNChoices = sizeof(_choiceLocaleChoices) / sizeof(wxString);
    _choiceLocale = new wxChoice(sbSizer6->GetStaticBox(), wxID_ANY, wxDefaultPosition, wxDefaultSize,
                                 _choiceLocaleNChoices, _choiceLocaleChoices, 0);
    _choiceLocale->SetSelection(0);
    sbSizer6->Add(_choiceLocale, 0, wxALL, 5);

    _staticText21 = new wxStaticText(sbSizer6->GetStaticBox(), wxID_ANY,
                                     _("Restart AtmoSwing for the change to take effect."), wxDefaultPosition,
                                     wxDefaultSize, 0);
    _staticText21->Wrap(-1);
    sbSizer6->Add(_staticText21, 0, wxALL, 5);

    bSizer16->Add(sbSizer6, 0, wxEXPAND | wxALL, 5);

    wxStaticBoxSizer* sbSizer7;
    sbSizer7 = new wxStaticBoxSizer(new wxStaticBox(_panelGeneralCommon, wxID_ANY, _("Logs")), wxVERTICAL);

    wxBoxSizer* bSizer20;
    bSizer20 = new wxBoxSizer(wxHORIZONTAL);

    wxBoxSizer* bSizer17;
    bSizer17 = new wxBoxSizer(wxVERTICAL);

    _radioBtnLogLevel1 = new wxRadioButton(sbSizer7->GetStaticBox(), wxID_ANY, _("Errors only (recommanded)"),
                                           wxDefaultPosition, wxDefaultSize, 0);
    bSizer17->Add(_radioBtnLogLevel1, 0, wxALL, 5);

    _radioBtnLogLevel2 = new wxRadioButton(sbSizer7->GetStaticBox(), wxID_ANY, _("Errors and warnings"),
                                           wxDefaultPosition, wxDefaultSize, 0);
    bSizer17->Add(_radioBtnLogLevel2, 0, wxALL, 5);

    _radioBtnLogLevel3 = new wxRadioButton(sbSizer7->GetStaticBox(), wxID_ANY, _("Verbose"), wxDefaultPosition,
                                           wxDefaultSize, 0);
    bSizer17->Add(_radioBtnLogLevel3, 0, wxALL, 5);

    bSizer20->Add(bSizer17, 1, wxEXPAND, 5);

    wxBoxSizer* bSizer21;
    bSizer21 = new wxBoxSizer(wxVERTICAL);

    _checkBoxDisplayLogWindow = new wxCheckBox(sbSizer7->GetStaticBox(), wxID_ANY, _("Display window"),
                                               wxDefaultPosition, wxDefaultSize, 0);
    _checkBoxDisplayLogWindow->SetValue(true);
    bSizer21->Add(_checkBoxDisplayLogWindow, 0, wxALL, 5);

    _checkBoxSaveLogFile = new wxCheckBox(sbSizer7->GetStaticBox(), wxID_ANY, _("Save to a file"), wxDefaultPosition,
                                          wxDefaultSize, 0);
    _checkBoxSaveLogFile->SetValue(true);
    _checkBoxSaveLogFile->Enable(false);

    bSizer21->Add(_checkBoxSaveLogFile, 0, wxALL, 5);

    bSizer20->Add(bSizer21, 1, wxEXPAND, 5);

    sbSizer7->Add(bSizer20, 1, wxEXPAND, 5);

    bSizer16->Add(sbSizer7, 0, wxALL | wxEXPAND, 5);

    wxStaticBoxSizer* sbSizer18;
    sbSizer18 = new wxStaticBoxSizer(new wxStaticBox(_panelGeneralCommon, wxID_ANY, _("Directories")), wxVERTICAL);

    _staticTextArchivePredictorsDir = new wxStaticText(sbSizer18->GetStaticBox(), wxID_ANY,
                                                       _("Directory containing archive predictors"), wxDefaultPosition,
                                                       wxDefaultSize, 0);
    _staticTextArchivePredictorsDir->Wrap(-1);
    sbSizer18->Add(_staticTextArchivePredictorsDir, 0, wxRIGHT | wxLEFT, 5);

    _dirPickerArchivePredictors = new wxDirPickerCtrl(sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString,
                                                      _("Select a folder"), wxDefaultPosition, wxDefaultSize,
                                                      wxDIRP_USE_TEXTCTRL);
    sbSizer18->Add(_dirPickerArchivePredictors, 0, wxBOTTOM | wxRIGHT | wxLEFT | wxEXPAND, 5);

    _staticTextScenarioPredictorsDir = new wxStaticText(sbSizer18->GetStaticBox(), wxID_ANY,
                                                        _("Directory containing scenario predictors"),
                                                        wxDefaultPosition, wxDefaultSize, 0);
    _staticTextScenarioPredictorsDir->Wrap(-1);
    sbSizer18->Add(_staticTextScenarioPredictorsDir, 0, wxRIGHT | wxLEFT, 5);

    _dirPickerScenarioPredictors = new wxDirPickerCtrl(sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString,
                                                       _("Select a folder"), wxDefaultPosition, wxDefaultSize,
                                                       wxDIRP_USE_TEXTCTRL);
    sbSizer18->Add(_dirPickerScenarioPredictors, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    _staticTextPredictandDBDir = new wxStaticText(
        sbSizer18->GetStaticBox(), wxID_ANY, _("Default predictand DB directory"), wxDefaultPosition, wxDefaultSize, 0);
    _staticTextPredictandDBDir->Wrap(-1);
    sbSizer18->Add(_staticTextPredictandDBDir, 0, wxRIGHT | wxLEFT, 5);

    _dirPickerPredictandDB = new wxDirPickerCtrl(sbSizer18->GetStaticBox(), wxID_ANY, wxEmptyString,
                                                 _("Select a folder"), wxDefaultPosition, wxDefaultSize,
                                                 wxDIRP_USE_TEXTCTRL);
    sbSizer18->Add(_dirPickerPredictandDB, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    bSizer16->Add(sbSizer18, 0, wxEXPAND | wxALL, 5);

    _panelGeneralCommon->SetSizer(bSizer16);
    _panelGeneralCommon->Layout();
    bSizer16->Fit(_panelGeneralCommon);
    _notebookBase->AddPage(_panelGeneralCommon, _("General"), true);
    _panelAdvanced = new wxPanel(_notebookBase, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer26;
    bSizer26 = new wxBoxSizer(wxVERTICAL);

    _notebookAdvanced = new wxNotebook(_panelAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, 0);
    _panelGeneral = new wxPanel(_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer271;
    bSizer271 = new wxBoxSizer(wxVERTICAL);

    wxString _radioBoxGuiChoices[] = {_("Silent (no progressbar, much faster)"), _("Standard (recommanded)"),
                                      _("Verbose (not much used)")};
    int _radioBoxGuiNChoices = sizeof(_radioBoxGuiChoices) / sizeof(wxString);
    _radioBoxGui = new wxRadioBox(_panelGeneral, wxID_ANY, _("GUI options"), wxDefaultPosition, wxDefaultSize,
                                  _radioBoxGuiNChoices, _radioBoxGuiChoices, 1, wxRA_SPECIFY_COLS);
    _radioBoxGui->SetSelection(1);
    bSizer271->Add(_radioBoxGui, 0, wxALL | wxEXPAND, 5);

    _checkBoxResponsiveness = new wxCheckBox(_panelGeneral, wxID_ANY,
                                             _("Let the software be responsive while processing (recommended)."),
                                             wxDefaultPosition, wxDefaultSize, 0);
    _checkBoxResponsiveness->SetValue(true);
    bSizer271->Add(_checkBoxResponsiveness, 0, wxALL, 5);

    _panelGeneral->SetSizer(bSizer271);
    _panelGeneral->Layout();
    bSizer271->Fit(_panelGeneral);
    _notebookAdvanced->AddPage(_panelGeneral, _("General"), true);
    _panelProcessing = new wxPanel(_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer1611;
    bSizer1611 = new wxBoxSizer(wxVERTICAL);

    wxStaticBoxSizer* sbSizer15;
    sbSizer15 = new wxStaticBoxSizer(new wxStaticBox(_panelProcessing, wxID_ANY, _("Multithreading")), wxVERTICAL);

    _checkBoxAllowMultithreading = new wxCheckBox(sbSizer15->GetStaticBox(), wxID_ANY, _("Allow multithreading"),
                                                  wxDefaultPosition, wxDefaultSize, 0);
    _checkBoxAllowMultithreading->SetValue(true);
    sbSizer15->Add(_checkBoxAllowMultithreading, 0, wxALL, 5);

    wxBoxSizer* bSizer221;
    bSizer221 = new wxBoxSizer(wxHORIZONTAL);

    _staticTextThreadsNb = new wxStaticText(sbSizer15->GetStaticBox(), wxID_ANY, _("Max nb of threads"),
                                            wxDefaultPosition, wxDefaultSize, 0);
    _staticTextThreadsNb->Wrap(-1);
    bSizer221->Add(_staticTextThreadsNb, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    _textCtrlThreadsNb = new wxTextCtrl(sbSizer15->GetStaticBox(), wxID_ANY, wxEmptyString, wxDefaultPosition,
                                        wxSize(30, -1), 0);
    bSizer221->Add(_textCtrlThreadsNb, 0, wxRIGHT | wxLEFT | wxALIGN_CENTER_VERTICAL, 5);

    sbSizer15->Add(bSizer221, 0, wxEXPAND, 5);

    wxBoxSizer* bSizer241;
    bSizer241 = new wxBoxSizer(wxHORIZONTAL);

    _staticTextThreadsPriority = new wxStaticText(sbSizer15->GetStaticBox(), wxID_ANY, _("Threads priority"),
                                                  wxDefaultPosition, wxDefaultSize, 0);
    _staticTextThreadsPriority->Wrap(-1);
    bSizer241->Add(_staticTextThreadsPriority, 0, wxALL, 5);

    _sliderThreadsPriority = new wxSlider(sbSizer15->GetStaticBox(), wxID_ANY, 95, 0, 100, wxDefaultPosition,
                                          wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
    bSizer241->Add(_sliderThreadsPriority, 1, wxRIGHT | wxLEFT, 5);

    sbSizer15->Add(bSizer241, 0, wxEXPAND, 5);

    bSizer1611->Add(sbSizer15, 0, wxALL | wxEXPAND, 5);

    wxString _radioBoxProcessingMethodsChoices[] = {_("Multithreaded (only if allowed hereabove)"),
                                                    _("Standard (slower)")};
    int _radioBoxProcessingMethodsNChoices = sizeof(_radioBoxProcessingMethodsChoices) / sizeof(wxString);
    _radioBoxProcessingMethods = new wxRadioBox(_panelProcessing, wxID_ANY, _("Processing options"), wxDefaultPosition,
                                                wxDefaultSize, _radioBoxProcessingMethodsNChoices,
                                                _radioBoxProcessingMethodsChoices, 1, wxRA_SPECIFY_COLS);
    _radioBoxProcessingMethods->SetSelection(0);
    _radioBoxProcessingMethods->SetToolTip(_("These options don't affect the results, only the processor efficiency."));

    bSizer1611->Add(_radioBoxProcessingMethods, 0, wxALL | wxEXPAND, 5);

    _panelProcessing->SetSizer(bSizer1611);
    _panelProcessing->Layout();
    bSizer1611->Fit(_panelProcessing);
    _notebookAdvanced->AddPage(_panelProcessing, _("Processing"), false);
    _panelUserDirectories = new wxPanel(_notebookAdvanced, wxID_ANY, wxDefaultPosition, wxDefaultSize, wxTAB_TRAVERSAL);
    wxBoxSizer* bSizer24;
    bSizer24 = new wxBoxSizer(wxVERTICAL);

    wxStaticBoxSizer* sbSizer411;
    sbSizer411 = new wxStaticBoxSizer(new wxStaticBox(_panelUserDirectories, wxID_ANY, _("Working directories")),
                                      wxVERTICAL);

    _staticTextIntermediateResultsDir = new wxStaticText(sbSizer411->GetStaticBox(), wxID_ANY,
                                                         _("Directory to save intermediate temporary results"),
                                                         wxDefaultPosition, wxDefaultSize, 0);
    _staticTextIntermediateResultsDir->Wrap(-1);
    sbSizer411->Add(_staticTextIntermediateResultsDir, 0, wxALL, 5);

    _dirPickerIntermediateResults = new wxDirPickerCtrl(sbSizer411->GetStaticBox(), wxID_ANY, wxEmptyString,
                                                        _("Select a folder"), wxDefaultPosition, wxDefaultSize,
                                                        wxDIRP_USE_TEXTCTRL);
    sbSizer411->Add(_dirPickerIntermediateResults, 0, wxEXPAND | wxBOTTOM | wxRIGHT | wxLEFT, 5);

    bSizer24->Add(sbSizer411, 0, wxEXPAND | wxALL, 5);

    wxStaticBoxSizer* sbSizer17;
    sbSizer17 = new wxStaticBoxSizer(new wxStaticBox(_panelUserDirectories, wxID_ANY, _("User specific paths")),
                                     wxVERTICAL);

    wxFlexGridSizer* fgSizer9;
    fgSizer9 = new wxFlexGridSizer(5, 2, 0, 0);
    fgSizer9->SetFlexibleDirection(wxBOTH);
    fgSizer9->SetNonFlexibleGrowMode(wxFLEX_GROWMODE_SPECIFIED);

    _staticTextUserDirLabel = new wxStaticText(sbSizer17->GetStaticBox(), wxID_ANY, _("User working directory:"),
                                               wxDefaultPosition, wxDefaultSize, 0);
    _staticTextUserDirLabel->Wrap(-1);
    fgSizer9->Add(_staticTextUserDirLabel, 0, wxALL, 5);

    _staticTextUserDir = new wxStaticText(sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition,
                                          wxDefaultSize, 0);
    _staticTextUserDir->Wrap(-1);
    fgSizer9->Add(_staticTextUserDir, 0, wxALL, 5);

    _staticTextLogFileLabels = new wxStaticText(sbSizer17->GetStaticBox(), wxID_ANY, _("Log file:"), wxDefaultPosition,
                                                wxDefaultSize, 0);
    _staticTextLogFileLabels->Wrap(-1);
    fgSizer9->Add(_staticTextLogFileLabels, 0, wxALL, 5);

    _staticTextLogFile = new wxStaticText(sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition,
                                          wxDefaultSize, 0);
    _staticTextLogFile->Wrap(-1);
    fgSizer9->Add(_staticTextLogFile, 0, wxALL, 5);

    _staticTextPrefFileLabel = new wxStaticText(sbSizer17->GetStaticBox(), wxID_ANY, _("Preferences file:"),
                                                wxDefaultPosition, wxDefaultSize, 0);
    _staticTextPrefFileLabel->Wrap(-1);
    fgSizer9->Add(_staticTextPrefFileLabel, 0, wxALL, 5);

    _staticTextPrefFile = new wxStaticText(sbSizer17->GetStaticBox(), wxID_ANY, _("..."), wxDefaultPosition,
                                           wxDefaultSize, 0);
    _staticTextPrefFile->Wrap(-1);
    fgSizer9->Add(_staticTextPrefFile, 0, wxALL, 5);

    sbSizer17->Add(fgSizer9, 1, wxEXPAND, 5);

    bSizer24->Add(sbSizer17, 0, wxALL | wxEXPAND, 5);

    _panelUserDirectories->SetSizer(bSizer24);
    _panelUserDirectories->Layout();
    bSizer24->Fit(_panelUserDirectories);
    _notebookAdvanced->AddPage(_panelUserDirectories, _("User paths"), false);

    bSizer26->Add(_notebookAdvanced, 1, wxEXPAND | wxALL, 5);

    _panelAdvanced->SetSizer(bSizer26);
    _panelAdvanced->Layout();
    bSizer26->Fit(_panelAdvanced);
    _notebookBase->AddPage(_panelAdvanced, _("Advanced"), false);

    bSizer15->Add(_notebookBase, 1, wxEXPAND | wxALL, 5);

    _buttonsConfirmation = new wxStdDialogButtonSizer();
    _buttonsConfirmationOK = new wxButton(_panelBase, wxID_OK);
    _buttonsConfirmation->AddButton(_buttonsConfirmationOK);
    _buttonsConfirmationApply = new wxButton(_panelBase, wxID_APPLY);
    _buttonsConfirmation->AddButton(_buttonsConfirmationApply);
    _buttonsConfirmationCancel = new wxButton(_panelBase, wxID_CANCEL);
    _buttonsConfirmation->AddButton(_buttonsConfirmationCancel);
    _buttonsConfirmation->Realize();

    bSizer15->Add(_buttonsConfirmation, 0, wxALL | wxEXPAND, 5);

    _panelBase->SetSizer(bSizer15);
    _panelBase->Layout();
    bSizer15->Fit(_panelBase);
    bSizer14->Add(_panelBase, 1, wxEXPAND, 5);

    this->SetSizer(bSizer14);
    this->Layout();

    this->Centre(wxBOTH);

    // Connect Events
    _checkBoxAllowMultithreading->Connect(
        wxEVT_COMMAND_CHECKBOX_CLICKED,
        wxCommandEventHandler(asFramePreferencesDownscalerVirtual::OnChangeMultithreadingCheckBox), NULL, this);
    _buttonsConfirmationApply->Connect(wxEVT_COMMAND_BUTTON_CLICKED,
                                       wxCommandEventHandler(asFramePreferencesDownscalerVirtual::ApplyChanges), NULL,
                                       this);
    _buttonsConfirmationCancel->Connect(wxEVT_COMMAND_BUTTON_CLICKED,
                                        wxCommandEventHandler(asFramePreferencesDownscalerVirtual::CloseFrame), NULL,
                                        this);
    _buttonsConfirmationOK->Connect(wxEVT_COMMAND_BUTTON_CLICKED,
                                    wxCommandEventHandler(asFramePreferencesDownscalerVirtual::SaveAndClose), NULL,
                                    this);
}

asFramePreferencesDownscalerVirtual::~asFramePreferencesDownscalerVirtual() {
    // Disconnect Events
    _checkBoxAllowMultithreading->Disconnect(
        wxEVT_COMMAND_CHECKBOX_CLICKED,
        wxCommandEventHandler(asFramePreferencesDownscalerVirtual::OnChangeMultithreadingCheckBox), NULL, this);
    _buttonsConfirmationApply->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED,
                                          wxCommandEventHandler(asFramePreferencesDownscalerVirtual::ApplyChanges),
                                          NULL, this);
    _buttonsConfirmationCancel->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED,
                                           wxCommandEventHandler(asFramePreferencesDownscalerVirtual::CloseFrame), NULL,
                                           this);
    _buttonsConfirmationOK->Disconnect(wxEVT_COMMAND_BUTTON_CLICKED,
                                       wxCommandEventHandler(asFramePreferencesDownscalerVirtual::SaveAndClose), NULL,
                                       this);
}
