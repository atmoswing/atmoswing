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

#include "asFramePreferencesCalibrator.h"

#include "wx/fileconf.h"
#include "wx/thread.h"

asFramePreferencesCalibrator::asFramePreferencesCalibrator( wxWindow* parent, wxWindowID id )
:
asFramePreferencesCalibratorVirtual( parent, id )
{
    LoadPreferences();
    Fit();

        // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesCalibrator::CloseFrame( wxCommandEvent& event )
{
    Close();
}

void asFramePreferencesCalibrator::Update()
{
    LoadPreferences();
}

void asFramePreferencesCalibrator::LoadPreferences()
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = m_NotebookBase->GetThemeBackgroundColour();
    if (col.IsOk())
    {
        m_DirPickerPredictandDB->SetBackgroundColour(col);
        m_DirPickerIntermediateResults->SetBackgroundColour(col);
        m_DirPickerParameters->SetBackgroundColour(col);
        m_DirPickerArchivePredictors->SetBackgroundColour(col);
    }

    /*
     * General
     */

    // Log
    long defaultLogLevel = 1; // = selection +1
    long logLevel = pConfig->Read("/General/LogLevel", defaultLogLevel);
    m_RadioBoxLogLevel->SetSelection((int)logLevel-1);
    bool displayLogWindow;
    pConfig->Read("/General/DisplayLogWindow", &displayLogWindow, false);
    m_CheckBoxDisplayLogWindow->SetValue(displayLogWindow);

    // Paths
    wxString dirConfig = asConfig::GetDataDir()+"config"+DS;
    wxString dirData = asConfig::GetDataDir()+"data"+DS;
    wxString PredictandDBDir = pConfig->Read("/Paths/DataPredictandDBDir", dirData+"predictands");
    m_DirPickerPredictandDB->SetPath(PredictandDBDir);
    wxString ArchivePredictorsDir = pConfig->Read("/Paths/ArchivePredictorsDir", dirData+"predictors");
    m_DirPickerArchivePredictors->SetPath(ArchivePredictorsDir);
    wxString ForecastParametersDir = pConfig->Read("/Paths/ForecastParametersDir", dirConfig);
    m_DirPickerParameters->SetPath(ForecastParametersDir);

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = pConfig->Read("/General/GuiOptions", 1l);
    m_RadioBoxGui->SetSelection((int)guiOptions);
    if (guiOptions==0)
    {
        g_SilentMode = true;
    }
    else
    {
        g_SilentMode = false;
        g_VerboseMode = false;
        if (guiOptions==2l)
        {
            g_VerboseMode = true;
        }
    }

    // Advanced options
    bool responsive;
    pConfig->Read("/General/Responsive", &responsive, true);
    m_CheckBoxResponsiveness->SetValue(responsive);
    if (responsive)
    {
        g_Responsive = true;
    }
    else
    {
        g_Responsive = false;
    }

    // Multithreading
    bool allowMultithreading;
    pConfig->Read("/General/AllowMultithreading", &allowMultithreading, true);
    m_CheckBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads==-1) maxThreads = 2;
    wxString maxThreadsStr = wxString::Format("%d", maxThreads);
    wxString ProcessingMaxThreadNb = pConfig->Read("/General/ProcessingMaxThreadNb", maxThreadsStr);
    m_TextCtrlThreadsNb->SetValue(ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = pConfig->Read("/General/ProcessingThreadsPriority", 95l);
    m_SliderThreadsPriority->SetValue((int)ProcessingThreadsPriority);

    // Processing
    long defaultMethod = (long)asMULTITHREADS;
    long ProcessingMethod = pConfig->Read("/ProcessingOptions/ProcessingMethod", defaultMethod);
    if (!allowMultithreading)
    {
        m_RadioBoxProcessingMethods->Enable(0, false);
        if (ProcessingMethod==(long)asMULTITHREADS)
        {
            ProcessingMethod = (long)asINSERT;
        }
    }
    else
    {
        m_RadioBoxProcessingMethods->Enable(0, true);
    }
    m_RadioBoxProcessingMethods->SetSelection((int)ProcessingMethod);
    long defaultLinAlgebra = (long)asLIN_ALGEBRA_NOVAR;
    long ProcessingLinAlgebra = pConfig->Read("/ProcessingOptions/ProcessingLinAlgebra", defaultLinAlgebra);
    m_RadioBoxLinearAlgebra->SetSelection((int)ProcessingLinAlgebra);

    // User directories
    wxString IntermediateResultsDir = pConfig->Read("/Paths/IntermediateResultsDir", asConfig::GetTempDir()+"AtmoSwing");
    m_DirPickerIntermediateResults->SetPath(IntermediateResultsDir);
    wxString userpath = asConfig::GetUserDataDir();
    m_StaticTextUserDir->SetLabel(userpath);
    wxString logpath = asConfig::GetLogDir();
    logpath.Append("AtmoSwingCalibrator.log");
    m_StaticTextLogFile->SetLabel(logpath);
    m_StaticTextPrefFile->SetLabel(asConfig::GetUserDataDir("AtmoSwing calibrator")+"AtmoSwing.ini");
}

void asFramePreferencesCalibrator::SavePreferences( )
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * General
     */

    // Log    
    long logLevel = (long)m_RadioBoxLogLevel->GetSelection();
    pConfig->Write("/General/LogLevel", logLevel+1); // = selection +1
    bool displayLogWindow = m_CheckBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindow);

    // Paths
    wxString PredictandDBDir = m_DirPickerPredictandDB->GetPath();
    pConfig->Write("/Paths/DataPredictandDBDir", PredictandDBDir);
    wxString IntermediateResultsDir = m_DirPickerIntermediateResults->GetPath();
    pConfig->Write("/Paths/IntermediateResultsDir", IntermediateResultsDir);
    wxString ArchivePredictorsDir = m_DirPickerArchivePredictors->GetPath();
    pConfig->Write("/Paths/ArchivePredictorsDir", ArchivePredictorsDir);
    wxString ForecastParametersDir = m_DirPickerParameters->GetPath();
    pConfig->Write("/Paths/ForecastParametersDir", ForecastParametersDir);

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = (long)m_RadioBoxGui->GetSelection();
    pConfig->Write("/General/GuiOptions", guiOptions);
    if (guiOptions==0)
    {
        g_SilentMode = true;
    }
    else
    {
        g_SilentMode = false;
        g_VerboseMode = false;
        if (guiOptions==2l)
        {
            g_VerboseMode = true;
        }
    }

    // Advanced options
    bool responsive = m_CheckBoxResponsiveness->GetValue();
    pConfig->Write("/General/Responsive", responsive);
    if (responsive)
    {
        g_Responsive = true;
    }
    else
    {
        g_Responsive = false;
    }

    // Multithreading
    bool allowMultithreading = m_CheckBoxAllowMultithreading->GetValue();
    pConfig->Write("/General/AllowMultithreading", allowMultithreading);
    wxString ProcessingMaxThreadNb = m_TextCtrlThreadsNb->GetValue();
    if (!ProcessingMaxThreadNb.IsNumber()) ProcessingMaxThreadNb = "2";
    pConfig->Write("/General/ProcessingMaxThreadNb", ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = (long)m_SliderThreadsPriority->GetValue();
    pConfig->Write("/General/ProcessingThreadsPriority", ProcessingThreadsPriority);
    
    // Processing
    long ProcessingMethod = (long)m_RadioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && ProcessingMethod==(long)asMULTITHREADS)
    {
        ProcessingMethod = (long)asINSERT;
    }
    pConfig->Write("/ProcessingOptions/ProcessingMethod", ProcessingMethod);
    long ProcessingLinAlgebra = (long)m_RadioBoxLinearAlgebra->GetSelection();
    pConfig->Write("/ProcessingOptions/ProcessingLinAlgebra", ProcessingLinAlgebra);


    GetParent()->Update();
    pConfig->Flush();
}

void asFramePreferencesCalibrator::OnChangeMultithreadingCheckBox( wxCommandEvent& event )
{
    if (event.GetInt()==0)
    {
        m_RadioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (m_RadioBoxProcessingMethods->GetSelection()==asMULTITHREADS)
        {
            m_RadioBoxProcessingMethods->SetSelection(asINSERT);
        }
    }
    else
    {
        m_RadioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferencesCalibrator::SaveAndClose( wxCommandEvent& event )
{
    SavePreferences();
    Close();
}

void asFramePreferencesCalibrator::ApplyChanges( wxCommandEvent& event )
{
    SavePreferences();
}
