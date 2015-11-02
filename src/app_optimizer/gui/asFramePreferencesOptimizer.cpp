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

#include "asFramePreferencesOptimizer.h"

#include "wx/fileconf.h"
#include "wx/thread.h"

asFramePreferencesOptimizer::asFramePreferencesOptimizer( wxWindow* parent, wxWindowID id )
:
asFramePreferencesOptimizerVirtual( parent, id )
{
    LoadPreferences();
    Fit();

        // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesOptimizer::CloseFrame( wxCommandEvent& event )
{
    Close();
}

void asFramePreferencesOptimizer::Update()
{
    LoadPreferences();
}

void asFramePreferencesOptimizer::LoadPreferences()
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = m_notebookBase->GetThemeBackgroundColour();
    if (col.IsOk())
    {
        m_dirPickerPredictandDB->SetBackgroundColour(col);
        m_dirPickerIntermediateResults->SetBackgroundColour(col);
        m_dirPickerArchivePredictors->SetBackgroundColour(col);
    }

    /*
     * General
     */

    // Log
    long defaultLogLevel = 1; // = selection +1
    long logLevel = pConfig->Read("/General/LogLevel", defaultLogLevel);
    m_radioBoxLogLevel->SetSelection((int)logLevel-1);
    bool displayLogWindow;
    pConfig->Read("/General/DisplayLogWindow", &displayLogWindow, false);
    m_checkBoxDisplayLogWindow->SetValue(displayLogWindow);

    // Paths
    wxString dirConfig = asConfig::GetDataDir()+"config"+DS;
    wxString dirData = asConfig::GetDataDir()+"data"+DS;
    wxString PredictandDBDir = pConfig->Read("/Paths/DataPredictandDBDir", dirData+"predictands");
    m_dirPickerPredictandDB->SetPath(PredictandDBDir);
    wxString ArchivePredictorsDir = pConfig->Read("/Paths/ArchivePredictorsDir", dirData+"predictors");
    m_dirPickerArchivePredictors->SetPath(ArchivePredictorsDir);

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = pConfig->Read("/General/GuiOptions", 1l);
    m_radioBoxGui->SetSelection((int)guiOptions);
    if (guiOptions==0)
    {
        g_silentMode = true;
    }
    else
    {
        g_silentMode = false;
        g_verboseMode = false;
        if (guiOptions==2l)
        {
            g_verboseMode = true;
        }
    }

    // Advanced options
    bool responsive;
    pConfig->Read("/General/Responsive", &responsive, true);
    m_checkBoxResponsiveness->SetValue(responsive);
    if (responsive)
    {
        g_responsive = true;
    }
    else
    {
        g_responsive = false;
    }

    // Multithreading
    bool allowMultithreading;
    pConfig->Read("/Processing/AllowMultithreading", &allowMultithreading, true);
    m_checkBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads==-1) maxThreads = 2;
    wxString maxThreadsStr = wxString::Format("%d", maxThreads);
    wxString ProcessingMaxThreadNb = pConfig->Read("/Processing/MaxThreadNb", maxThreadsStr);
    m_textCtrlThreadsNb->SetValue(ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = pConfig->Read("/Processing/ThreadsPriority", 95l);
    m_sliderThreadsPriority->SetValue((int)ProcessingThreadsPriority);

    // Processing
    long defaultMethod = (long)asMULTITHREADS;
    long ProcessingMethod = pConfig->Read("/Processing/Method", defaultMethod);
    if (!allowMultithreading)
    {
        m_radioBoxProcessingMethods->Enable(0, false);
        if (ProcessingMethod==(long)asMULTITHREADS)
        {
            ProcessingMethod = (long)asINSERT;
        }
    }
    else
    {
        m_radioBoxProcessingMethods->Enable(0, true);
    }
    m_radioBoxProcessingMethods->SetSelection((int)ProcessingMethod);
    long defaultLinAlgebra = (long)asLIN_ALGEBRA_NOVAR;
    long ProcessingLinAlgebra = pConfig->Read("/Processing/LinAlgebra", defaultLinAlgebra);
    m_radioBoxLinearAlgebra->SetSelection((int)ProcessingLinAlgebra);

    // User directories
    wxString IntermediateResultsDir = pConfig->Read("/Paths/IntermediateResultsDir", asConfig::GetTempDir()+"AtmoSwing");
    m_dirPickerIntermediateResults->SetPath(IntermediateResultsDir);
    wxString userpath = asConfig::GetUserDataDir();
    m_staticTextUserDir->SetLabel(userpath);
    wxString logpath = asConfig::GetLogDir();
    logpath.Append("AtmoSwingOptimizer.log");
    m_staticTextLogFile->SetLabel(logpath);
    m_staticTextPrefFile->SetLabel(asConfig::GetUserDataDir()+"AtmoSwingOptimizer.ini");
}

void asFramePreferencesOptimizer::SavePreferences( )
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    /*
     * General
     */

    // Log    
    long logLevel = (long)m_radioBoxLogLevel->GetSelection();
    pConfig->Write("/General/LogLevel", logLevel+1); // = selection +1
    bool displayLogWindow = m_checkBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindow);

    // Paths
    wxString PredictandDBDir = m_dirPickerPredictandDB->GetPath();
    pConfig->Write("/Paths/DataPredictandDBDir", PredictandDBDir);
    wxString IntermediateResultsDir = m_dirPickerIntermediateResults->GetPath();
    pConfig->Write("/Paths/IntermediateResultsDir", IntermediateResultsDir);
    wxString ArchivePredictorsDir = m_dirPickerArchivePredictors->GetPath();
    pConfig->Write("/Paths/ArchivePredictorsDir", ArchivePredictorsDir);

    /*
     * Advanced
     */

    // GUI options
    long guiOptions = (long)m_radioBoxGui->GetSelection();
    pConfig->Write("/General/GuiOptions", guiOptions);
    if (guiOptions==0)
    {
        g_silentMode = true;
    }
    else
    {
        g_silentMode = false;
        g_verboseMode = false;
        if (guiOptions==2l)
        {
            g_verboseMode = true;
        }
    }

    // Advanced options
    bool responsive = m_checkBoxResponsiveness->GetValue();
    pConfig->Write("/General/Responsive", responsive);
    if (responsive)
    {
        g_responsive = true;
    }
    else
    {
        g_responsive = false;
    }

    // Multithreading
    bool allowMultithreading = m_checkBoxAllowMultithreading->GetValue();
    pConfig->Write("/Processing/AllowMultithreading", allowMultithreading);
    wxString ProcessingMaxThreadNb = m_textCtrlThreadsNb->GetValue();
    if (!ProcessingMaxThreadNb.IsNumber()) ProcessingMaxThreadNb = "2";
    pConfig->Write("/Processing/MaxThreadNb", ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = (long)m_sliderThreadsPriority->GetValue();
    pConfig->Write("/Processing/ThreadsPriority", ProcessingThreadsPriority);
    
    // Processing
    long ProcessingMethod = (long)m_radioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && ProcessingMethod==(long)asMULTITHREADS)
    {
        ProcessingMethod = (long)asINSERT;
    }
    pConfig->Write("/Processing/Method", ProcessingMethod);
    long ProcessingLinAlgebra = (long)m_radioBoxLinearAlgebra->GetSelection();
    pConfig->Write("/Processing/LinAlgebra", ProcessingLinAlgebra);


    GetParent()->Update();
    pConfig->Flush();
}

void asFramePreferencesOptimizer::OnChangeMultithreadingCheckBox( wxCommandEvent& event )
{
    if (event.GetInt()==0)
    {
        m_radioBoxProcessingMethods->Enable(asMULTITHREADS, false);
        if (m_radioBoxProcessingMethods->GetSelection()==asMULTITHREADS)
        {
            m_radioBoxProcessingMethods->SetSelection(asINSERT);
        }
    }
    else
    {
        m_radioBoxProcessingMethods->Enable(asMULTITHREADS, true);
    }
}

void asFramePreferencesOptimizer::SaveAndClose( wxCommandEvent& event )
{
    SavePreferences();
    Close();
}

void asFramePreferencesOptimizer::ApplyChanges( wxCommandEvent& event )
{
    SavePreferences();
}
