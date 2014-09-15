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

#include "asFramePreferencesForecaster.h"

#include "wx/fileconf.h"
#include "wx/thread.h"

asFramePreferencesForecaster::asFramePreferencesForecaster( wxWindow* parent, asBatchForecasts* batchForecasts, wxWindowID id )
:
asFramePreferencesForecasterVirtual( parent, id )
{
    m_BatchForecasts = batchForecasts;
    LoadPreferences();
    Fit();

        // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif
}

void asFramePreferencesForecaster::CloseFrame( wxCommandEvent& event )
{
    Close();
}

void asFramePreferencesForecaster::Update()
{
    LoadPreferences();
}

void asFramePreferencesForecaster::LoadPreferences()
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();

    // Fix the color of the file/dir pickers
    wxColour col = m_NotebookBase->GetThemeBackgroundColour();
    if (col.IsOk())
    {
        m_DirPickerPredictandDB->SetBackgroundColour(col);
        m_DirPickerForecastResults->SetBackgroundColour(col);
        m_DirPickerParameters->SetBackgroundColour(col);
        m_DirPickerArchivePredictors->SetBackgroundColour(col);
        m_DirPickerRealtimePredictorSaving->SetBackgroundColour(col);
    }
    
    /*
     * Batch file properties
     */

    // Paths
    m_DirPickerPredictandDB->SetPath(m_BatchForecasts->GetPredictandDBDirectory());
    m_DirPickerForecastResults->SetPath(m_BatchForecasts->GetForecastsOutputDirectory());
    m_DirPickerRealtimePredictorSaving->SetPath(m_BatchForecasts->GetPredictorsRealtimeDirectory());
    m_DirPickerArchivePredictors->SetPath(m_BatchForecasts->GetPredictorsArchiveDirectory());
    m_DirPickerParameters->SetPath(m_BatchForecasts->GetParametersFileDirectory());

    /*
     * General
     */
    
    // Log
    long defaultLogLevelForecaster = 1; // = selection +1
    long logLevelForecaster = pConfig->Read("/General/LogLevel", defaultLogLevelForecaster);
    m_RadioBoxLogLevel->SetSelection((int)logLevelForecaster-1);
    bool displayLogWindowForecaster;
    pConfig->Read("/General/DisplayLogWindow", &displayLogWindowForecaster, false);
    m_CheckBoxDisplayLogWindow->SetValue(displayLogWindowForecaster);

    // Proxy
    bool checkBoxProxy;
    pConfig->Read("/Internet/UsesProxy", &checkBoxProxy, false);
    m_CheckBoxProxy->SetValue(checkBoxProxy);
    wxString ProxyAddress = pConfig->Read("/Internet/ProxyAddress", wxEmptyString);
    m_TextCtrlProxyAddress->SetValue(ProxyAddress);
    wxString ProxyPort = pConfig->Read("/Internet/ProxyPort", wxEmptyString);
    m_TextCtrlProxyPort->SetValue(ProxyPort);
    wxString ProxyUser = pConfig->Read("/Internet/ProxyUser", wxEmptyString);
    m_TextCtrlProxyUser->SetValue(ProxyUser);
    wxString ProxyPasswd = pConfig->Read("/Internet/ProxyPasswd", wxEmptyString);
    m_TextCtrlProxyPasswd->SetValue(ProxyPasswd);

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

    // Downloads
    int maxPrevStepsNb = 5;
    wxString maxPrevStepsNbStr = wxString::Format("%d", maxPrevStepsNb);
    wxString InternetMaxPrevStepsNb = pConfig->Read("/Internet/MaxPreviousStepsNb", maxPrevStepsNbStr);
    m_TextCtrlMaxPrevStepsNb->SetValue(InternetMaxPrevStepsNb);
    int maxParallelRequests = 5;
    wxString maxParallelRequestsStr = wxString::Format("%d", maxParallelRequests);
    wxString InternetParallelRequestsNb = pConfig->Read("/Internet/ParallelRequestsNb", maxParallelRequestsStr);
    m_TextCtrlMaxRequestsNb->SetValue(InternetParallelRequestsNb);
    bool restrictDownloads;
    pConfig->Read("/Internet/RestrictDownloads", &restrictDownloads, true);
    m_CheckBoxRestrictDownloads->SetValue(restrictDownloads);

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
    bool multiForecaster;
    pConfig->Read("/General/MultiInstances", &multiForecaster, false);
    m_CheckBoxMultiInstancesForecaster->SetValue(multiForecaster);

    // Multithreading
    bool allowMultithreading;
    pConfig->Read("/Processing/AllowMultithreading", &allowMultithreading, true);
    m_CheckBoxAllowMultithreading->SetValue(allowMultithreading);
    int maxThreads = wxThread::GetCPUCount();
    if (maxThreads==-1) maxThreads = 2;
    wxString maxThreadsStr = wxString::Format("%d", maxThreads);
    wxString ProcessingMaxThreadNb = pConfig->Read("/Processing/MaxThreadNb", maxThreadsStr);
    m_TextCtrlThreadsNb->SetValue(ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = pConfig->Read("/Processing/ThreadsPriority", 95l);
    m_SliderThreadsPriority->SetValue((int)ProcessingThreadsPriority);

    // Processing
    long defaultMethod = (long)asMULTITHREADS;
    long ProcessingMethod = pConfig->Read("/Processing/Method", defaultMethod);
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
    long ProcessingLinAlgebra = pConfig->Read("/Processing/LinAlgebra", defaultLinAlgebra);
    m_RadioBoxLinearAlgebra->SetSelection((int)ProcessingLinAlgebra);

    // User directories
    wxString userpath = asConfig::GetUserDataDir();
    m_StaticTextUserDir->SetLabel(userpath);
    wxString logpathForecaster = asConfig::GetLogDir();
    logpathForecaster.Append("AtmoSwingForecaster.log");
    m_StaticTextLogFile->SetLabel(logpathForecaster);
    m_StaticTextPrefFile->SetLabel(asConfig::GetUserDataDir()+"AtmoSwingForecaster.ini");

}

void asFramePreferencesForecaster::SavePreferences( )
{
    wxConfigBase *pConfig;
    pConfig = wxFileConfig::Get();
    
    /*
     * Batch file properties
     */

    // Paths
    m_BatchForecasts->SetPredictandDBDirectory(m_DirPickerPredictandDB->GetPath());
    m_BatchForecasts->SetForecastsOutputDirectory(m_DirPickerForecastResults->GetPath());
    m_BatchForecasts->SetPredictorsRealtimeDirectory(m_DirPickerRealtimePredictorSaving->GetPath());
    m_BatchForecasts->SetPredictorsArchiveDirectory(m_DirPickerArchivePredictors->GetPath());
    m_BatchForecasts->SetParametersFileDirectory(m_DirPickerParameters->GetPath());

    /*
     * General
     */

    // Log
    long logLevelForecaster = (long)m_RadioBoxLogLevel->GetSelection();
    pConfig->Write("/General/LogLevel", logLevelForecaster+1); // = selection +1
    bool displayLogWindowForecaster = m_CheckBoxDisplayLogWindow->GetValue();
    pConfig->Write("/General/DisplayLogWindow", displayLogWindowForecaster);

    // Proxy
    bool checkBoxProxy = m_CheckBoxProxy->GetValue();
    pConfig->Write("/Internet/UsesProxy", checkBoxProxy);
    wxString ProxyAddress = m_TextCtrlProxyAddress->GetValue();
    pConfig->Write("/Internet/ProxyAddress", ProxyAddress);
    wxString ProxyPort = m_TextCtrlProxyPort->GetValue();
    pConfig->Write("/Internet/ProxyPort", ProxyPort);
    wxString ProxyUser = m_TextCtrlProxyUser->GetValue();
    pConfig->Write("/Internet/ProxyUser", ProxyUser);
    wxString ProxyPasswd = m_TextCtrlProxyPasswd->GetValue();
    pConfig->Write("/Internet/ProxyPasswd", ProxyPasswd);

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

    // Downloads
    wxString InternetMaxPrevStepsNb = m_TextCtrlMaxPrevStepsNb->GetValue();
    if (!InternetMaxPrevStepsNb.IsNumber()) InternetMaxPrevStepsNb = "5";
    pConfig->Write("/Internet/MaxPreviousStepsNb", InternetMaxPrevStepsNb);
    wxString InternetParallelRequestsNb = m_TextCtrlMaxRequestsNb->GetValue();
    if (!InternetParallelRequestsNb.IsNumber()) InternetParallelRequestsNb = "5";
    pConfig->Write("/Internet/ParallelRequestsNb", InternetParallelRequestsNb);
    bool restrictDownloads = m_CheckBoxRestrictDownloads->GetValue();
    pConfig->Write("/Internet/RestrictDownloads", restrictDownloads);

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

    bool multiForecaster = m_CheckBoxMultiInstancesForecaster->GetValue();
    pConfig->Write("/General/MultiInstances", multiForecaster);

    // Multithreading
    bool allowMultithreading = m_CheckBoxAllowMultithreading->GetValue();
    pConfig->Write("/Processing/AllowMultithreading", allowMultithreading);
    wxString ProcessingMaxThreadNb = m_TextCtrlThreadsNb->GetValue();
    if (!ProcessingMaxThreadNb.IsNumber()) ProcessingMaxThreadNb = "2";
    pConfig->Write("/Processing/MaxThreadNb", ProcessingMaxThreadNb);
    long ProcessingThreadsPriority = (long)m_SliderThreadsPriority->GetValue();
    pConfig->Write("/Processing/ThreadsPriority", ProcessingThreadsPriority);

    // Processing
    long ProcessingMethod = (long)m_RadioBoxProcessingMethods->GetSelection();
    if (!allowMultithreading && ProcessingMethod==(long)asMULTITHREADS)
    {
        ProcessingMethod = (long)asINSERT;
    }
    pConfig->Write("/Processing/Method", ProcessingMethod);
    long ProcessingLinAlgebra = (long)m_RadioBoxLinearAlgebra->GetSelection();
    pConfig->Write("/Processing/LinAlgebra", ProcessingLinAlgebra);
    
    if(GetParent()!=NULL)
    {
        GetParent()->Update();
    }

    pConfig->Flush();
    m_BatchForecasts->Save();
}

void asFramePreferencesForecaster::OnChangeMultithreadingCheckBox( wxCommandEvent& event )
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

void asFramePreferencesForecaster::SaveAndClose( wxCommandEvent& event )
{
    SavePreferences();
    Close();
}

void asFramePreferencesForecaster::ApplyChanges( wxCommandEvent& event )
{
    SavePreferences();
}
