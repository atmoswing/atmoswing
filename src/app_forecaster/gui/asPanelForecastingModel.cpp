/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asPanelForecastingModel.h"

#include "asPanelsManagerForecastingModels.h"
#include "asFileForecastingModels.h"

asPanelForecastingModel::asPanelForecastingModel( wxWindow* parent )
:
asPanelForecastingModelVirtual( parent )
{
    // Set the buttons bitmaps
    m_BpButtonClose->SetBitmapLabel(img_close);
    m_BpButtonReduce->SetBitmapLabel(img_shown);

    m_Reduced = false;

        // Fix the color of the file/dir pickers
    wxColour col = parent->GetParent()->GetBackgroundColour();
    if (col.IsOk())
    {
        SetBackgroundColour(col);
    }

    #if defined(__WXMSW__)
        SetWindowStyleFlag(wxRAISED_BORDER);
    #elif defined(__WXMAC__)
        SetWindowStyleFlag(wxRAISED_BORDER);
    #elif defined(__UNIX__)
        SetWindowStyleFlag(wxSIMPLE_BORDER);
    #else
        SetWindowStyleFlag(wxRAISED_BORDER);
    #endif
}

void asPanelForecastingModel::ReducePanel( wxCommandEvent& event )
{
    wxWindow* topFrame = m_PanelsManager->GetTopFrame(this);
    topFrame->Freeze();

    if(m_Reduced)
    {
        m_Reduced = false;
        m_BpButtonReduce->SetBitmapLabel(img_shown);
        m_SizerPanel->Show(m_SizerFields, true);
    } else {
        m_Reduced = true;
        m_BpButtonReduce->SetBitmapLabel(img_hidden);
        m_SizerPanel->Hide(m_SizerFields, true);
    }

    // Refresh elements
    m_SizerPanel->Layout();
    Layout();
    GetSizer()->Fit(GetParent());
    topFrame->Layout();
    topFrame->Refresh();

    topFrame->Thaw();
}

void asPanelForecastingModel::ReducePanel()
{
    m_Reduced = true;
    m_BpButtonReduce->SetBitmapLabel(img_hidden);
    m_SizerPanel->Hide(m_SizerFields, true);

    m_PanelsManager->LayoutFrame(this);
}

void asPanelForecastingModel::ClosePanel( wxCommandEvent& event )
{
    m_PanelsManager->RemovePanel(this);
}

bool asPanelForecastingModel::Layout()
{
    asPanelForecastingModelVirtual::Layout();
    return true;
}

void asPanelForecastingModel::ChangeModelName( wxCommandEvent& event )
{
    m_StaticTextModelName->SetLabel(m_TextCtrlModelName->GetValue());
}

bool asPanelForecastingModel::GenerateXML( asFileForecastingModels &file )
{
    if(!file.InsertElementAndAttribute(wxEmptyString, "Model", "name", m_TextCtrlModelName->GetValue())) return false;

    if(!file.GoToLastNodeWithPath("Model")) return false;

    if(!file.SetElementAttribute(wxEmptyString, "description", m_TextCtrlModelDescription->GetValue())) return false;

    if(!file.InsertElementAndAttribute(wxEmptyString, _T("ParametersFileName"), _T("value"), m_TextCtrlParametersFileName->GetValue())) return false;
    if(!file.InsertElementAndAttribute(wxEmptyString, _T("PredictandDB"), _T("value"), m_TextCtrlPredictandDB->GetValue())) return false;
    if(!file.InsertElementAndAttribute(wxEmptyString, _T("PredictorsArchiveDir"), _T("value"), m_DirPickerPredictorsArchive->GetPath())) return false;

    if(!file.GoANodeBack()) return false;

    return true;
}
