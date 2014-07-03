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
 
#include "asPanelForecastingModel.h"

#include "asPanelsManagerForecastingModels.h"
#include "asFileForecastingModels.h"

asPanelForecastingModel::asPanelForecastingModel( wxWindow* parent )
:
asPanelForecastingModelVirtual( parent )
{
    m_ParentFrame = NULL;
    m_PanelsManager = NULL;

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
