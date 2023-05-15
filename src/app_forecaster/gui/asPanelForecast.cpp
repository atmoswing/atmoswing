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

#include "asPanelForecast.h"

#include "asBatchForecasts.h"
#include "asFrameStyledTextCtrl.h"
#include "asPanelsManagerForecasts.h"
#include "asParametersForecast.h"

asPanelForecast::asPanelForecast(wxWindow* parent, asBatchForecasts* batch)
    : asPanelForecastVirtual(parent),
      m_parentFrame(nullptr),
      m_panelsManager(nullptr),
      m_batchForecasts(batch) {
    // Led
    m_led = new awxLed(this, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_RED, 0);
    m_led->SetState(awxLED_OFF);
    m_sizerHeader->Insert(0, m_led, 0, wxALL | wxALIGN_CENTER_VERTICAL, 5);

    // Set the buttons bitmaps
    m_bpButtonClose->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::CLOSE));
    m_bpButtonInfo->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::INFO));
    m_bpButtonDetails->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::DETAILS));
    m_bpButtonEdit->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::EDIT));
    m_bpButtonWarning->SetBitmapLabel(asBitmaps::Get(asBitmaps::ID_MISC::WARNING));
    m_bpButtonWarning->Hide();

    // Fix the color of the file/dir pickers
    wxColour col = parent->GetParent()->GetBackgroundColour();
    if (col.IsOk()) {
        SetBackgroundColour(col);
    }
}

void asPanelForecast::CheckFileExists() {
    wxASSERT(m_batchForecasts);
    wxString fileName = GetParametersFileName();
    wxString dirPath = m_batchForecasts->GetParametersFileDirectory();
    if (wxFileExists(dirPath + DS + fileName)) {
        m_bpButtonWarning->Hide();
        SetTooTipContent(dirPath + DS + fileName);
    } else {
        m_bpButtonWarning->Show();
        Layout();
        Refresh();
        m_bpButtonInfo->SetToolTip(wxEmptyString);
    }
}

void asPanelForecast::SetTooTipContent(const wxString &filePath) {
    asParametersForecast param;
    if (param.LoadFromFile(filePath)) {
        wxString description = param.GetDescription();
        m_bpButtonInfo->SetToolTip(description);
    }
}

void asPanelForecast::ClosePanel(wxCommandEvent& event) {
    m_panelsManager->RemovePanel(this);
}

void asPanelForecast::OnEditForecastFile(wxCommandEvent& event) {
    auto button = dynamic_cast<wxWindow*>(event.GetEventObject());
    wxASSERT(button);
    auto panel = dynamic_cast<asPanelForecast*>(button->GetParent());
    wxASSERT(panel);
    wxString value = panel->GetParametersFileName();

    wxTextEntryDialog dialog(this, _("Enter the file name (without the path)"), _("Parameters file name"), value);

    if (dialog.ShowModal() == wxID_CANCEL) return;

    wxString filename = dialog.GetValue();
    panel->SetParametersFileName(filename);
}

void asPanelForecast::OnDetailsForecastFile(wxCommandEvent& event) {
    wxASSERT(m_batchForecasts);
    wxString fileName = GetParametersFileName();
    wxString dirPath = m_batchForecasts->GetParametersFileDirectory();
    if (wxFileExists(dirPath + DS + fileName)) {
        asFileText file(dirPath + DS + fileName);
        file.Open();
        wxString fileContent = file.GetContent();
        asFrameStyledTextCtrl* frameText = new asFrameStyledTextCtrl(this);
        frameText->SetTitle(fileName);
        frameText->SetContent(fileContent);
        frameText->Show();
    }
}

bool asPanelForecast::Layout() {
    asPanelForecastVirtual::Layout();
    return true;
}
