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
 
#include "asPanelForecast.h"

#include "asPanelsManagerForecasts.h"

asPanelForecast::asPanelForecast( wxWindow* parent )
:
asPanelForecastVirtual( parent )
{
    m_parentFrame = NULL;
    m_panelsManager = NULL;

    // Led
    m_led = new awxLed( this, wxID_ANY, wxDefaultPosition, wxDefaultSize, awxLED_RED, 0 );
	m_led->SetState( awxLED_OFF );
	m_sizerHeader->Insert( 0, m_led, 0, wxALL|wxALIGN_CENTER_VERTICAL, 5 );

    // Set the buttons bitmaps
    m_bpButtonClose->SetBitmapLabel(*_img_close);

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

void asPanelForecast::ClosePanel( wxCommandEvent& event )
{
    m_panelsManager->RemovePanel(this);
}

bool asPanelForecast::Layout()
{
    asPanelForecastVirtual::Layout();
    return true;
}

void asPanelForecast::ChangeForecastName( wxCommandEvent& event )
{
    //
}
