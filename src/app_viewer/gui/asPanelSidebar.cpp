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

#include "asPanelSidebar.h"

#include "asIncludes.h"

#include "asBitmaps.h"

asPanelSidebar::asPanelSidebar(wxWindow* parent, wxWindowID id, const wxPoint& pos, const wxSize& size, long style)
    : asPanelSidebarVirtual(parent, id, pos, size, style) {
    _bitmapCaret->SetBitmap(asBitmaps::Get(asBitmaps::ID_MISC::SHOWN, wxSize(16, 16)));
}

void asPanelSidebar::OnReducePanel(wxMouseEvent& event) {
    GetParent()->Freeze();

    if (_sizerMain->IsShown(_sizerContent)) {
        _sizerMain->Hide(_sizerContent, true);
        _bitmapCaret->SetBitmap(asBitmaps::Get(asBitmaps::ID_MISC::HIDDEN, wxSize(16, 16)));
    } else {
        _sizerMain->Show(_sizerContent, true);
        _bitmapCaret->SetBitmap(asBitmaps::Get(asBitmaps::ID_MISC::SHOWN, wxSize(16, 16)));
    }

    // Refresh elements
    _sizerMain->Layout();
    Layout();

    GetParent()->FitInside();

    GetParent()->Thaw();
}

void asPanelSidebar::ReducePanel() {
    if (_sizerMain->IsShown(_sizerContent)) {
        _sizerMain->Hide(_sizerContent, true);
        _bitmapCaret->SetBitmap(asBitmaps::Get(asBitmaps::ID_MISC::HIDDEN, wxSize(16, 16)));
    } else {
        _sizerMain->Show(_sizerContent, true);
        _bitmapCaret->SetBitmap(asBitmaps::Get(asBitmaps::ID_MISC::SHOWN, wxSize(16, 16)));
    }
}

void asPanelSidebar::OnPaint(wxCommandEvent& event) {
    event.Skip();
}
