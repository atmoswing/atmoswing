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
 * Portions Copyright 2023 Pascal Horton, Terranum.
 */

#include "asFrameStyledTextCtrl.h"

#include "wx/settings.h"

asFrameStyledTextCtrl::asFrameStyledTextCtrl(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos)
    : asFrameStyledTextCtrlVirtual(parent, id, title, pos) {}

void asFrameStyledTextCtrl::SetContent(const wxString& content) {
    _scintilla->SetText(content);
    SetLexerXml();
}

void asFrameStyledTextCtrl::SetLexerXml() {
    const wxColour colTag = *wxBLUE;
    const wxColour colAttr = *wxRED;

    _scintilla->SetLexer(wxSTC_LEX_XML);

    _scintilla->StyleClearAll();

    _scintilla->StyleSetForeground(wxSTC_H_TAG, colTag);
    _scintilla->StyleSetForeground(wxSTC_H_TAGUNKNOWN, colTag);
    _scintilla->StyleSetForeground(wxSTC_H_ATTRIBUTE, colAttr);
    _scintilla->StyleSetForeground(wxSTC_H_ATTRIBUTEUNKNOWN, colAttr);
    _scintilla->StyleSetBold(wxSTC_H_ATTRIBUTEUNKNOWN, true);
    _scintilla->StyleSetForeground(wxSTC_H_OTHER, colTag);
    _scintilla->StyleSetForeground(wxSTC_H_COMMENT, wxColour("GREY"));
    _scintilla->StyleSetForeground(wxSTC_H_ENTITY, colAttr);
    _scintilla->StyleSetBold(wxSTC_H_ENTITY, true);
    _scintilla->StyleSetForeground(wxSTC_H_TAGEND, colTag);
    _scintilla->StyleSetForeground(wxSTC_H_XMLSTART, colTag);
    _scintilla->StyleSetForeground(wxSTC_H_XMLEND, colTag);
    _scintilla->StyleSetForeground(wxSTC_H_CDATA, colAttr);
}