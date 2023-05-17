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

#include "wx/settings.h"

#include "asFrameStyledTextCtrl.h"

/**
 * Constructor of the styled text control frame.
 *
 * @param parent The parent window.
 * @param id An identifier for the control. wxID_ANY is taken as a default value.
 * @param title The title of the frame.
 * @param pos The position of the frame.
 */
asFrameStyledTextCtrl::asFrameStyledTextCtrl(wxWindow* parent, wxWindowID id, const wxString& title, const wxPoint& pos)
    : asFrameStyledTextCtrlVirtual(parent, id, title, pos) {}

/**
 * Set the content of the styled text control.
 *
 * @param content The string to set.
 */
void asFrameStyledTextCtrl::SetContent(const wxString& content) {
    m_scintilla->SetText(content);
    SetLexerXml();
}

/**
 * Set the lexer to XML fo styling.
 *
 * @note This is a copy of the function from the wxWidgets sample
 *       (https://github.com/wxWidgets/wxWidgets/blob/master/samples/stc/stctest.cpp).
 */
void asFrameStyledTextCtrl::SetLexerXml()
{
    const wxColour colTag = *wxBLUE;
    const wxColour colAttr = *wxRED;

    m_scintilla->SetLexer(wxSTC_LEX_XML);

    m_scintilla->StyleClearAll();

    m_scintilla->StyleSetForeground(wxSTC_H_TAG, colTag);
    m_scintilla->StyleSetForeground(wxSTC_H_TAGUNKNOWN, colTag);
    m_scintilla->StyleSetForeground(wxSTC_H_ATTRIBUTE, colAttr);
    m_scintilla->StyleSetForeground(wxSTC_H_ATTRIBUTEUNKNOWN, colAttr);
    m_scintilla->StyleSetBold(wxSTC_H_ATTRIBUTEUNKNOWN, true);
    m_scintilla->StyleSetForeground(wxSTC_H_OTHER, colTag);
    m_scintilla->StyleSetForeground(wxSTC_H_COMMENT, wxColour("GREY"));
    m_scintilla->StyleSetForeground(wxSTC_H_ENTITY, colAttr);
    m_scintilla->StyleSetBold(wxSTC_H_ENTITY, true);
    m_scintilla->StyleSetForeground(wxSTC_H_TAGEND, colTag);
    m_scintilla->StyleSetForeground(wxSTC_H_XMLSTART, colTag);
    m_scintilla->StyleSetForeground(wxSTC_H_XMLEND, colTag);
    m_scintilla->StyleSetForeground(wxSTC_H_CDATA, colAttr);
}