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

#ifndef AS_FRAME_STYLED_TEXT_CTRL_H
#define AS_FRAME_STYLED_TEXT_CTRL_H

#include "asGlobEnums.h"
#include "AtmoSwingForecasterGui.h"

class asFrameStyledTextCtrl : public asFrameStyledTextCtrlVirtual {
  public:
    /**
     * Constructor of the styled text control frame.
     *
     * @param parent The parent window.
     * @param id An identifier for the control. wxID_ANY is taken as a default value.
     * @param title The title of the frame.
     * @param pos The position of the frame.
     */
    explicit asFrameStyledTextCtrl(wxWindow* parent, wxWindowID id = asWINDOW_PARAMETERS_DETAILS,
                                   const wxString& title = wxEmptyString, const wxPoint& pos = wxDefaultPosition);

    /**
     * Destructor of the styled text control frame.
     */
    ~asFrameStyledTextCtrl() override = default;

    /**
     * Set the content of the styled text control.
     *
     * @param content The string to set.
     */
    void SetContent(const wxString& content);

    /**
     * Set the lexer to XML fo styling.
     *
     * @note This is a copy of the function from the wxWidgets sample
     *       (https://github.com/wxWidgets/wxWidgets/blob/master/samples/stc/stctest.cpp).
     */
    void SetLexerXml();

    /**
     * Get the scintilla control.
     *
     * @return The scintilla control.
     */
    wxStyledTextCtrl* GetScintilla() {
        return m_scintilla;
    }
};

#endif
