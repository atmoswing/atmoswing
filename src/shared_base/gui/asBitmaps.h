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

#ifndef AS_BITMAPS_H
#define AS_BITMAPS_H

#include "wx/wxprec.h"
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#include "asIncludes.h"

class asBitmaps {
  public:
    static wxString SvgToolbar[];
    static wxString SvgBullets[];
    static wxString SvgMisc[];
    static wxString SvgTreeCtrl[];
    static wxString SvgLogo;
    static wxString ColorBlack;
    static wxString ColorWhite;

    enum ID_TOOLBAR {
        FRAME_ANALOGS = 0,
        FRAME_DISTRIBUTIONS,
        FRAME_PREDICTORS,
        MAP_FIT,
        MAP_MOVE,
        MAP_SELECT,
        MAP_CROSS,
        MAP_ZOOM_IN,
        MAP_ZOOM_OUT,
        OPEN,
        PREFERENCES,
        RUN,
        STOP
    };

    enum ID_BULLETS {
        BULLET_GREEN = 0,
        BULLET_RED,
        BULLET_WHITE,
        BULLET_YELLOW
    };

    enum ID_MISC {
        CLOSE = 0,
        HIDDEN,
        PLUS,
        SHOWN,
        UPDATE,
        ARROW_LEFT,
        ARROW_RIGHT
    };

    enum ID_TREECTRL {
        ICON_PRECIP = 0,
        ICON_TEMP,
        ICON_WIND,
        ICON_LIGHTNING,
        ICON_OTHER
    };

    static wxString GetColor();

    static wxBitmap Get(asBitmaps::ID_TOOLBAR id, const wxSize& size = wxSize(32, 32));

    static wxBitmap Get(asBitmaps::ID_BULLETS id, const wxSize& size = wxSize(16, 16));

    static wxBitmap Get(asBitmaps::ID_MISC id, const wxSize& size = wxSize(16, 16));

    static wxBitmap Get(asBitmaps::ID_TREECTRL id, const wxSize& size = wxSize(16, 16));

    static wxBitmap GetLogo(const wxSize& size = wxSize(128, 128));
};

#endif
