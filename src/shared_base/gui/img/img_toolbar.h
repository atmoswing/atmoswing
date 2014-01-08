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

#ifndef _TOOLBAR_H_
#define _TOOLBAR_H_ 1

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#    include <wx/wx.h>
#endif

extern void initialize_images_toolbar(void);

extern wxBitmap img_database_options;
extern wxBitmap img_database_run;
extern wxBitmap img_frame_dots;
extern wxBitmap img_frame_forecaster;
extern wxBitmap img_frame_grid;
extern wxBitmap img_frame_plots;
extern wxBitmap img_frame_viewer;
extern wxBitmap img_map_cross;
extern wxBitmap img_map_cursor;
extern wxBitmap img_map_fit;
extern wxBitmap img_map_move;
extern wxBitmap img_map_zoom_in;
extern wxBitmap img_map_zoom_out;
extern wxBitmap img_open;
extern wxBitmap img_preferences;
extern wxBitmap img_print;
extern wxBitmap img_run_history;
extern wxBitmap img_run;
extern wxBitmap img_run_cancel;

#endif /* _TOOLBAR_H_ */
