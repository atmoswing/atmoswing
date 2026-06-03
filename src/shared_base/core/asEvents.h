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

#ifndef AS_EVENTS_H
#define AS_EVENTS_H

#include <wx/event.h>

// Custom wxEvent declarations used across the AtmoSwing apps.
// Definitions live in the corresponding .cpp files (search for `wxDEFINE_EVENT(asEVT_...`).
//
// Previously these were declared at the bottom of asIncludes.h *outside* the header guard,
// which was harmless (wxDECLARE_EVENT expands to a function declaration that can repeat
// across TUs) but unconventional. Moved here for clarity.

wxDECLARE_EVENT(asEVT_STATUS_STARTING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_RUNNING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_FAILED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_SUCCESS, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_DOWNLOADING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_DOWNLOADED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_LOADING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_LOADED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_SAVING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_SAVED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_PROCESSING, wxCommandEvent);
wxDECLARE_EVENT(asEVT_STATUS_PROCESSED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_CLEAR, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_NEW_ADDED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_RATIO_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_SELECT_FIRST, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_FORECAST_QUANTILE_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_STATION_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_ANALOG_DATE_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_LEAD_TIME_SELECTION_CHANGED, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_OPEN_WORKSPACE, wxCommandEvent);
wxDECLARE_EVENT(asEVT_ACTION_OPEN_BATCHFORECASTS, wxCommandEvent);

#endif  // AS_EVENTS_H
