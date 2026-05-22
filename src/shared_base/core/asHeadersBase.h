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

#ifndef AS_HEADERS_BASE_H
#define AS_HEADERS_BASE_H

// =====================================================================================
// asHeadersBase.h — minimum-dependency header for project HEADERS.
//
// PURPOSE
//   Reduce the cost of `#include "asIncludes.h"` propagating through ~200 project
//   headers. Use THIS header in .h files that only need the small set below.
//   .cpp files should continue to include "asIncludes.h" (the omnibus) for the
//   full wxWidgets / asUtils / asLog / asThreadsManager surface.
//
// WHAT'S PROVIDED
//   - wxObject / wxString (for class-base and member types in headers)
//   - wxASSERT / wxFAIL_MSG (for inline header methods)
//   - All project type aliases via asTypeDefs.h (a1f, a2f, vi, vd, vwxs, ...)
//   - Project enums via asGlobEnums.h (Order, TimeFormat, asSUCCESS, asNOT_FOUND, ...)
//   - std::vector, std::unique_ptr (for class members)
//
// WHAT'S NOT PROVIDED (use full asIncludes.h or include directly if needed)
//   - wxLog* / wxASSERT (only via #include "asIncludes.h" or <wx/log.h>+<wx/debug.h>)
//   - wxFileConfig
//   - asUtils.h (asFind, asStrF, asRound, ...)
//   - asLog.h, asTime.h, asThreadsManager.h
//   - Eigen Core math (only the aliases via asTypeDefs.h)
//
// WX OMNIBUS NOT PULLED IN
//   asGlobEnums.h only includes <wx/defs.h> (and only when USE_GUI=1), so including
//   asHeadersBase.h does NOT drag in wx/wx.h's ~40 transitive headers. If a consumer
//   header needs wxLog/wxFrame/etc., it must include the appropriate wx headers directly
//   (or fall back to "asIncludes.h" if it really needs the full surface).
// =====================================================================================

#include <memory>
#include <vector>

#include <wx/debug.h>   // wxASSERT, wxFAIL_MSG (commonly used in inline header methods)
#include <wx/object.h>  // wxObject base class
#include <wx/string.h>  // wxString

#include "asGlobEnums.h"
#include "asTypeDefs.h"

#endif  // AS_HEADERS_BASE_H
