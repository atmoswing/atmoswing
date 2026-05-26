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
//   Project .h files include THIS header (not "asIncludes.h"), which keeps the wx/wx.h
//   omnibus from cascading through ~200 project headers and inflating compile times.
//   .cpp files include "asIncludes.h" for the full wxWidgets / asUtils / asLog /
//   asThreadsManager / Eigen surface — paid once per translation unit, not propagated.
//
// WHAT'S PROVIDED
//   - wxObject / wxString (for class-base and member types in headers)
//   - wxASSERT / wxFAIL_MSG (for inline header methods)
//   - _() translation macro (used in inline error messages)
//   - All project type aliases via asTypeDefs.h (a1f, a2f, vi, vd, vwxs, ...)
//   - Project enums via asGlobEnums.h (Order, TimeFormat, asSUCCESS, asNOT_FOUND, ...)
//   - std::vector, std::unique_ptr (for class members)
// =====================================================================================

#include <memory>
#include <stdexcept>
#include <vector>

#include <wx/debug.h>   // wxASSERT, wxFAIL_MSG (commonly used in inline header methods)
#include <wx/intl.h>    // _() translation macro (used in inline error messages)
#include <wx/object.h>  // wxObject base class
#include <wx/string.h>  // wxString

#include "asGlobEnums.h"
#include "asTypeDefs.h"

#endif  // AS_HEADERS_BASE_H
