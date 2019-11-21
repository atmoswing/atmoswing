/***************************************************************************
                lsversion_core.h

                             -------------------
    copyright            : (C) 2010 CREALP Lucien Schreiber
    email                : lucien.schreiber at crealp dot vs dot ch
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef _LSVERSION_CORE_H
#define _LSVERSION_CORE_H

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
// Include wxWidgets' headers
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

#ifdef USE_LSVERSION
#include "lsversion_param.h"
#endif

class lsVersion {
   public:
    static wxString GetSoftName();

    static wxString GetSoftGIT();

    static wxString GetSoftGITBranch();

    static wxString GetSoftGITRevisionHash();

    static wxString GetSoftGITRevisionNb();

    static wxString GetwxWidgetsNumber();

    static wxString GetGDALNumber();

    static wxString GetGEOSNumber();

    static wxString GetCurlNumber();

    static wxString GetSQLiteNumber();

    static wxString GetMySQLNumber();

    static wxString GetNetCDFNumber();

    static wxString GetProjNumber();

    static wxString GetEigenNumber();

    static wxString GetPNGNumber();

    static wxString GetJpegNumber();

    static wxString GetJasperNumber();

    static wxString GetEcCodesNumber();

    static wxString GetAllModules();
};

#endif
