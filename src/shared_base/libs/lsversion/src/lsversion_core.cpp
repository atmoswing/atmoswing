/***************************************************************************
                lsversion_core.cpp

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

#include "lsversion_core.h"

wxString lsVersion::GetSoftName() {
    wxString myName = wxEmptyString;
#ifdef lsVERSION_SOFT_NAME
    myName = lsVERSION_SOFT_NAME;
#endif
    return myName;
}

wxString lsVersion::GetSoftGIT() {
    wxString myGITVersion = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION
    myGITVersion = lsVERSION_SOFT_VERSION;
#endif
    return myGITVersion;
}

wxString lsVersion::GetSoftGITBranch() {
    wxString myGITtxt = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION_BRANCH
    myGITtxt = lsVERSION_SOFT_VERSION_BRANCH;
#endif
    return myGITtxt;
}

wxString lsVersion::GetSoftGITRevisionHash() {
    wxString myGITtxt = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION_REVISION
    myGITtxt = lsVERSION_SOFT_VERSION_REVISION;
#endif
    return myGITtxt;
}

wxString lsVersion::GetSoftGITRevisionNb() {
    wxString myGITtxt = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION
    myGITtxt = lsVERSION_SOFT_VERSION;
#endif
    return myGITtxt;
}

wxString lsVersion::GetwxWidgetsNumber() {
    wxString mywxVersion = wxString::Format("%d.%d.%d", wxMAJOR_VERSION, wxMINOR_VERSION, wxRELEASE_NUMBER);
    if (wxSUBRELEASE_NUMBER != 0) {
        mywxVersion.Append(wxString::Format(".%d", wxSUBRELEASE_NUMBER));
    }
    return mywxVersion;
}

wxString lsVersion::GetGDALNumber() {
    wxString myGDAL = wxEmptyString;
#ifdef GDAL_INCLUDE_DIR
    myGDAL = GDAL_RELEASE_NAME;
#endif
    return myGDAL;
}

wxString lsVersion::GetGEOSNumber() {
    wxString myGEOS = wxEmptyString;
#ifdef GEOS_INCLUDE_DIR
    myGEOS = GEOS_VERSION;
#endif
    return myGEOS;
}

wxString lsVersion::GetCurlNumber() {
    wxString myTxt = wxEmptyString;
#ifdef CURL_INCLUDE_DIR
    myTxt = wxString(LIBCURL_VERSION);
#endif
    return myTxt;
}

wxString lsVersion::GetSQLiteNumber() {
    wxString mySQlite = wxEmptyString;
#ifdef SQLITE_LIBRARIES
    mySQlite = wxString(sqlite3_libversion());
#endif
    return mySQlite;
}

wxString lsVersion::GetMySQLNumber() {
    wxString myMySQL = wxEmptyString;
#ifdef MYSQL_INCLUDE_DIR
    myMySQL = wxString(mysql_get_client_info(), wxConvUTF8);
#endif
    return myMySQL;
}

wxString lsVersion::GetNetCDFNumber() {
    wxString ncVers = wxEmptyString;
#ifdef NETCDF_INCLUDE_DIRS
    ncVers = wxString(nc_inq_libvers());
#endif
    return ncVers;
}

wxString lsVersion::GetProjNumber() {
    wxString myProj = wxEmptyString;
#ifdef PROJ4_INCLUDE_DIR
#ifdef PJ_VERSION
    myProj = wxString::Format("%d", PJ_VERSION);
    // Adding points
    if (!myProj.IsEmpty()) {
        wxString myProjDots = wxEmptyString;
        for (unsigned int i = 0; i < myProj.Length(); i++) {
            if (i != myProj.Length() - 1) {
                myProjDots.Append(myProj.Mid(i, 1) + ".");
            } else {
                myProjDots.Append(myProj.Mid(i, 1));
            }
        }
        myProj = myProjDots;
    }
#endif
#ifdef PROJ_VERSION_MAJOR
    myProj = wxString::Format("%d.%d.%d", PROJ_VERSION_MAJOR, PROJ_VERSION_MINOR, PROJ_VERSION_PATCH);
#endif
#endif
    return myProj;
}

wxString lsVersion::GetEigenNumber() {
    wxString myTxt = wxEmptyString;
#ifdef EIGEN_VERSION
    myTxt = wxString::Format("%d.%d.%d", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif
    return myTxt;
}

wxString lsVersion::GetPNGNumber() {
    wxString myTxt = wxEmptyString;
#ifdef PNG_INCLUDE_DIRS
    myTxt = wxString(PNG_LIBPNG_VER_STRING);
#endif
    return myTxt;
}

wxString lsVersion::GetJpegNumber() {
    wxString myTxt = wxEmptyString;
#ifdef JPEG_INCLUDE_DIR
#ifdef JPEG_LIB_VERSION
    myTxt = wxString::Format("%d", JPEG_LIB_VERSION);
    // Adding points
    if (!myTxt.IsEmpty()) {
        wxString myTxtDots = wxEmptyString;
        for (unsigned int i = 0; i < myTxt.Length(); i++) {
            if (i != myTxt.Length() - 1) {
                myTxtDots.Append(myTxt.Mid(i, 1) + ".");
            } else {
                myTxtDots.Append(myTxt.Mid(i, 1));
            }
        }
        myTxt = myTxtDots;
    }
#endif
#ifdef JPEG_LIB_VERSION_MAJOR
    myTxt = wxString::Format("%d.%d", JPEG_LIB_VERSION_MAJOR, JPEG_LIB_VERSION_MINOR);
#endif
#endif
    return myTxt;
}

wxString lsVersion::GetJasperNumber() {
    wxString myTxt = wxEmptyString;
#ifdef JASPER_INCLUDE_DIR
    // myTxt = wxString(JAS_VERSION);
#endif
    return myTxt;
}

wxString lsVersion::GetEcCodesNumber() {
    wxString myTxt = wxEmptyString;
#ifdef ECCODES_LIBRARIES
    myTxt = wxString(ECCODES_VERSION_STR);
#endif
    return myTxt;
}

wxString lsVersion::GetAllModules() {
    wxString myModules = wxEmptyString;

    if (GetGDALNumber() != wxEmptyString) {
        myModules.Append("GDAL: " + GetGDALNumber() + "\n");
    }

    if (GetGEOSNumber() != wxEmptyString) {
        myModules.Append("GEOS: " + GetGEOSNumber() + "\n");
    }

    if (GetCurlNumber() != wxEmptyString) {
        myModules.Append("libCurl: " + GetCurlNumber() + "\n");
    }

    if (GetSQLiteNumber() != wxEmptyString) {
        myModules.Append("SQLite: " + GetSQLiteNumber() + "\n");
    }

    if (GetMySQLNumber() != wxEmptyString) {
        myModules.Append("MySQL: " + GetMySQLNumber() + "\n");
    }

    if (GetNetCDFNumber() != wxEmptyString) {
        myModules.Append("NetCDF: " + GetNetCDFNumber().BeforeFirst(' ') + "\n");
    }

    if (GetProjNumber() != wxEmptyString) {
        myModules.Append("Proj4: " + GetProjNumber() + "\n");
    }

    if (GetEigenNumber() != wxEmptyString) {
        myModules.Append("Eigen: " + GetEigenNumber() + "\n");
    }

    if (GetPNGNumber() != wxEmptyString) {
        myModules.Append("PNG: " + GetPNGNumber() + "\n");
    }

    if (GetJpegNumber() != wxEmptyString) {
        myModules.Append("JPEG: " + GetJpegNumber() + "\n");
    }

    if (GetJasperNumber() != wxEmptyString) {
        myModules.Append("Jasper: " + GetJasperNumber() + "\n");
    }

    if (GetEcCodesNumber() != wxEmptyString) {
        myModules.Append("ecCodes: " + GetEcCodesNumber() + "\n");
    }

    myModules.Append("wxWidgets: " + GetwxWidgetsNumber() + "\n");

    myModules.Append(wxGetOsDescription());

    return myModules;
}
