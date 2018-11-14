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

wxString lsVersion::GetSoftName()
{
    wxString myName = wxEmptyString;
#ifdef lsVERSION_SOFT_NAME
    myName = lsVERSION_SOFT_NAME;
#endif
    return myName;
}


wxString lsVersion::GetSoftGIT()
{
    wxString myGITVersion = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION
    myGITVersion = lsVERSION_SOFT_VERSION;
#endif
    return myGITVersion;
}


wxString lsVersion::GetSoftGITBranch()
{
    wxString myGITtxt = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION_BRANCH
    myGITtxt = lsVERSION_SOFT_VERSION_BRANCH;
#endif
    return myGITtxt;
}


wxString lsVersion::GetSoftGITRevisionHash()
{
    wxString myGITtxt = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION_REVISION
    myGITtxt = lsVERSION_SOFT_VERSION_REVISION;
#endif
    return myGITtxt;
}


wxString lsVersion::GetSoftGITRevisionNb()
{
    wxString myGITtxt = wxEmptyString;
#ifdef lsVERSION_SOFT_VERSION
    myGITtxt = lsVERSION_SOFT_VERSION;
#endif
    return myGITtxt;
}


wxString lsVersion::GetwxWidgetsNumber()
{
    wxString mywxVersion = wxString::Format("%d.%d.%d", wxMAJOR_VERSION, wxMINOR_VERSION, wxRELEASE_NUMBER);
    if (wxSUBRELEASE_NUMBER != 0) {
        mywxVersion.Append(wxString::Format(".%d", wxSUBRELEASE_NUMBER));
    }
    return mywxVersion;
}


wxString lsVersion::GetGDALNumber()
{
    wxString myGDAL = wxEmptyString;
#ifdef GDAL_INCLUDE_DIR
    myGDAL = GDAL_RELEASE_NAME;
#endif
    return myGDAL;
}


wxString lsVersion::GetGEOSNumber()
{
    wxString myGEOS = wxEmptyString;
#ifdef GEOS_INCLUDE_DIR
    myGEOS = GEOS_VERSION;
#endif
    return myGEOS;
}


wxString lsVersion::GetCurlNumber()
{
    wxString myTxt = wxEmptyString;
#ifdef CURL_INCLUDE_DIR
    myTxt = wxString(LIBCURL_VERSION);
#endif
    return myTxt;
}


wxString lsVersion::GetSQLiteNumber()
{
    wxString mySQlite = wxEmptyString;
#ifdef SQLITE_LIBRARIES
    mySQlite  = wxString(sqlite3_libversion());
#endif
    return mySQlite;
}


wxString lsVersion::GetMySQLNumber()
{
    wxString myMySQL = wxEmptyString;
#ifdef MYSQL_INCLUDE_DIR
    myMySQL = wxString(mysql_get_client_info(), wxConvUTF8);
#endif
    return myMySQL;
}


wxString lsVersion::GetNetCDFNumber()
{
    wxString ncVers = wxEmptyString;
#ifdef NETCDF_INCLUDE_DIRS
    ncVers = wxString(nc_inq_libvers());
#endif
    return ncVers;
}


wxString lsVersion::GetProjNumber()
{
    wxString myProj = wxEmptyString;
#ifdef PROJ_LIBRARY
    myProj = wxString::Format("%d", PJ_VERSION);
#elif defined PROJ4_INCLUDE_DIR
    myProj = wxString::Format("%d", PJ_VERSION);
#endif
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

    return myProj;
}


wxString lsVersion::GetEigenNumber()
{
    wxString myTxt = wxEmptyString;
#ifdef EIGEN_VERSION
    myTxt = wxString::Format("%d.%d.%d", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif
    return myTxt;
}


wxString lsVersion::GetAllModules()
{
    wxString myModules = wxEmptyString;

    if (GetGDALNumber() != wxEmptyString) {
        myModules.Append(_T("GDAL: ") + GetGDALNumber() + _T("\n"));
    }

    if (GetGEOSNumber() != wxEmptyString) {
        myModules.Append(_T("GEOS: ") + GetGEOSNumber() + _T("\n"));
    }

    if (GetCurlNumber() != wxEmptyString) {
        myModules.Append(_T("libCurl: ") + GetCurlNumber() + _T("\n"));
    }

    if (GetSQLiteNumber() != wxEmptyString) {
        myModules.Append(_T("SQLite: ") + GetSQLiteNumber() + _T("\n"));
    }

    if (GetMySQLNumber() != wxEmptyString) {
        myModules.Append(_T("MySQL: ") + GetMySQLNumber() + _T("\n"));
    }

    if (GetNetCDFNumber() != wxEmptyString) {
        myModules.Append(_T("NetCDF: ") + GetNetCDFNumber().BeforeFirst(' ') + _T("\n"));
    }

    if (GetProjNumber() != wxEmptyString) {
        myModules.Append(_T("Proj4: ") + GetProjNumber() + _T("\n"));
    }

    if (GetEigenNumber() != wxEmptyString) {
        myModules.Append(_T("Eigen: ") + GetEigenNumber() + _T("\n"));
    }

    myModules.Append(_T("wxWidgets: ") + GetwxWidgetsNumber() + _T("\n"));

    myModules.Append(wxGetOsDescription());

    return myModules;
}

