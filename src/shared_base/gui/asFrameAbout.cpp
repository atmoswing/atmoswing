/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#include "asFrameAbout.h"
#include "asIncludes.h"
#include "tinyxml.h"
#ifndef MINIMAL_LINKS
    #include "gdal.h"
    #include "curl/curl.h"
#endif
#include "proj_api.h"
#ifdef __WXMSW__
    #include "config.h" // netCDF
#endif

asFrameAbout::asFrameAbout( wxWindow* parent )
:
asFrameAboutVirtual( parent )
{
    // Icon
#ifdef __WXMSW__
    SetIcon(wxICON(myicon));
#endif

    // Get Atmoswing version
    m_StaticTextVersion->SetLabel(asVersion::GetFullString());

    // Get wxWidgets version
    wxString versionWxWigets(wxVERSION_STRING);
    m_VersionWxWidgets = new wxStaticText( m_Panel, wxID_ANY, versionWxWigets, wxDefaultPosition, wxDefaultSize, 0 );
    m_VersionWxWidgets->Wrap( -1 );
    m_GridSizer->Add( m_VersionWxWidgets, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

    // Get Eigen version
    wxString versionEigen = wxString::Format("Eigen %d.%d.%d", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
	m_VersionEigen = new wxStaticText( m_Panel, wxID_ANY, versionEigen, wxDefaultPosition, wxDefaultSize, 0 );
	m_VersionEigen->Wrap( -1 );
	m_GridSizer->Add( m_VersionEigen, 0, wxRIGHT|wxLEFT, 5 );

	// Get TiCpp version
    wxString versionTiCPP = wxString::Format("TiCPP (TinyXML %d.%d.%d)", TIXML_MAJOR_VERSION, TIXML_MINOR_VERSION, TIXML_PATCH_VERSION);
    m_VersionTiCPP = new wxStaticText( m_Panel, wxID_ANY, versionTiCPP, wxDefaultPosition, wxDefaultSize, 0 );
    m_VersionTiCPP->Wrap( -1 );
    m_GridSizer->Add( m_VersionTiCPP, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

    #ifndef MINIMAL_LINKS
    // Get Gdal version
    wxString versionGdal = wxString::Format("Gdal %d.%d.%d.%d", GDAL_VERSION_MAJOR, GDAL_VERSION_MINOR, GDAL_VERSION_REV, GDAL_VERSION_BUILD);
	m_VersionGdal = new wxStaticText( m_Panel, wxID_ANY, versionGdal, wxDefaultPosition, wxDefaultSize, 0 );
	m_VersionGdal->Wrap( -1 );
	m_GridSizer->Add( m_VersionGdal, 0, wxRIGHT|wxLEFT, 5 );

    // Get cURL version
    wxString versionCurl(curl_version());
    versionCurl.Replace("/", " ");
    int index1 = versionCurl.Find(" ");
    if (index1!=wxNOT_FOUND)
    {
        int index2 = versionCurl.SubString(index1+1, versionCurl.Len()-1).Find(" ");
        if (index2!=wxNOT_FOUND)
        {
            versionCurl.Truncate(index1+index2+1);
        }
    }
    m_VersionCurl = new wxStaticText( m_Panel, wxID_ANY, versionCurl, wxDefaultPosition, wxDefaultSize, 0 );
    m_VersionCurl->Wrap( -1 );
    m_GridSizer->Add( m_VersionCurl, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );
    #endif

    // Get Proj version
    wxString versionProj = "Proj ";
    wxString versNb = wxString::Format("%d", PJ_VERSION);
    for (unsigned int i=0; i<versNb.Length(); i++)
    {
        if (i!=versNb.Length()-1)
        {
            versionProj.Append(versNb.Mid(i,1)+".");
        }
        else
        {
            versionProj.Append(versNb.Mid(i,1));
        }
    }
	m_VersionProj = new wxStaticText( m_Panel, wxID_ANY, versionProj, wxDefaultPosition, wxDefaultSize, 0 );
	m_VersionProj->Wrap( -1 );
	m_GridSizer->Add( m_VersionProj, 0, wxRIGHT|wxLEFT, 5 );

    // Set NetCDF
#ifdef __WXMSW__
    wxString ncVers(PACKAGE_VERSION);
    wxString versionNetCDF = "NetCDF " + ncVers;
#else
    wxString versionNetCDF = "NetCDF 4";
#endif
    m_VersionNetCDF = new wxStaticText( m_Panel, wxID_ANY, versionNetCDF, wxDefaultPosition, wxDefaultSize, 0 );
    m_VersionNetCDF->Wrap( -1 );
    m_GridSizer->Add( m_VersionNetCDF, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );

    // Set VroomGIS
	m_VersionVroomGIS = new wxStaticText( m_Panel, wxID_ANY, "VroomGIS", wxDefaultPosition, wxDefaultSize, 0 );
	m_VersionVroomGIS->Wrap( -1 );
	m_GridSizer->Add( m_VersionVroomGIS, 0, wxRIGHT|wxLEFT, 5 );

	// Set wxPlotCtrl
    m_VersionWxPlotCtrl = new wxStaticText( m_Panel, wxID_ANY, "wxPlotCtrl", wxDefaultPosition, wxDefaultSize, 0 );
    m_VersionWxPlotCtrl->Wrap( -1 );
    m_GridSizer->Add( m_VersionWxPlotCtrl, 0, wxRIGHT|wxLEFT|wxALIGN_RIGHT, 5 );
}
