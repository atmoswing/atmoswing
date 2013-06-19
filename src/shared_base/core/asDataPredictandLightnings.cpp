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
 
#include "asDataPredictandLightnings.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asCatalog.h>
#include <asCatalogPredictands.h>


asDataPredictandLightnings::asDataPredictandLightnings(DataParameter dataParameter, DataTemporalResolution dataTemporalResolution, DataSpatialAggregation dataSpatialAggregation)
:
asDataPredictand(dataParameter, dataTemporalResolution, dataSpatialAggregation)
{
    //ctor
	m_HasNormalizedData = false;
	m_HasReferenceValues = false;
}

asDataPredictandLightnings::~asDataPredictandLightnings()
{
    //dtor
}

bool asDataPredictandLightnings::InitContainers()
{
    if (!InitBaseContainers()) return false;
    return true;
}

bool asDataPredictandLightnings::Load(const wxString &filePath)
{
    // Open the NetCDF file
    asLogMessage(wxString::Format(_("Opening the file %s"), filePath.c_str()));
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if(!ncFile.Open())
    {
        asLogError(wxString::Format(_("Couldn't open file %s"), filePath.c_str()));
        return false;
    }
    else
    {
        asLogMessage(_("File successfully opened"));
    }

	// Load common data
	LoadCommonData(ncFile);

	// Close the netCDF file
	ncFile.Close();

    return true;
}

bool asDataPredictandLightnings::Save(const wxString &AlternateDestinationDir)
{
    // Get the file path
    wxString PredictandDBFilePath = GetDBFilePathSaving(AlternateDestinationDir);

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(PredictandDBFilePath, asFileNetcdf::Replace);
    if(!ncFile.Open()) return false;

	// Set common definitions
	SetCommonDefinitions(ncFile);
	
    // End definitions: leave define mode
    ncFile.EndDef();

	// Save common data
    SaveCommonData(ncFile);

    // Close:save new netCDF dataset
    ncFile.Close();

    return true;
}

bool asDataPredictandLightnings::BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir, const wxString &AlternatePatternDir, const wxString &AlternateDestinationDir)
{
    if(!g_UnitTesting) asLogMessage(_("Building the predictand DB."));

    // Initialize the members
    if(!InitMembers(catalogFilePath)) return false;

    // Resize matrices
    if(!InitContainers()) return false;

	// Load data from files
    if(!ParseData(catalogFilePath, AlternateDataDir, AlternatePatternDir)) return false;

    Save(AlternateDestinationDir);

    if(!g_UnitTesting) asLogMessage(_("Predictand DB saved."));

    #if wxUSE_GUI
        if (!g_SilentMode)
        {
            wxMessageBox(_("Predictand DB saved."));
        }
    #endif

    return true;
}
