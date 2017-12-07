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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#include "asPredictandLightnings.h"

#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asCatalogPredictands.h>


asPredictandLightnings::asPredictandLightnings(Parameter dataParameter, TemporalResolution dataTemporalResolution,
                                               SpatialAggregation dataSpatialAggregation)
        : asPredictand(dataParameter, dataTemporalResolution, dataSpatialAggregation)
{
    m_hasNormalizedData = false;
    m_hasReferenceValues = false;
}

asPredictandLightnings::~asPredictandLightnings()
{
    //dtor
}

bool asPredictandLightnings::InitContainers()
{
    return InitBaseContainers();
}

bool asPredictandLightnings::Load(const wxString &filePath)
{
    // Open the NetCDF file
    wxLogVerbose(_("Opening the file %s"), filePath);
    asFileNetcdf ncFile(filePath, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        wxLogError(_("Couldn't open file %s"), filePath);
        return false;
    } else {
        wxLogVerbose(_("File successfully opened"));
    }

    // Load common data
    LoadCommonData(ncFile);

    // Close the netCDF file
    ncFile.Close();

    return true;
}

bool asPredictandLightnings::Save(const wxString &AlternateDestinationDir) const
{
    // Get the file path
    wxString PredictandDBFilePath = GetDBFilePathSaving(AlternateDestinationDir);

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(PredictandDBFilePath, asFileNetcdf::Replace);
    if (!ncFile.Open())
        return false;

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

bool asPredictandLightnings::BuildPredictandDB(const wxString &catalogFilePath, const wxString &AlternateDataDir,
                                               const wxString &AlternatePatternDir,
                                               const wxString &AlternateDestinationDir)
{
    if (!g_unitTesting) {
        wxLogVerbose(_("Building the predictand DB."));
    }

    // Initialize the members
    if (!InitMembers(catalogFilePath))
        return false;

    // Resize matrices
    if (!InitContainers())
        return false;

    // Load data from files
    if (!ParseData(catalogFilePath, AlternateDataDir, AlternatePatternDir))
        return false;

    Save(AlternateDestinationDir);

    if (!g_unitTesting) {
        wxLogVerbose(_("Predictand DB saved."));
    }

#if wxUSE_GUI
    if (!g_silentMode) {
        wxMessageBox(_("Predictand DB saved."));
    }
#endif

    return true;
}
