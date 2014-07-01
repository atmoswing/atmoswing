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

#include "asResults.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTime.h>


asResults::asResults()
{
    m_SaveIntermediateResults = false;
    m_LoadIntermediateResults = false;
    m_CurrentStep = 0;
    m_FileVersion = 1.3f;
}

asResults::~asResults()
{
    //dtor
}

bool asResults::Load(const wxString &AlternateFilePath)
{
    return false;
}

bool asResults::Save(const wxString &AlternateFilePath)
{
    return false;
}

bool asResults::Exists()
{
    return asFile::Exists(m_FilePath);
}

wxString asResults::GetPredictandStationIdsList()
{
    wxString id;

    if (m_PredictandStationIds.size()==1)
    {
        id << m_PredictandStationIds[0];
    }
    else
    {
        for (int i=0; i<m_PredictandStationIds.size(); i++)
        {
            id << m_PredictandStationIds[i];

            if (i<m_PredictandStationIds.size()-1)
            {
                id << ",";
            }
        }
    }

    return id;
}

bool asResults::DefTargetDatesAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Target dates", "target_dates");
    ncFile.PutAtt("var_desc", "Date of the day to forecast", "target_dates");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "target_dates");
    return true;
}

bool asResults::DefStationsIdsAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Stations IDs", "stations_ids");
    ncFile.PutAtt("var_desc", "The stations IDs", "stations_ids");
    return true;
}

bool asResults::DefAnalogsNbAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs number", "analogs_nb");
    ncFile.PutAtt("var_desc", "Analogs number for the lead times", "analogs_nb");
    return true;
}

bool asResults::DefTargetValuesNormAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Target predictand normalized values", "target_values_norm");
    ncFile.PutAtt("var_desc", "Observed predictand values in a nomalized form", "target_values_norm");
    return true;
}

bool asResults::DefTargetValuesGrossAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Target predictand gross values", "target_values_gross");
    ncFile.PutAtt("var_desc", "Observed predictand values in the original form", "target_values_gross");
    return true;
}

bool asResults::DefAnalogsCriteriaAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs criteria", "analogs_criteria");
    ncFile.PutAtt("var_desc", "Criteria matching the dates from the analog method", "analogs_criteria");
    return true;
}

bool asResults::DefAnalogsDatesAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs dates", "analogs_dates");
    ncFile.PutAtt("var_desc", "Analogs dates from the analog method", "analogs_dates");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "analogs_dates");
    return true;
}

bool asResults::DefAnalogsValuesNormAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs predictand normalized values", "analogs_values_norm");
    ncFile.PutAtt("var_desc", "Predictand values (normalized) corresponding to the scores from the analog method", "analogs_values_norm");
    return true;
}

bool asResults::DefAnalogsValuesGrossAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs predictand gross values", "analogs_values_gross");
    ncFile.PutAtt("var_desc", "Predictand values (original) corresponding to the scores from the analog method", "analogs_values_gross");
    return true;
}

bool asResults::DefForecastScoresAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Forecast scores", "forecast_scores");
    ncFile.PutAtt("var_desc", "Scores of the forecast resulting from the analog method", "forecast_scores");
    return true;
}

bool asResults::DefForecastScoreFinalAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Final score", "forecast_score");
    ncFile.PutAtt("var_desc", "Final score of the method", "forecast_score");
    return true;
}

bool asResults::DefLonLatAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Lat", "lat");
    ncFile.PutAtt("long_name", "Lon", "lon");
    ncFile.PutAtt("units", "degrees_north", "lat");
    ncFile.PutAtt("units", "degrees_east", "lon");
    return true;
}

bool asResults::DefLevelAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Level", "level");
    ncFile.PutAtt("units", "millibar", "level");
    return true;
}

bool asResults::DefScoresMapAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Forecast score", "forecast_scores");
    ncFile.PutAtt("var_desc", "Map of the forecast scores", "forecast_scores");
    ncFile.PutAtt("units", "no unit", "forecast_scores");
    return true;
}
