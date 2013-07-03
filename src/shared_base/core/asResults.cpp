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
 
#include "asResults.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTime.h>


asResults::asResults()
{
    m_SaveIntermediateResults = false;
    m_LoadIntermediateResults = false;
    m_CurrentStep = 0;
    m_PredictandStationId = 0;
    m_FileVersion = 1.1f;
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

bool asResults::DefTargetDatesAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Target dates", "targetdates");
    ncFile.PutAtt("var_desc", "Date of the day to forecast", "targetdates");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "targetdates");
    return true;
}

bool asResults::DefStationsIdsAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Stations IDs", "stationsids");
    ncFile.PutAtt("var_desc", "The stations IDs", "stationsids");
    return true;
}

bool asResults::DefAnalogsNbAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs number", "analogsnb");
    ncFile.PutAtt("var_desc", "Analogs number for the lead times", "analogsnb");
    return true;
}

bool asResults::DefTargetValuesNormAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Target predictand normalized values", "targetvaluesnorm");
    ncFile.PutAtt("var_desc", "Observed predictand values in a nomalized form", "targetvaluesnorm");
    return true;
}

bool asResults::DefTargetValuesGrossAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Target predictand gross values", "targetvaluesgross");
    ncFile.PutAtt("var_desc", "Observed predictand values in the original form", "targetvaluesgross");
    return true;
}

bool asResults::DefAnalogsCriteriaAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs criteria", "analogscriteria");
    ncFile.PutAtt("var_desc", "Criteria matching the dates from the analog method", "analogscriteria");
    return true;
}

bool asResults::DefAnalogsDatesAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs dates", "analogsdates");
    ncFile.PutAtt("var_desc", "Analogs dates from the analog method", "analogsdates");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "analogsdates");
    return true;
}

bool asResults::DefAnalogsValuesNormAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs predictand normalized values", "analogsvaluesnorm");
    ncFile.PutAtt("var_desc", "Predictand values (normalized) corresponding to the scores from the analog method", "analogsvaluesnorm");
    return true;
}

bool asResults::DefAnalogsValuesGrossAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Analogs predictand gross values", "analogsvaluesgross");
    ncFile.PutAtt("var_desc", "Predictand values (original) corresponding to the scores from the analog method", "analogsvaluesgross");
    return true;
}

bool asResults::DefForecastScoresAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Forecast scores", "forecastscores");
    ncFile.PutAtt("var_desc", "Scores of the forecast resulting from the analog method", "forecastscores");
    return true;
}

bool asResults::DefForecastScoreFinalAttributes(asFileNetcdf &ncFile)
{
    ncFile.PutAtt("long_name", "Final score", "forecastscore");
    ncFile.PutAtt("var_desc", "Final score of the method", "forecastscore");
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
    ncFile.PutAtt("long_name", "Forecast score", "forecastscores");
    ncFile.PutAtt("var_desc", "Map of the forecast scores", "forecastscores");
    ncFile.PutAtt("units", "no unit", "forecastscores");
    return true;
}
