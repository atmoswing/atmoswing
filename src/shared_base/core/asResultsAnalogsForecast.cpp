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

#include "asResultsAnalogsForecast.h"

#include "asFileNetcdf.h"
#include "asThreadsManager.h"

asResultsAnalogsForecast::asResultsAnalogsForecast(const wxString &modelName)
:
asResults()
{
    m_ModelName = modelName;
    m_FilePath = wxEmptyString;
}

asResultsAnalogsForecast::~asResultsAnalogsForecast()
{
    //dtor
}

void asResultsAnalogsForecast::Init(asParametersForecast &params, double leadTimeOrigin)
{
    // Resize to 0 to avoid keeping old results
    m_TargetDates.resize(0);
    m_StationsNames.resize(0);
    m_StationsIds.resize(0);
    m_StationsHeights.resize(0);
    m_AnalogsNb.resize(0);
    m_AnalogsCriteria.resize(0);
    m_AnalogsDates.resize(0);
    m_AnalogsValuesGross.resize(0);
    m_StationsLat.resize(0);
    m_StationsLon.resize(0);
    m_StationsLocCoordU.resize(0);
    m_StationsLocCoordV.resize(0);
    m_ReturnPeriods.resize(0);
    m_DailyPrecipitationsForReturnPeriods.resize(0);

    m_LeadTimeOrigin = leadTimeOrigin;
    m_DateProcessed = asTime::NowMJD(asUTM);

    // Set the analogs number
    m_AnalogsNb.resize(params.GetLeadTimeNb());
    for (int i=0; i<params.GetLeadTimeNb(); i++)
    {
        m_AnalogsNb[i] = params.GetAnalogsNumberLeadTime(m_CurrentStep, i);
    }

    BuildFileName();
}

void asResultsAnalogsForecast::BuildFileName()
{
    wxASSERT(!m_ModelName.IsEmpty());

    // Base directory
    m_FilePath = wxFileConfig::Get()->Read("/StandardPaths/ForecastResultsDir", asConfig::GetDefaultUserWorkingDir() + "ForecastResults" + DS);
    m_FilePath.Append(DS);

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");
    wxString directory = asTime::GetStringTime(m_LeadTimeOrigin, dirstructure);
    m_FilePath.Append(directory);
    m_FilePath.Append(DS);

    // Filename
    wxString modelname = m_ModelName;
    wxString nowstr = asTime::GetStringTime(m_LeadTimeOrigin, "YYYYMMDDhh");
    wxString ext = "fcst";
    wxString filename = wxString::Format("%s.%s.%s",nowstr.c_str(),modelname.c_str(),ext.c_str());
    m_FilePath.Append(filename);
}

bool asResultsAnalogsForecast::Save(const wxString &AlternateFilePath)
{
    wxASSERT(!m_ModelName.IsEmpty());
    wxASSERT(!m_PredictandDBName.IsEmpty());
    wxASSERT(m_TargetDates.size()>0);
    wxASSERT(m_AnalogsNb.size()>0);
    wxASSERT(m_StationsNames.size()>0);
    wxASSERT(m_StationsHeights.size()>0);
    wxASSERT(m_StationsIds.size()>0);
    wxASSERT(m_ReturnPeriods.size()>0);
    wxASSERT(m_AnalogsCriteria.size()>0);
    wxASSERT(m_AnalogsDates.size()>0);
    wxASSERT(m_AnalogsValuesGross.size()>0);
    wxASSERT(m_StationsLon.size()>0);
    wxASSERT(m_StationsLat.size()>0);
    wxASSERT(m_StationsLocCoordU.size()>0);
    wxASSERT(m_StationsLocCoordV.size()>0);
    wxASSERT(m_DailyPrecipitationsForReturnPeriods.cols()>0);
    wxASSERT(m_DailyPrecipitationsForReturnPeriods.rows()>0);

    wxString message = _("Saving forecast file: ") + m_FilePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty())
    {
        ResultsFile = m_FilePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    // Get the elements size
    size_t Nleadtime = m_TargetDates.size();
    size_t Nanalogstot = m_AnalogsNb.sum();
    size_t Nstations = m_StationsIds.size();
    size_t Nreturnperiods = m_ReturnPeriods.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Set the dates of the forecast
    ncFile.PutAtt("modelName", m_ModelName);
    ncFile.PutAtt("dateProcessed", &m_DateProcessed);
    ncFile.PutAtt("leadTimeOrigin", &m_LeadTimeOrigin);
    ncFile.PutAtt("predictandDBName", m_PredictandDBName);

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("leadtime");
    ncFile.DefDim("analogstot", Nanalogstot);
    ncFile.DefDim("stations", Nstations);
    ncFile.DefDim("returnperiods", Nreturnperiods);

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNamesLeadTime;
    DimNamesLeadTime.push_back("leadtime");
    VectorStdString DimNamesAnalogsTot;
    DimNamesAnalogsTot.push_back("analogstot");
    VectorStdString DimNamesStations;
    DimNamesStations.push_back("stations");
    VectorStdString DimNamesAnalogsStations;
    DimNamesAnalogsStations.push_back("stations");
    DimNamesAnalogsStations.push_back("analogstot");
    VectorStdString DimNameReturnPeriods;
    DimNameReturnPeriods.push_back("returnperiods");
    VectorStdString DimNames2DReturnPeriods;
    DimNames2DReturnPeriods.push_back("returnperiods");
    DimNames2DReturnPeriods.push_back("stations");

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("targetdates", NC_FLOAT, 1, DimNamesLeadTime);
    ncFile.DefVar("analogsnb", NC_INT, 1, DimNamesLeadTime);
    ncFile.DefVar("stationsnames", NC_STRING, 1, DimNamesStations);
    ncFile.DefVar("stationsids", NC_INT, 1, DimNamesStations);
    ncFile.DefVar("stationsheights", NC_FLOAT, 1, DimNamesStations);
    ncFile.DefVar("lon", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("lat", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("loccoordu", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("loccoordv", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("returnperiods", NC_FLOAT, 1, DimNameReturnPeriods);
    ncFile.DefVar("dailyprecipitationsforreturnperiods", NC_FLOAT, 2, DimNames2DReturnPeriods);
    ncFile.DefVar("analogscriteria", NC_FLOAT, 1, DimNamesAnalogsTot);
    ncFile.DefVar("analogsdates", NC_FLOAT, 1, DimNamesAnalogsTot);
    ncFile.DefVar("analogsvaluesgross", NC_FLOAT, 2, DimNamesAnalogsStations);

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefStationsIdsAttributes(ncFile);
    DefAnalogsNbAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesGrossAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

    ncFile.PutAtt("long_name", "Station names", "stationsnames");
    ncFile.PutAtt("var_desc", "Name of the weather stations", "stationsnames");
    ncFile.PutAtt("long_name", "Station heights", "stationsheights");
    ncFile.PutAtt("var_desc", "Altitude of the weather stations", "stationsheights");
    ncFile.PutAtt("units", "m", "stationsheights");
    ncFile.PutAtt("long_name", "Longitude", "lon");
    ncFile.PutAtt("var_desc", "Longitudes of the stations positions", "lon");
    ncFile.PutAtt("units", "degrees", "lon");
    ncFile.PutAtt("long_name", "Latitude", "lat");
    ncFile.PutAtt("var_desc", "Latitudes of the stations positions", "lat");
    ncFile.PutAtt("units", "degrees", "lat");
    ncFile.PutAtt("long_name", "Local coordinate U", "loccoordu");
    ncFile.PutAtt("var_desc", "Local coordinate for the U axis (west-east)", "loccoordu");
    ncFile.PutAtt("units", "m", "loccoordu");
    ncFile.PutAtt("long_name", "Local coordinate V", "loccoordv");
    ncFile.PutAtt("var_desc", "Local coordinate for the V axis (west-east)", "loccoordv");
    ncFile.PutAtt("units", "m", "loccoordv");
    ncFile.PutAtt("long_name", "Return periods", "returnperiods");
    ncFile.PutAtt("var_desc", "Return periods", "returnperiods");
    ncFile.PutAtt("units", "year", "returnperiods");
    ncFile.PutAtt("long_name", "Daily precipitation for return periods", "dailyprecipitationsforreturnperiods");
    ncFile.PutAtt("var_desc", "Daily precipitation corresponding to the return periods", "dailyprecipitationsforreturnperiods");
    ncFile.PutAtt("units", "mm", "dailyprecipitationsforreturnperiods");

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startLeadTime[] = {0};
    size_t countLeadTime[] = {Nleadtime};
    size_t startAnalogsTot[] = {0};
    size_t countAnalogsTot[] = {Nanalogstot};
    size_t startStations[] = {0};
    size_t countStations[] = {Nstations};
    size_t startAnalogsStations[] = {0,0};
    size_t countAnalogsStations[] = {Nstations, Nanalogstot};
    size_t startReturnPeriod[] = {0};
    size_t countReturnPeriod[] = {Nreturnperiods};
    size_t startReturnPeriodPrecip[] = {0, 0};
    size_t countReturnPeriodPrecip[] = {Nreturnperiods, Nstations};

    // Set the matrices in vectors
    VectorFloat analogsCriteria(Nanalogstot);
    VectorFloat analogsDates(Nanalogstot);
    VectorFloat analogsValuesGross(Nanalogstot*Nstations);

    int ind = 0, indVal = 0;
    for (unsigned int i_time=0; i_time<Nleadtime; i_time++)
    {
        for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
        {
            analogsCriteria[ind] = m_AnalogsCriteria[i_time][i_analog];
            analogsDates[ind] = m_AnalogsDates[i_time][i_analog];
            ind++;

            for (unsigned int i_station=0; i_station<Nstations; i_station++)
            {
                analogsValuesGross[indVal] = m_AnalogsValuesGross[i_time](i_station, i_analog);
                indVal++;
            }
        }
    }
/*
    // Display dimensions to chck in case of an error.
    wxLogMessage(wxString::Format("targetdates - start=%d, count=%d, size=%d", (int)startLeadTime[0], (int)countLeadTime[0], (int)m_TargetDates.size()));
    wxLogMessage(wxString::Format("analogsnb - start=%d, count=%d, size=%d", (int)startLeadTime[0], (int)countLeadTime[0], (int)m_AnalogsNb.size()));
    wxLogMessage(wxString::Format("stationsnames - start=%d, count=%d, size=%d, Nstations=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsNames.size(), Nstations));
    wxLogMessage(wxString::Format("stationsids - start=%d, count=%d, size=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsIds.size()));
    wxLogMessage(wxString::Format("stationsheights - start=%d, count=%d, size=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsHeights.size()));
    wxLogMessage(wxString::Format("lon - start=%d, count=%d, size=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsLon.size()));
    wxLogMessage(wxString::Format("lat - start=%d, count=%d, size=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsLat.size()));
    wxLogMessage(wxString::Format("loccoordu - start=%d, count=%d, size=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsLocCoordU.size()));
    wxLogMessage(wxString::Format("loccoordv - start=%d, count=%d, size=%d", (int)startStations[0], (int)countStations[0], (int)m_StationsLocCoordV.size()));
    wxLogMessage(wxString::Format("returnperiods - start=%d, count=%d, size=%d", (int)startReturnPeriod[0], (int)countReturnPeriod[0], (int)m_ReturnPeriods.size()));
    wxLogMessage(wxString::Format("dailyprecipitationsforreturnperiods - start[0]=%d, start[1]=%d, count[0]=%d, count[1]=%d, rows=%d, cols=%d", (int)startReturnPeriodPrecip[0], (int)startReturnPeriodPrecip[1], (int)countReturnPeriodPrecip[0], (int)countReturnPeriodPrecip[1], (int)m_DailyPrecipitationsForReturnPeriods.rows(), (int)m_DailyPrecipitationsForReturnPeriods.cols()));
    wxLogMessage(wxString::Format("analogscriteria - start=%d, count=%d, size=%d", (int)startAnalogsTot[0], (int)countAnalogsTot[0], (int)analogsCriteria.size()));
    wxLogMessage(wxString::Format("analogsdates - start=%d, count=%d, size=%d", (int)startAnalogsTot[0], (int)countAnalogsTot[0], (int)analogsDates.size()));
    wxLogMessage(wxString::Format("analogsvaluesgross - start[0]=%d, start[1]=%d, count[0]=%d, count[1]=%d, size=%d", (int)startAnalogsStations[0], (int)startAnalogsStations[1], (int)countAnalogsStations[0], (int)countAnalogsStations[1], (int)analogsValuesGross.size()));
*/
    // Write data
    ncFile.PutVarArray("targetdates", startLeadTime, countLeadTime, &m_TargetDates[0]);
    ncFile.PutVarArray("analogsnb", startLeadTime, countLeadTime, &m_AnalogsNb[0]);
    ncFile.PutVarArray("stationsnames", startStations, countStations, &m_StationsNames[0], Nstations);
    ncFile.PutVarArray("stationsids", startStations, countStations, &m_StationsIds[0]);
    ncFile.PutVarArray("stationsheights", startStations, countStations, &m_StationsHeights[0]);
    ncFile.PutVarArray("lon", startStations, countStations, &m_StationsLon(0));
    ncFile.PutVarArray("lat", startStations, countStations, &m_StationsLat(0));
    ncFile.PutVarArray("loccoordu", startStations, countStations, &m_StationsLocCoordU(0));
    ncFile.PutVarArray("loccoordv", startStations, countStations, &m_StationsLocCoordV(0));
    ncFile.PutVarArray("returnperiods", startReturnPeriod, countReturnPeriod, &m_ReturnPeriods(0));
    ncFile.PutVarArray("dailyprecipitationsforreturnperiods", startReturnPeriodPrecip, countReturnPeriodPrecip, &m_DailyPrecipitationsForReturnPeriods(0,0));
    ncFile.PutVarArray("analogscriteria", startAnalogsTot, countAnalogsTot, &analogsCriteria[0]);
    ncFile.PutVarArray("analogsdates", startAnalogsTot, countAnalogsTot, &analogsDates[0]);
    ncFile.PutVarArray("analogsvaluesgross", startAnalogsStations, countAnalogsStations, &analogsValuesGross[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asResultsAnalogsForecast::Load(const wxString &AlternateFilePath)
{
    // Get the file path
    if (!AlternateFilePath.IsEmpty())
    {
        m_FilePath = AlternateFilePath;
    }

    // If we don't want to save or the file doesn't exist
    if(!Exists()) return false;
    if(m_CurrentStep!=0) return false;

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(m_FilePath, asFileNetcdf::ReadOnly);
    if(!ncFile.Open()) return false;

    // Get general attributes
    m_ModelName = ncFile.GetAttString("modelName");
    m_DateProcessed = ncFile.GetAttDouble("dateProcessed");
    m_LeadTimeOrigin = ncFile.GetAttDouble("leadTimeOrigin");
    m_PredictandDBName = ncFile.GetAttString("predictandDBName");

    // Get the elements size
    int Nleadtime = ncFile.GetDimLength("leadtime");
    int Nanalogstot = ncFile.GetDimLength("analogstot");
    int Nstations = ncFile.GetDimLength("stations");

    // Get lead time data
    m_TargetDates.resize( Nleadtime );
    ncFile.GetVar("targetdates", &m_TargetDates[0]);
    m_AnalogsNb.resize( Nleadtime );
    ncFile.GetVar("analogsnb", &m_AnalogsNb[0]);
    m_StationsNames.resize( Nstations );
    ncFile.GetVar("stationsnames", &m_StationsNames[0], Nstations);
    m_StationsIds.resize( Nstations );
    ncFile.GetVar("stationsids", &m_StationsIds[0]);
    m_StationsHeights.resize( Nstations );
    ncFile.GetVar("stationsheights", &m_StationsHeights[0]);
    m_StationsLon.resize( Nstations );
    ncFile.GetVar("lon", &m_StationsLon[0]);
    m_StationsLat.resize( Nstations );
    ncFile.GetVar("lat", &m_StationsLat[0]);
    m_StationsLocCoordU.resize( Nstations );
    ncFile.GetVar("loccoordu", &m_StationsLocCoordU[0]);
    m_StationsLocCoordV.resize( Nstations );
    ncFile.GetVar("loccoordv", &m_StationsLocCoordV[0]);

    // Get return periods properties
    int returnPeriodsNb = ncFile.GetDimLength("returnperiods");
    m_ReturnPeriods.resize( returnPeriodsNb );
    ncFile.GetVar("returnperiods", &m_ReturnPeriods[0]);
    size_t startReturnPeriodPrecip[2] = {0, 0};
    size_t countReturnPeriodPrecip[2] = {size_t(returnPeriodsNb), size_t(Nstations)};
    m_DailyPrecipitationsForReturnPeriods.resize( Nstations, returnPeriodsNb );
    ncFile.GetVarArray("dailyprecipitationsforreturnperiods", startReturnPeriodPrecip, countReturnPeriodPrecip, &m_DailyPrecipitationsForReturnPeriods(0,0));

    // Create vectors for matrices data
    VectorFloat analogsCriteria(Nanalogstot);
    VectorFloat analogsDates(Nanalogstot);
    VectorFloat analogsValuesGross(Nanalogstot*Nstations);

    // Get data
    size_t IndexStart1D[] = {0};
    size_t IndexCount1D[] = {size_t(Nanalogstot)};
    size_t IndexStart2D[] = {0,0};
    size_t IndexCount2D[] = {size_t(Nstations), size_t(Nanalogstot)};
    ncFile.GetVarArray("analogscriteria", IndexStart1D, IndexCount1D, &analogsCriteria[0]);
    ncFile.GetVarArray("analogsdates", IndexStart1D, IndexCount1D, &analogsDates[0]);
    ncFile.GetVarArray("analogsvaluesgross", IndexStart2D, IndexCount2D, &analogsValuesGross[0]);

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    // Set data into the matrices
    int ind = 0, indVal = 0;
    for (int i_time=0; i_time<Nleadtime; i_time++)
    {
        Array1DFloat analogsCriteriaLeadTime(m_AnalogsNb[i_time]);
        Array1DFloat analogsDatesLeadTime(m_AnalogsNb[i_time]);
        Array2DFloat analogsValuesGrossLeadTime(Nstations, m_AnalogsNb[i_time]);
        for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
        {
            analogsCriteriaLeadTime(i_analog) = analogsCriteria[ind];
            analogsDatesLeadTime(i_analog) = analogsDates[ind];
            ind++;

            for (int i_station=0; i_station<Nstations; i_station++)
            {
                analogsValuesGrossLeadTime(i_station, i_analog) = analogsValuesGross[indVal];
                indVal++;
            }
        }

        m_AnalogsCriteria.push_back( analogsCriteriaLeadTime );
        m_AnalogsValuesGross.push_back( analogsValuesGrossLeadTime );
        m_AnalogsDates.push_back( analogsDatesLeadTime );
    }

    wxASSERT(!m_ModelName.IsEmpty());
    wxASSERT(!m_PredictandDBName.IsEmpty());
    wxASSERT(m_TargetDates.size()>0);
    wxASSERT(m_AnalogsNb.size()>0);
    wxASSERT(m_StationsIds.size()>0);
    wxASSERT(m_StationsNames.size()>0);
    wxASSERT(m_StationsHeights.size()>0);
    wxASSERT(m_ReturnPeriods.size()>0);
    wxASSERT(m_AnalogsCriteria.size()>0);
    wxASSERT(m_AnalogsDates.size()>0);
    wxASSERT(m_AnalogsValuesGross.size()>0);
    wxASSERT(m_StationsLon.size()>0);
    wxASSERT(m_StationsLat.size()>0);
    wxASSERT(m_StationsLocCoordU.size()>0);
    wxASSERT(m_StationsLocCoordV.size()>0);
    wxASSERT(m_DailyPrecipitationsForReturnPeriods.cols()>0);
    wxASSERT(m_DailyPrecipitationsForReturnPeriods.rows()>0);

    return true;
}

wxArrayString asResultsAnalogsForecast::GetStationNamesWxArrayString()
{
    wxArrayString stationsNames;
    for (unsigned int i=0; i<m_StationsNames.size(); i++)
    {
        stationsNames.Add(m_StationsNames[i]);
    }
    return stationsNames;
}

wxArrayString asResultsAnalogsForecast::GetStationNamesAndHeightsWxArrayString()
{
    wxArrayString stationsNames;
    for (unsigned int i=0; i<m_StationsNames.size(); i++)
    {
        wxString label = wxString::Format("%s (%4.0fm)", m_StationsNames[i].c_str(), m_StationsHeights[i]);
        stationsNames.Add(label);
    }
    return stationsNames;
}

wxString asResultsAnalogsForecast::GetStationNameAndHeight(int i_stat)
{
    wxString stationName = wxString::Format("%s (%4.0fm)", m_StationsNames[i_stat].c_str(), m_StationsHeights[i_stat]);
    return stationName;
}
