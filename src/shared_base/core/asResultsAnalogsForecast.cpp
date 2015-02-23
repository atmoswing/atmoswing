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

#include "asResultsAnalogsForecast.h"

#include "asFileNetcdf.h"
#include "asThreadsManager.h"
#include <wx/tokenzr.h>

asResultsAnalogsForecast::asResultsAnalogsForecast()
:
asResults()
{
    m_FilePath = wxEmptyString;
    m_HasReferenceValues = false;
    m_LeadTimeOrigin = 0.0;

    // Default values for former versions
    m_PredictandParameter = Precipitation;
    m_PredictandTemporalResolution = Daily;
    m_PredictandSpatialAggregation = Station;
    m_PredictandDatasetId = wxEmptyString;
    m_PredictandDatabase = wxEmptyString;
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
    m_StationsLocCoordX.resize(0);
    m_StationsLocCoordY.resize(0);
    m_ReferenceAxis.resize(0);
    m_ReferenceValues.resize(0,0);

    m_MethodId = params.GetMethodId();
    m_MethodIdDisplay = params.GetMethodIdDisplay();
    m_SpecificTag = params.GetSpecificTag();
    m_SpecificTagDisplay = params.GetSpecificTagDisplay();
    m_Description = params.GetDescription();
    m_PredictandDatabase = params.GetPredictandDatabase();
    m_PredictandStationIds = params.GetPredictandStationIds();

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
    wxASSERT(!m_ForecastsDirectory.IsEmpty());

    if(m_MethodId.IsEmpty() || m_SpecificTag.IsEmpty())
    {
        asLogError(_("The provided ID or the tag is empty, which isn't allowed !"));
    }

    // Base directory
    m_FilePath = m_ForecastsDirectory;
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
    wxString modelname = m_MethodId + '.' + m_SpecificTag;
    wxString nowstr = asTime::GetStringTime(m_LeadTimeOrigin, "YYYYMMDDhh");
    wxString ext = "fcst";
    wxString filename = wxString::Format("%s.%s.%s",nowstr.c_str(),modelname.c_str(),ext.c_str());
    m_FilePath.Append(filename);
}

bool asResultsAnalogsForecast::Save(const wxString &AlternateFilePath)
{
    wxASSERT(!m_FilePath.IsEmpty());
    wxASSERT(m_TargetDates.size()>0);
    wxASSERT(m_AnalogsNb.size()>0);
    wxASSERT(m_StationsNames.size()>0);
    wxASSERT(m_StationsHeights.size()>0);
    wxASSERT(m_StationsIds.size()>0);
    wxASSERT(m_AnalogsCriteria.size()>0);
    wxASSERT(m_AnalogsDates.size()>0);
    wxASSERT(m_AnalogsValuesGross.size()>0);
    wxASSERT(m_StationsLon.size()>0);
    wxASSERT(m_StationsLat.size()>0);
    wxASSERT(m_StationsLocCoordX.size()>0);
    wxASSERT(m_StationsLocCoordY.size()>0);

    if (m_HasReferenceValues)
    {
        wxASSERT(m_ReferenceAxis.size()>0);
        wxASSERT(m_ReferenceValues.cols()>0);
        wxASSERT(m_ReferenceValues.rows()>0);
    }

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
    size_t Nreferenceaxis = m_ReferenceAxis.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Set general attributes
    ncFile.PutAtt("version", &m_FileVersion);
    int dataParameter = (int)m_PredictandParameter;
    ncFile.PutAtt("predictand_parameter", &dataParameter);
    int dataTemporalResolution = (int)m_PredictandTemporalResolution;
    ncFile.PutAtt("predictand_temporal_resolution", &dataTemporalResolution);
    int dataSpatialAggregation = (int)m_PredictandSpatialAggregation;
    ncFile.PutAtt("predictand_spatial_aggregation", &dataSpatialAggregation);
    ncFile.PutAtt("predictand_dataset_id", m_PredictandDatasetId);
    ncFile.PutAtt("predictand_database", m_PredictandDatabase);
    ncFile.PutAtt("predictand_station_ids", GetPredictandStationIdsString());
    ncFile.PutAtt("method_id", m_MethodId);
    ncFile.PutAtt("method_id_display", m_MethodIdDisplay);
    ncFile.PutAtt("specific_tag", m_SpecificTag);
    ncFile.PutAtt("specific_tag_display", m_SpecificTagDisplay);
    ncFile.PutAtt("description", m_Description);
    ncFile.PutAtt("date_processed", &m_DateProcessed);
    ncFile.PutAtt("lead_time_origin", &m_LeadTimeOrigin);
    short hasReferenceValues = 0;
    if (m_HasReferenceValues)
    {
        hasReferenceValues = 1;
    }
    ncFile.PutAtt("has_reference_values", &hasReferenceValues);

    // Define dimensions. No unlimited dimension.
    ncFile.DefDim("lead_time", Nleadtime);
    ncFile.DefDim("analogs_tot", Nanalogstot);
    ncFile.DefDim("stations", Nstations);
    if (m_HasReferenceValues)
    {
        ncFile.DefDim("reference_axis", Nreferenceaxis);
    }

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNamesLeadTime;
    DimNamesLeadTime.push_back("lead_time");
    VectorStdString DimNamesAnalogsTot;
    DimNamesAnalogsTot.push_back("analogs_tot");
    VectorStdString DimNamesStations;
    DimNamesStations.push_back("stations");
    VectorStdString DimNamesAnalogsStations;
    DimNamesAnalogsStations.push_back("stations");
    DimNamesAnalogsStations.push_back("analogs_tot");
    VectorStdString DimNameReferenceAxis;
    VectorStdString DimNameReferenceValues;
    if (m_HasReferenceValues)
    {
        DimNameReferenceAxis.push_back("reference_axis");
        DimNameReferenceValues.push_back("stations");
        DimNameReferenceValues.push_back("reference_axis");
    }

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("target_dates", NC_FLOAT, 1, DimNamesLeadTime);
    ncFile.DefVar("analogs_nb", NC_INT, 1, DimNamesLeadTime);
    ncFile.DefVar("stations_names", NC_STRING, 1, DimNamesStations);
    ncFile.DefVar("stations_ids", NC_INT, 1, DimNamesStations);
    ncFile.DefVar("stations_heights", NC_FLOAT, 1, DimNamesStations);
    ncFile.DefVar("lon", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("lat", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("loc_coord_x", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("loc_coord_y", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("analogs_criteria", NC_FLOAT, 1, DimNamesAnalogsTot);
    ncFile.DefVar("analogs_dates", NC_FLOAT, 1, DimNamesAnalogsTot);
    ncFile.DefVar("analogs_values_gross", NC_FLOAT, 2, DimNamesAnalogsStations);
    ncFile.DefVarDeflate("analogs_values_gross");
    if (m_HasReferenceValues)
    {
        ncFile.DefVar("reference_axis", NC_FLOAT, 1, DimNameReferenceAxis);
        ncFile.DefVar("reference_values", NC_FLOAT, 2, DimNameReferenceValues);
    }

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefStationsIdsAttributes(ncFile);
    DefAnalogsNbAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesGrossAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

    ncFile.PutAtt("long_name", "Station names", "stations_names");
    ncFile.PutAtt("var_desc", "Name of the weather stations", "stations_names");
    ncFile.PutAtt("long_name", "Station heights", "stations_heights");
    ncFile.PutAtt("var_desc", "Altitude of the weather stations", "stations_heights");
    ncFile.PutAtt("units", "m", "stations_heights");
    ncFile.PutAtt("long_name", "Longitude", "lon");
    ncFile.PutAtt("var_desc", "Longitudes of the stations positions", "lon");
    ncFile.PutAtt("units", "degrees", "lon");
    ncFile.PutAtt("long_name", "Latitude", "lat");
    ncFile.PutAtt("var_desc", "Latitudes of the stations positions", "lat");
    ncFile.PutAtt("units", "degrees", "lat");
    ncFile.PutAtt("long_name", "Local coordinate X", "loc_coord_x");
    ncFile.PutAtt("var_desc", "Local coordinate for the X axis (west-east)", "loc_coord_x");
    ncFile.PutAtt("units", "m", "loc_coord_x");
    ncFile.PutAtt("long_name", "Local coordinate Y", "loc_coord_y");
    ncFile.PutAtt("var_desc", "Local coordinate for the Y axis (west-east)", "loc_coord_y");
    ncFile.PutAtt("units", "m", "loc_coord_y");
    if (m_HasReferenceValues)
    {
        ncFile.PutAtt("long_name", "Reference axis", "reference_axis");
        ncFile.PutAtt("var_desc", "Reference axis", "reference_axis");
        ncFile.PutAtt("long_name", "Reference values", "reference_values");
        ncFile.PutAtt("var_desc", "Reference values", "reference_values");
    }

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

    // Set the matrices in vectors
    VectorFloat analogsCriteria(Nanalogstot);
    VectorFloat analogsDates(Nanalogstot);
    VectorFloat analogsValuesGross(Nanalogstot*Nstations);
    
    int ind = 0;
    for (unsigned int i_time=0; i_time<Nleadtime; i_time++)
    {
        for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
        {
            analogsCriteria[ind] = m_AnalogsCriteria[i_time][i_analog];
            analogsDates[ind] = m_AnalogsDates[i_time][i_analog];
            ind++;
        }
    }

    int indVal = 0;
    for (unsigned int i_station=0; i_station<Nstations; i_station++)
    {
        for (unsigned int i_time=0; i_time<Nleadtime; i_time++)
        {
            for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
            {
                analogsValuesGross[indVal] = m_AnalogsValuesGross[i_time](i_station, i_analog);
                indVal++;
            }
        }
    }

    // Write data
    ncFile.PutVarArray("target_dates", startLeadTime, countLeadTime, &m_TargetDates[0]);
    ncFile.PutVarArray("analogs_nb", startLeadTime, countLeadTime, &m_AnalogsNb[0]);
    ncFile.PutVarArray("stations_names", startStations, countStations, &m_StationsNames[0], Nstations);
    ncFile.PutVarArray("stations_ids", startStations, countStations, &m_StationsIds[0]);
    ncFile.PutVarArray("stations_heights", startStations, countStations, &m_StationsHeights[0]);
    ncFile.PutVarArray("lon", startStations, countStations, &m_StationsLon(0));
    ncFile.PutVarArray("lat", startStations, countStations, &m_StationsLat(0));
    ncFile.PutVarArray("loc_coord_x", startStations, countStations, &m_StationsLocCoordX(0));
    ncFile.PutVarArray("loc_coord_y", startStations, countStations, &m_StationsLocCoordY(0));
    ncFile.PutVarArray("analogs_criteria", startAnalogsTot, countAnalogsTot, &analogsCriteria[0]);
    ncFile.PutVarArray("analogs_dates", startAnalogsTot, countAnalogsTot, &analogsDates[0]);
    ncFile.PutVarArray("analogs_values_gross", startAnalogsStations, countAnalogsStations, &analogsValuesGross[0]);
    if (m_HasReferenceValues)
    {
        size_t startReferenceAxis[] = {0};
        size_t countReferenceAxis[] = {Nreferenceaxis};
        size_t startReferenceValues[] = {0, 0};
        size_t countReferenceValues[] = {Nstations, Nreferenceaxis};
        ncFile.PutVarArray("reference_axis", startReferenceAxis, countReferenceAxis, &m_ReferenceAxis(0));
        ncFile.PutVarArray("reference_values", startReferenceValues, countReferenceValues, &m_ReferenceValues(0,0));
    }

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

    // Get global attributes
    float version = ncFile.GetAttFloat("version");
    if (version>1.5f)
    {
        asLogError(wxString::Format(_("The forecast file was made with more recent version of AtmoSwing (file version %.1f). It cannot be opened here."), version));
        return false;
    }

    if (asTools::IsNaN(version) || version<1.1)
    {
        asLogWarning(_("The forecast file was made with an older version of AtmoSwing."));
        m_PredictandParameter = Precipitation;
        m_PredictandTemporalResolution = Daily;
        m_PredictandSpatialAggregation = Station;
        m_PredictandDatasetId = "MeteoSwiss-Rhone";
        m_MethodId = ncFile.GetAttString("modelName");
        m_MethodIdDisplay = ncFile.GetAttString("modelName");
        m_SpecificTag = wxEmptyString;
        m_SpecificTagDisplay = wxEmptyString;
        m_Description = wxEmptyString;
        m_DateProcessed = ncFile.GetAttDouble("dateProcessed");
        m_LeadTimeOrigin = ncFile.GetAttDouble("leadTimeOrigin");
        m_HasReferenceValues = true;
    }
    else
    {
        if(version<1.5)
        {
            m_MethodId = ncFile.GetAttString("model_name");
            m_MethodIdDisplay = ncFile.GetAttString("model_name");
            m_SpecificTag = wxEmptyString;
            m_SpecificTagDisplay = wxEmptyString;
            m_Description = wxEmptyString;
        }
        else
        {
            m_MethodId = ncFile.GetAttString("method_id");
            m_MethodIdDisplay = ncFile.GetAttString("method_id_display");
            m_SpecificTag = ncFile.GetAttString("specific_tag");
            m_SpecificTagDisplay = ncFile.GetAttString("specific_tag_display");
            m_Description = ncFile.GetAttString("description");
        }

        m_PredictandParameter = (DataParameter)ncFile.GetAttInt("predictand_parameter");
        m_PredictandTemporalResolution = (DataTemporalResolution)ncFile.GetAttInt("predictand_temporal_resolution");
        m_PredictandSpatialAggregation = (DataSpatialAggregation)ncFile.GetAttInt("predictand_spatial_aggregation");
        m_PredictandDatasetId = ncFile.GetAttString("predictand_dataset_id");

        if(version>=1.5)
        {
            m_PredictandDatabase = ncFile.GetAttString("predictand_database");
            SetPredictandStationIds(ncFile.GetAttString("predictand_station_ids"));
        }

        m_DateProcessed = ncFile.GetAttDouble("date_processed");
        m_LeadTimeOrigin = ncFile.GetAttDouble("lead_time_origin");
        m_HasReferenceValues = false;
        if (ncFile.GetAttShort("has_reference_values") == 1)
        {
            m_HasReferenceValues = true;
        }
    }

    // Get the elements size
    int Nleadtime;
    int Nanalogstot; 
    int Nstations;
    if (asTools::IsNaN(version) || version<1.1)
    {
        Nleadtime = ncFile.GetDimLength("leadtime");
        Nanalogstot = ncFile.GetDimLength("analogstot");
        Nstations = ncFile.GetDimLength("stations");
    }
    else
    {
        Nleadtime = ncFile.GetDimLength("lead_time");
        Nanalogstot = ncFile.GetDimLength("analogs_tot");
        Nstations = ncFile.GetDimLength("stations");
    }

    // Get lead time data
    m_TargetDates.resize( Nleadtime );
    m_AnalogsNb.resize( Nleadtime );
    m_StationsNames.resize( Nstations );
    m_StationsIds.resize( Nstations );
    m_StationsHeights.resize( Nstations );
    m_StationsLon.resize( Nstations );
    m_StationsLat.resize( Nstations );
    m_StationsLocCoordX.resize( Nstations );
    m_StationsLocCoordY.resize( Nstations );

    if (asTools::IsNaN(version) || version<1.1)
    {
        ncFile.GetVar("targetdates", &m_TargetDates[0]);
        ncFile.GetVar("analogsnb", &m_AnalogsNb[0]);
        ncFile.GetVar("stationsnames", &m_StationsNames[0], Nstations);
        ncFile.GetVar("stationsids", &m_StationsIds[0]);
        ncFile.GetVar("stationsheights", &m_StationsHeights[0]);
        ncFile.GetVar("loccoordu", &m_StationsLocCoordX[0]);
        ncFile.GetVar("loccoordv", &m_StationsLocCoordY[0]);
    }
    else
    {
        ncFile.GetVar("target_dates", &m_TargetDates[0]);
        ncFile.GetVar("analogs_nb", &m_AnalogsNb[0]);
        ncFile.GetVar("stations_names", &m_StationsNames[0], Nstations);
        ncFile.GetVar("stations_ids", &m_StationsIds[0]);
        ncFile.GetVar("stations_heights", &m_StationsHeights[0]);

        if (version<1.4) {
            ncFile.GetVar("loc_coord_u", &m_StationsLocCoordX[0]);
            ncFile.GetVar("loc_coord_v", &m_StationsLocCoordY[0]);
        }
        else {
            ncFile.GetVar("loc_coord_x", &m_StationsLocCoordX[0]);
            ncFile.GetVar("loc_coord_y", &m_StationsLocCoordY[0]);
        }
    }
    
    ncFile.GetVar("lon", &m_StationsLon[0]);
    ncFile.GetVar("lat", &m_StationsLat[0]);

    // Get return periods properties
    if (m_HasReferenceValues)
    {
        if (asTools::IsNaN(version) || version<1.1)
        {
            int referenceAxisLength = ncFile.GetDimLength("returnperiods");
            m_ReferenceAxis.resize( referenceAxisLength );
            ncFile.GetVar("returnperiods", &m_ReferenceAxis[0]);
            size_t startReferenceValues[2] = {0, 0};
            size_t countReferenceValues[2] = {size_t(referenceAxisLength), size_t(Nstations)};
            m_ReferenceValues.resize( Nstations, referenceAxisLength );
            ncFile.GetVarArray("dailyprecipitationsforreturnperiods", startReferenceValues, countReferenceValues, &m_ReferenceValues(0,0));
        }
        else
        {
            int referenceAxisLength = ncFile.GetDimLength("reference_axis");
            m_ReferenceAxis.resize( referenceAxisLength );
            ncFile.GetVar("reference_axis", &m_ReferenceAxis[0]);
            size_t startReferenceValues[2] = {0, 0};
            size_t countReferenceValues[2] = {0, 0};
            if (version==1.1f)
            {
                countReferenceValues[0] = size_t(referenceAxisLength);
                countReferenceValues[1] = size_t(Nstations);
            }
            else
            {
                countReferenceValues[0] = size_t(Nstations);
                countReferenceValues[1] = size_t(referenceAxisLength);
            }
            m_ReferenceValues.resize( Nstations, referenceAxisLength );
            ncFile.GetVarArray("reference_values", startReferenceValues, countReferenceValues, &m_ReferenceValues(0,0));
        }
    }

    // Create vectors for matrices data
    VectorFloat analogsCriteria(Nanalogstot);
    VectorFloat analogsDates(Nanalogstot);
    VectorFloat analogsValuesGross(Nanalogstot*Nstations);

    // Get data
    size_t IndexStart1D[] = {0};
    size_t IndexCount1D[] = {size_t(Nanalogstot)};
    size_t IndexStart2D[] = {0,0};
    size_t IndexCount2D[] = {size_t(Nstations), size_t(Nanalogstot)};
    if (asTools::IsNaN(version) || version<1.1)
    {
        ncFile.GetVarArray("analogscriteria", IndexStart1D, IndexCount1D, &analogsCriteria[0]);
        ncFile.GetVarArray("analogsdates", IndexStart1D, IndexCount1D, &analogsDates[0]);
        ncFile.GetVarArray("analogsvaluesgross", IndexStart2D, IndexCount2D, &analogsValuesGross[0]);
    }
    else
    {
        ncFile.GetVarArray("analogs_criteria", IndexStart1D, IndexCount1D, &analogsCriteria[0]);
        ncFile.GetVarArray("analogs_dates", IndexStart1D, IndexCount1D, &analogsDates[0]);
        ncFile.GetVarArray("analogs_values_gross", IndexStart2D, IndexCount2D, &analogsValuesGross[0]);
    }

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    // Set data into the matrices
    int ind = 0;
    for (int i_time=0; i_time<Nleadtime; i_time++)
    {
        Array1DFloat analogsCriteriaLeadTime(m_AnalogsNb[i_time]);
        Array1DFloat analogsDatesLeadTime(m_AnalogsNb[i_time]);

        for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
        {
            analogsCriteriaLeadTime(i_analog) = analogsCriteria[ind];
            analogsDatesLeadTime(i_analog) = analogsDates[ind];
            ind++;
        }

        m_AnalogsCriteria.push_back( analogsCriteriaLeadTime );
        m_AnalogsDates.push_back( analogsDatesLeadTime );
    }
    
    int indVal = 0;
    if (asTools::IsNaN(version) || version<1.1)
    {
        for (int i_time=0; i_time<Nleadtime; i_time++)
        {
            Array2DFloat analogsValuesGrossLeadTime(Nstations, m_AnalogsNb[i_time]);

            for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
            {
                for (int i_station=0; i_station<Nstations; i_station++)
                {
                    analogsValuesGrossLeadTime(i_station, i_analog) = analogsValuesGross[indVal];
                    indVal++;
                }
            }

            m_AnalogsValuesGross.push_back( analogsValuesGrossLeadTime );
        }
    }
    else
    {
        // Create containers
        for (int i_time=0; i_time<Nleadtime; i_time++)
        {
            Array2DFloat analogsValuesGrossLeadTime(Nstations, m_AnalogsNb[i_time]);
            analogsValuesGrossLeadTime.fill(NaNFloat);
            m_AnalogsValuesGross.push_back( analogsValuesGrossLeadTime );
        }

        for (int i_station=0; i_station<Nstations; i_station++)
        {
            for (int i_time=0; i_time<Nleadtime; i_time++)
            {
                for (int i_analog=0; i_analog<m_AnalogsNb[i_time]; i_analog++)
                {
                    m_AnalogsValuesGross[i_time](i_station, i_analog) = analogsValuesGross[indVal];
                    indVal++;
                }
            }
        }
    }

    wxASSERT(!m_FilePath.IsEmpty());
    wxASSERT(!m_PredictandDatasetId.IsEmpty());
    wxASSERT(m_TargetDates.size()>0);
    wxASSERT(m_AnalogsNb.size()>0);
    wxASSERT(m_StationsIds.size()>0);
    wxASSERT(m_StationsNames.size()>0);
    wxASSERT(m_StationsHeights.size()>0);
    wxASSERT(m_AnalogsCriteria.size()>0);
    wxASSERT(m_AnalogsDates.size()>0);
    wxASSERT(m_AnalogsValuesGross.size()>0);
    wxASSERT(m_StationsLon.size()>0);
    wxASSERT(m_StationsLat.size()>0);
    wxASSERT(m_StationsLocCoordX.size()>0);
    wxASSERT(m_StationsLocCoordY.size()>0);
    if (m_HasReferenceValues)
    {
        wxASSERT(m_ReferenceAxis.size()>0);
        wxASSERT(m_ReferenceValues.cols()>0);
        wxASSERT(m_ReferenceValues.rows()>0);
    }

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
        wxString label;
        if(!asTools::IsNaN(m_StationsHeights[i]))
        {
            label = wxString::Format("%s (%4.0fm)", m_StationsNames[i].c_str(), m_StationsHeights[i]);
        }
        else
        {
            label = wxString::Format("%s", m_StationsNames[i].c_str());
        }
        stationsNames.Add(label);
    }
    return stationsNames;
}

wxString asResultsAnalogsForecast::GetStationNameAndHeight(int i_stat)
{
    wxString stationName;
    if(!asTools::IsNaN(m_StationsHeights[i_stat]))
    {
        stationName = wxString::Format("%s (%4.0fm)", m_StationsNames[i_stat].c_str(), m_StationsHeights[i_stat]);
    }
    else
    {
        stationName = wxString::Format("%s", m_StationsNames[i_stat].c_str());
    }
    return stationName;
}

wxString asResultsAnalogsForecast::GetPredictandStationIdsString()
{
    wxString Ids;

    for (int i=0; i<m_PredictandStationIds.size(); i++)
    {
        Ids << m_PredictandStationIds[i];

        if (i<m_PredictandStationIds.size()-1)
        {
            Ids.Append(",");
        }
    }

    return Ids;
}

void asResultsAnalogsForecast::SetPredictandStationIds(wxString val)
{
    wxStringTokenizer tokenizer(val, ":,; ");
    while ( tokenizer.HasMoreTokens() )
    {
        wxString token = tokenizer.GetNextToken();
        long stationId;
        token.ToLong(&stationId);
        m_PredictandStationIds.push_back(stationId);
    }
}

bool asResultsAnalogsForecast::IsCompatibleWith(asResultsAnalogsForecast * otherForecast)
{
    bool compatible = true;

    if (!m_MethodId.IsSameAs(otherForecast->GetMethodId(), false)) compatible = false;
    if (m_PredictandParameter != otherForecast->GetPredictandParameter()) compatible = false;
    if (m_PredictandTemporalResolution != otherForecast->GetPredictandTemporalResolution()) compatible = false;
    if (m_PredictandSpatialAggregation != otherForecast->GetPredictandSpatialAggregation()) compatible = false;
    if (!m_PredictandDatasetId.IsSameAs(otherForecast->GetPredictandDatasetId(), false)) compatible = false;
    if (!m_PredictandDatabase.IsSameAs(otherForecast->GetPredictandDatabase(), false)) compatible = false;
    if (m_HasReferenceValues != otherForecast->HasReferenceValues()) compatible = false;
    if (m_LeadTimeOrigin != otherForecast->GetLeadTimeOrigin()) compatible = false;

    Array1DFloat targetDates = otherForecast->GetTargetDates();
    if (m_TargetDates.size() != targetDates.size()) {
        compatible = false;
    }
    else {
        for (int i=0; i<m_TargetDates.size(); i++) {
            if (m_TargetDates[i] != targetDates[i]) compatible = false;
        }
    }

    Array1DInt stationsIds = otherForecast->GetStationsIds();
    if (m_StationsIds.size() != stationsIds.size()) {
        compatible = false;
    }
    else {
        for (int i=0; i<m_StationsIds.size(); i++) {
            if (m_StationsIds[i] != stationsIds[i]) compatible = false;
        }
    }

    Array1DFloat referenceAxis = otherForecast->GetReferenceAxis();
    if (m_ReferenceAxis.size() != referenceAxis.size()) {
        compatible = false;
    }
    else {
        for (int i=0; i<m_ReferenceAxis.size(); i++) {
            if (m_ReferenceAxis[i] != referenceAxis[i]) compatible = false;
        }
    }

    if (!compatible)
    {
        asLogError(wxString::Format(_("The forecasts \"%s\" and \"%s\" are not compatible"), m_SpecificTagDisplay.c_str(), otherForecast->GetSpecificTagDisplay().c_str()));
        return false;
    }

    return true;
}

bool asResultsAnalogsForecast::IsSpecificForStation(int stationId)
{
    for (int i=0; i<m_PredictandStationIds.size(); i++)
    {
        if (m_PredictandStationIds[i]==stationId)
        {
            return true;
        }
    }
    return false;
}

int asResultsAnalogsForecast::GetStationRowFromId(int stationId)
{
    for (int i=0; i<m_StationsIds.size(); i++)
    {
        if (m_StationsIds[i]==stationId)
        {
            return i;
        }
    }

    wxFAIL;
    asLogError(wxString::Format("The station ID %d was not found in the forecast results.", stationId));
    return -1;
}