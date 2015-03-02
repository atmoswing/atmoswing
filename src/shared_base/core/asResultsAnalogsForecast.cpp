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
    m_filePath = wxEmptyString;
    m_hasReferenceValues = false;
    m_leadTimeOrigin = 0.0;

    // Default values for former versions
    m_predictandParameter = Precipitation;
    m_predictandTemporalResolution = Daily;
    m_predictandSpatialAggregation = Station;
    m_predictandDatasetId = wxEmptyString;
    m_predictandDatabase = wxEmptyString;
}

asResultsAnalogsForecast::~asResultsAnalogsForecast()
{
    //dtor
}

void asResultsAnalogsForecast::Init(asParametersForecast &params, double leadTimeOrigin)
{
    // Resize to 0 to avoid keeping old results
    m_targetDates.resize(0);
    m_stationNames.resize(0);
    m_stationOfficialIds.resize(0);
    m_stationIds.resize(0);
    m_stationHeights.resize(0);
    m_analogsNb.resize(0);
    m_analogsCriteria.resize(0);
    m_analogsDates.resize(0);
    m_analogsValuesGross.resize(0);
    m_stationXCoords.resize(0);
    m_stationYCoords.resize(0);
    m_referenceAxis.resize(0);
    m_referenceValues.resize(0,0);

    m_methodId = params.GetMethodId();
    m_methodIdDisplay = params.GetMethodIdDisplay();
    m_specificTag = params.GetSpecificTag();
    m_specificTagDisplay = params.GetSpecificTagDisplay();
    m_description = params.GetDescription();
    m_predictandDatabase = params.GetPredictandDatabase();
    m_predictandStationIds = params.GetPredictandStationIds();

    m_leadTimeOrigin = leadTimeOrigin;
    m_dateProcessed = asTime::NowMJD(asUTM);

    // Set the analogs number
    m_analogsNb.resize(params.GetLeadTimeNb());
    for (int i=0; i<params.GetLeadTimeNb(); i++)
    {
        m_analogsNb[i] = params.GetAnalogsNumberLeadTime(m_currentStep, i);
    }

    BuildFileName();
}

void asResultsAnalogsForecast::BuildFileName()
{
    wxASSERT(!m_forecastsDirectory.IsEmpty());

    if(m_methodId.IsEmpty() || m_specificTag.IsEmpty())
    {
        asLogError(_("The provided ID or the tag is empty, which isn't allowed !"));
    }

    // Base directory
    m_filePath = m_forecastsDirectory;
    m_filePath.Append(DS);

    // Directory
    wxString dirstructure = "YYYY";
    dirstructure.Append(DS);
    dirstructure.Append("MM");
    dirstructure.Append(DS);
    dirstructure.Append("DD");
    wxString directory = asTime::GetStringTime(m_leadTimeOrigin, dirstructure);
    m_filePath.Append(directory);
    m_filePath.Append(DS);

    // Filename
    wxString forecastname = m_methodId + '.' + m_specificTag;
    wxString nowstr = asTime::GetStringTime(m_leadTimeOrigin, "YYYYMMDDhh");
    wxString ext = "asff";
    wxString filename = wxString::Format("%s.%s.%s",nowstr.c_str(),forecastname.c_str(),ext.c_str());
    m_filePath.Append(filename);
}

bool asResultsAnalogsForecast::Save(const wxString &AlternateFilePath)
{
    wxASSERT(!m_filePath.IsEmpty());
    wxASSERT(m_targetDates.size()>0);
    wxASSERT(m_analogsNb.size()>0);
    wxASSERT(m_stationNames.size()>0);
    wxASSERT(m_stationOfficialIds.size()>0);
    wxASSERT(m_stationHeights.size()>0);
    wxASSERT(m_stationIds.size()>0);
    wxASSERT(m_analogsCriteria.size()>0);
    wxASSERT(m_analogsDates.size()>0);
    wxASSERT(m_analogsValuesGross.size()>0);
    wxASSERT(m_stationXCoords.size()>0);
    wxASSERT(m_stationYCoords.size()>0);

    if (m_hasReferenceValues)
    {
        wxASSERT(m_referenceAxis.size()>0);
        wxASSERT(m_referenceValues.cols()>0);
        wxASSERT(m_referenceValues.rows()>0);
    }

    wxString message = _("Saving forecast file: ") + m_filePath;
    asLogMessage(message);

    // Get the file path
    wxString ResultsFile;
    if (AlternateFilePath.IsEmpty())
    {
        ResultsFile = m_filePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    // Get the elements size
    size_t Nleadtime = m_targetDates.size();
    size_t Nanalogstot = m_analogsNb.sum();
    size_t Nstations = m_stationIds.size();
    size_t Nreferenceaxis = m_referenceAxis.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Set general attributes
    ncFile.PutAtt("version", &m_fileVersion);
    int dataParameter = (int)m_predictandParameter;
    ncFile.PutAtt("predictand_parameter", &dataParameter);
    int dataTemporalResolution = (int)m_predictandTemporalResolution;
    ncFile.PutAtt("predictand_temporal_resolution", &dataTemporalResolution);
    int dataSpatialAggregation = (int)m_predictandSpatialAggregation;
    ncFile.PutAtt("predictand_spatial_aggregation", &dataSpatialAggregation);
    ncFile.PutAtt("predictand_dataset_id", m_predictandDatasetId);
    ncFile.PutAtt("predictand_database", m_predictandDatabase);
    ncFile.PutAtt("predictand_station_ids", GetPredictandStationIdsString());
    ncFile.PutAtt("method_id", m_methodId);
    ncFile.PutAtt("method_id_display", m_methodIdDisplay);
    ncFile.PutAtt("specific_tag", m_specificTag);
    ncFile.PutAtt("specific_tag_display", m_specificTagDisplay);
    ncFile.PutAtt("description", m_description);
    ncFile.PutAtt("date_processed", &m_dateProcessed);
    ncFile.PutAtt("lead_time_origin", &m_leadTimeOrigin);
    short hasReferenceValues = 0;
    if (m_hasReferenceValues)
    {
        hasReferenceValues = 1;
    }
    ncFile.PutAtt("has_reference_values", &hasReferenceValues);

    // Define dimensions. No unlimited dimension.
    ncFile.DefDim("lead_time", Nleadtime);
    ncFile.DefDim("analogs_tot", Nanalogstot);
    ncFile.DefDim("stations", Nstations);
    if (m_hasReferenceValues)
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
    if (m_hasReferenceValues)
    {
        DimNameReferenceAxis.push_back("reference_axis");
        DimNameReferenceValues.push_back("stations");
        DimNameReferenceValues.push_back("reference_axis");
    }

    // Define variables: the analogcriteria and the corresponding dates
    ncFile.DefVar("target_dates", NC_FLOAT, 1, DimNamesLeadTime);
    ncFile.DefVar("analogs_nb", NC_INT, 1, DimNamesLeadTime);
    ncFile.DefVar("station_names", NC_STRING, 1, DimNamesStations);
    ncFile.DefVar("station_ids", NC_INT, 1, DimNamesStations);
    ncFile.DefVar("station_official_ids", NC_STRING, 1, DimNamesStations);
    ncFile.DefVar("station_heights", NC_FLOAT, 1, DimNamesStations);
    ncFile.DefVar("station_x_coords", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("station_y_coords", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("analog_criteria", NC_FLOAT, 1, DimNamesAnalogsTot);
    ncFile.DefVar("analog_dates", NC_FLOAT, 1, DimNamesAnalogsTot);
    ncFile.DefVar("analog_values", NC_FLOAT, 2, DimNamesAnalogsStations);
    ncFile.DefVarDeflate("analog_values");
    if (m_hasReferenceValues)
    {
        ncFile.DefVar("reference_axis", NC_FLOAT, 1, DimNameReferenceAxis);
        ncFile.DefVar("reference_values", NC_FLOAT, 2, DimNameReferenceValues);
    }

    // Put attributes
    DefTargetDatesAttributes(ncFile);
    DefStationIdsAttributes(ncFile);
    DefStationOfficialIdsAttributes(ncFile);
    DefAnalogsNbAttributes(ncFile);
    DefAnalogsCriteriaAttributes(ncFile);
    DefAnalogsValuesAttributes(ncFile);
    DefAnalogsDatesAttributes(ncFile);

    ncFile.PutAtt("long_name", "Station names", "station_names");
    ncFile.PutAtt("var_desc", "Name of the weather stations", "station_names");
    ncFile.PutAtt("long_name", "Station heights", "station_heights");
    ncFile.PutAtt("var_desc", "Altitude of the weather stations", "station_heights");
    ncFile.PutAtt("units", "m", "station_heights");
    ncFile.PutAtt("long_name", "X coordinate", "station_x_coords");
    ncFile.PutAtt("var_desc", "X coordinate (west-east)", "station_x_coords");
    ncFile.PutAtt("units", "m", "station_x_coords");
    ncFile.PutAtt("long_name", "Y coordinate", "station_y_coords");
    ncFile.PutAtt("var_desc", "Y coordinate (west-east)", "station_y_coords");
    ncFile.PutAtt("units", "m", "station_y_coords");
    if (m_hasReferenceValues)
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
        for (int i_analog=0; i_analog<m_analogsNb[i_time]; i_analog++)
        {
            analogsCriteria[ind] = m_analogsCriteria[i_time][i_analog];
            analogsDates[ind] = m_analogsDates[i_time][i_analog];
            ind++;
        }
    }

    int indVal = 0;
    for (unsigned int i_station=0; i_station<Nstations; i_station++)
    {
        for (unsigned int i_time=0; i_time<Nleadtime; i_time++)
        {
            for (int i_analog=0; i_analog<m_analogsNb[i_time]; i_analog++)
            {
                analogsValuesGross[indVal] = m_analogsValuesGross[i_time](i_station, i_analog);
                indVal++;
            }
        }
    }

    // Write data
    ncFile.PutVarArray("target_dates", startLeadTime, countLeadTime, &m_targetDates[0]);
    ncFile.PutVarArray("analogs_nb", startLeadTime, countLeadTime, &m_analogsNb[0]);
    ncFile.PutVarArray("station_names", startStations, countStations, &m_stationNames[0], Nstations);
    ncFile.PutVarArray("station_official_ids", startStations, countStations, &m_stationOfficialIds[0], Nstations);
    ncFile.PutVarArray("station_ids", startStations, countStations, &m_stationIds[0]);
    ncFile.PutVarArray("station_heights", startStations, countStations, &m_stationHeights[0]);
    ncFile.PutVarArray("station_x_coords", startStations, countStations, &m_stationXCoords(0));
    ncFile.PutVarArray("station_y_coords", startStations, countStations, &m_stationYCoords(0));
    ncFile.PutVarArray("analog_criteria", startAnalogsTot, countAnalogsTot, &analogsCriteria[0]);
    ncFile.PutVarArray("analog_dates", startAnalogsTot, countAnalogsTot, &analogsDates[0]);
    ncFile.PutVarArray("analog_values", startAnalogsStations, countAnalogsStations, &analogsValuesGross[0]);
    if (m_hasReferenceValues)
    {
        size_t startReferenceAxis[] = {0};
        size_t countReferenceAxis[] = {Nreferenceaxis};
        size_t startReferenceValues[] = {0, 0};
        size_t countReferenceValues[] = {Nstations, Nreferenceaxis};
        ncFile.PutVarArray("reference_axis", startReferenceAxis, countReferenceAxis, &m_referenceAxis(0));
        ncFile.PutVarArray("reference_values", startReferenceValues, countReferenceValues, &m_referenceValues(0,0));
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
        m_filePath = AlternateFilePath;
    }

    // If we don't want to save or the file doesn't exist
    if(!Exists()) return false;
    if(m_currentStep!=0) return false;

    ThreadsManager().CritSectionNetCDF().Enter();

    // Open the NetCDF file
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::ReadOnly);
    if(!ncFile.Open()) return false;

    // Get global attributes
    float version = ncFile.GetAttFloat("version");
    if (version>1.6f)
    {
        asLogError(wxString::Format(_("The forecast file was made with more recent version of AtmoSwing (file version %.1f). It cannot be opened here."), version));
        return false;
    }

    if (asTools::IsNaN(version) || version<=1.0)
    {
        asLogWarning(_("The forecast file was made with an older version of AtmoSwing."));
        m_predictandParameter = Precipitation;
        m_predictandTemporalResolution = Daily;
        m_predictandSpatialAggregation = Station;
        m_predictandDatasetId = "MeteoSwiss-Rhone";
        m_methodId = ncFile.GetAttString("modelName");
        m_methodIdDisplay = ncFile.GetAttString("modelName");
        m_specificTag = wxEmptyString;
        m_specificTagDisplay = wxEmptyString;
        m_description = wxEmptyString;
        m_dateProcessed = ncFile.GetAttDouble("dateProcessed");
        m_leadTimeOrigin = ncFile.GetAttDouble("leadTimeOrigin");
        m_hasReferenceValues = true;
    }
    else
    {
        if(version<=1.4)
        {
            m_methodId = ncFile.GetAttString("model_name");
            m_methodIdDisplay = ncFile.GetAttString("model_name");
            m_specificTag = wxEmptyString;
            m_specificTagDisplay = wxEmptyString;
            m_description = wxEmptyString;
        }
        else
        {
            m_methodId = ncFile.GetAttString("method_id");
            m_methodIdDisplay = ncFile.GetAttString("method_id_display");
            m_specificTag = ncFile.GetAttString("specific_tag");
            m_specificTagDisplay = ncFile.GetAttString("specific_tag_display");
            m_description = ncFile.GetAttString("description");
        }

        m_predictandParameter = (DataParameter)ncFile.GetAttInt("predictand_parameter");
        m_predictandTemporalResolution = (DataTemporalResolution)ncFile.GetAttInt("predictand_temporal_resolution");
        m_predictandSpatialAggregation = (DataSpatialAggregation)ncFile.GetAttInt("predictand_spatial_aggregation");
        m_predictandDatasetId = ncFile.GetAttString("predictand_dataset_id");

        if(version>=1.5)
        {
            m_predictandDatabase = ncFile.GetAttString("predictand_database");
            SetPredictandStationIds(ncFile.GetAttString("predictand_station_ids"));
        }

        m_dateProcessed = ncFile.GetAttDouble("date_processed");
        m_leadTimeOrigin = ncFile.GetAttDouble("lead_time_origin");
        m_hasReferenceValues = false;
        if (ncFile.GetAttShort("has_reference_values") == 1)
        {
            m_hasReferenceValues = true;
        }
    }

    // Get the elements size
    int Nleadtime;
    int Nanalogstot; 
    int Nstations;
    if (asTools::IsNaN(version) || version<=1.0)
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
    m_targetDates.resize( Nleadtime );
    m_analogsNb.resize( Nleadtime );
    m_stationNames.resize( Nstations );
    m_stationOfficialIds.resize( Nstations );
    m_stationIds.resize( Nstations );
    m_stationHeights.resize( Nstations );
    m_stationXCoords.resize( Nstations );
    m_stationYCoords.resize( Nstations );

    if (asTools::IsNaN(version) || version<=1.0)
    {
        ncFile.GetVar("targetdates", &m_targetDates[0]);
        ncFile.GetVar("analogsnb", &m_analogsNb[0]);
        ncFile.GetVar("stationsnames", &m_stationNames[0], Nstations);
        ncFile.GetVar("stationsids", &m_stationIds[0]);
        ncFile.GetVar("stationsheights", &m_stationHeights[0]);
        ncFile.GetVar("loccoordu", &m_stationXCoords[0]);
        ncFile.GetVar("loccoordv", &m_stationYCoords[0]);
    }
    else if (version<=1.3)
    {
        ncFile.GetVar("target_dates", &m_targetDates[0]);
        ncFile.GetVar("analogs_nb", &m_analogsNb[0]);
        ncFile.GetVar("stations_names", &m_stationNames[0], Nstations);
        ncFile.GetVar("stations_ids", &m_stationIds[0]);
        ncFile.GetVar("stations_heights", &m_stationHeights[0]);
        ncFile.GetVar("loc_coord_u", &m_stationXCoords[0]);
        ncFile.GetVar("loc_coord_v", &m_stationYCoords[0]);
    }
    else if (version<=1.5)
    {
        ncFile.GetVar("target_dates", &m_targetDates[0]);
        ncFile.GetVar("analogs_nb", &m_analogsNb[0]);
        ncFile.GetVar("stations_names", &m_stationNames[0], Nstations);
        ncFile.GetVar("stations_ids", &m_stationIds[0]);
        ncFile.GetVar("stations_heights", &m_stationHeights[0]);
        ncFile.GetVar("loc_coord_x", &m_stationXCoords[0]);
        ncFile.GetVar("loc_coord_y", &m_stationYCoords[0]);
    }
    else
        ncFile.GetVar("target_dates", &m_targetDates[0]);
        ncFile.GetVar("analogs_nb", &m_analogsNb[0]);
        ncFile.GetVar("station_names", &m_stationNames[0], Nstations);
        ncFile.GetVar("station_official_ids", &m_stationOfficialIds[0], Nstations);
        ncFile.GetVar("station_ids", &m_stationIds[0]);
        ncFile.GetVar("station_heights", &m_stationHeights[0]);
        ncFile.GetVar("station_x_coords", &m_stationXCoords[0]);
        ncFile.GetVar("station_y_coords", &m_stationYCoords[0]);
    }

    // Get return periods properties
    if (m_hasReferenceValues)
    {
        if (asTools::IsNaN(version) || version<=1.0)
        {
            int referenceAxisLength = ncFile.GetDimLength("returnperiods");
            m_referenceAxis.resize( referenceAxisLength );
            ncFile.GetVar("returnperiods", &m_referenceAxis[0]);
            size_t startReferenceValues[2] = {0, 0};
            size_t countReferenceValues[2] = {size_t(referenceAxisLength), size_t(Nstations)};
            m_referenceValues.resize( Nstations, referenceAxisLength );
            ncFile.GetVarArray("dailyprecipitationsforreturnperiods", startReferenceValues, countReferenceValues, &m_referenceValues(0,0));
        }
        else
        {
            int referenceAxisLength = ncFile.GetDimLength("reference_axis");
            m_referenceAxis.resize( referenceAxisLength );
            ncFile.GetVar("reference_axis", &m_referenceAxis[0]);
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
            m_referenceValues.resize( Nstations, referenceAxisLength );
            ncFile.GetVarArray("reference_values", startReferenceValues, countReferenceValues, &m_referenceValues(0,0));
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
    if (asTools::IsNaN(version) || version<=1.0)
    {
        ncFile.GetVarArray("analogscriteria", IndexStart1D, IndexCount1D, &analogsCriteria[0]);
        ncFile.GetVarArray("analogsdates", IndexStart1D, IndexCount1D, &analogsDates[0]);
        ncFile.GetVarArray("analogsvaluesgross", IndexStart2D, IndexCount2D, &analogsValuesGross[0]);
    }
    else if (version<=1.5)
    {
        ncFile.GetVarArray("analogs_criteria", IndexStart1D, IndexCount1D, &analogsCriteria[0]);
        ncFile.GetVarArray("analogs_dates", IndexStart1D, IndexCount1D, &analogsDates[0]);
        ncFile.GetVarArray("analogs_values_gross", IndexStart2D, IndexCount2D, &analogsValuesGross[0]);
    }
    else
    {
        ncFile.GetVarArray("analog_criteria", IndexStart1D, IndexCount1D, &analogsCriteria[0]);
        ncFile.GetVarArray("analog_dates", IndexStart1D, IndexCount1D, &analogsDates[0]);
        ncFile.GetVarArray("analog_values", IndexStart2D, IndexCount2D, &analogsValuesGross[0]);
    }

    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    // Set data into the matrices
    int ind = 0;
    for (int i_time=0; i_time<Nleadtime; i_time++)
    {
        Array1DFloat analogsCriteriaLeadTime(m_analogsNb[i_time]);
        Array1DFloat analogsDatesLeadTime(m_analogsNb[i_time]);

        for (int i_analog=0; i_analog<m_analogsNb[i_time]; i_analog++)
        {
            analogsCriteriaLeadTime(i_analog) = analogsCriteria[ind];
            analogsDatesLeadTime(i_analog) = analogsDates[ind];
            ind++;
        }

        m_analogsCriteria.push_back( analogsCriteriaLeadTime );
        m_analogsDates.push_back( analogsDatesLeadTime );
    }
    
    int indVal = 0;
    if (asTools::IsNaN(version) || version<=1.0)
    {
        for (int i_time=0; i_time<Nleadtime; i_time++)
        {
            Array2DFloat analogsValuesGrossLeadTime(Nstations, m_analogsNb[i_time]);

            for (int i_analog=0; i_analog<m_analogsNb[i_time]; i_analog++)
            {
                for (int i_station=0; i_station<Nstations; i_station++)
                {
                    analogsValuesGrossLeadTime(i_station, i_analog) = analogsValuesGross[indVal];
                    indVal++;
                }
            }

            m_analogsValuesGross.push_back( analogsValuesGrossLeadTime );
        }
    }
    else
    {
        // Create containers
        for (int i_time=0; i_time<Nleadtime; i_time++)
        {
            Array2DFloat analogsValuesGrossLeadTime(Nstations, m_analogsNb[i_time]);
            analogsValuesGrossLeadTime.fill(NaNFloat);
            m_analogsValuesGross.push_back( analogsValuesGrossLeadTime );
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
                for (int i_analog=0; i_analog<m_analogsNb[i_time]; i_analog++)
                {
                    m_analogsValuesGross[i_time](i_station, i_analog) = analogsValuesGross[indVal];
                    indVal++;
                }
            }
        }
    }

    wxASSERT(!m_filePath.IsEmpty());
    wxASSERT(!m_predictandDatasetId.IsEmpty());
    wxASSERT(m_targetDates.size()>0);
    wxASSERT(m_analogsNb.size()>0);
    wxASSERT(m_stationIds.size()>0);
    wxASSERT(m_stationNames.size()>0);
    wxASSERT(m_stationHeights.size()>0);
    wxASSERT(m_analogsCriteria.size()>0);
    wxASSERT(m_analogsDates.size()>0);
    wxASSERT(m_analogsValuesGross.size()>0);
    wxASSERT(m_stationXCoords.size()>0);
    wxASSERT(m_stationYCoords.size()>0);
    if (m_hasReferenceValues)
    {
        wxASSERT(m_referenceAxis.size()>0);
        wxASSERT(m_referenceValues.cols()>0);
        wxASSERT(m_referenceValues.rows()>0);
    }

    return true;
}

wxArrayString asResultsAnalogsForecast::GetStationNamesWxArrayString()
{
    wxArrayString stationsNames;
    for (unsigned int i=0; i<m_stationNames.size(); i++)
    {
        stationsNames.Add(m_stationNames[i]);
    }
    return stationsNames;
}

wxArrayString asResultsAnalogsForecast::GetStationNamesAndHeightsWxArrayString()
{
    wxArrayString stationsNames;
    for (unsigned int i=0; i<m_stationNames.size(); i++)
    {
        wxString label;
        if(!asTools::IsNaN(m_stationHeights[i]))
        {
            label = wxString::Format("%s (%4.0fm)", m_stationNames[i].c_str(), m_stationHeights[i]);
        }
        else
        {
            label = wxString::Format("%s", m_stationNames[i].c_str());
        }
        stationsNames.Add(label);
    }
    return stationsNames;
}

wxString asResultsAnalogsForecast::GetStationNameAndHeight(int i_stat)
{
    wxString stationName;
    if(!asTools::IsNaN(m_stationHeights[i_stat]))
    {
        stationName = wxString::Format("%s (%4.0fm)", m_stationNames[i_stat].c_str(), m_stationHeights[i_stat]);
    }
    else
    {
        stationName = wxString::Format("%s", m_stationNames[i_stat].c_str());
    }
    return stationName;
}

wxString asResultsAnalogsForecast::GetPredictandStationIdsString()
{
    wxString Ids;

    for (int i=0; i<m_predictandStationIds.size(); i++)
    {
        Ids << m_predictandStationIds[i];

        if (i<m_predictandStationIds.size()-1)
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
        m_predictandStationIds.push_back(stationId);
    }
}

bool asResultsAnalogsForecast::IsCompatibleWith(asResultsAnalogsForecast * otherForecast)
{
    bool compatible = true;

    if (!m_methodId.IsSameAs(otherForecast->GetMethodId(), false)) compatible = false;
    if (m_predictandParameter != otherForecast->GetPredictandParameter()) compatible = false;
    if (m_predictandTemporalResolution != otherForecast->GetPredictandTemporalResolution()) compatible = false;
    if (m_predictandSpatialAggregation != otherForecast->GetPredictandSpatialAggregation()) compatible = false;
    if (!m_predictandDatasetId.IsSameAs(otherForecast->GetPredictandDatasetId(), false)) compatible = false;
    if (!m_predictandDatabase.IsSameAs(otherForecast->GetPredictandDatabase(), false)) compatible = false;
    if (m_hasReferenceValues != otherForecast->HasReferenceValues()) compatible = false;
    if (m_leadTimeOrigin != otherForecast->GetLeadTimeOrigin()) compatible = false;

    Array1DFloat targetDates = otherForecast->GetTargetDates();
    if (m_targetDates.size() != targetDates.size()) {
        compatible = false;
    }
    else {
        for (int i=0; i<m_targetDates.size(); i++) {
            if (m_targetDates[i] != targetDates[i]) compatible = false;
        }
    }

    Array1DInt stationsIds = otherForecast->GetStationIds();
    if (m_stationIds.size() != stationsIds.size()) {
        compatible = false;
    }
    else {
        for (int i=0; i<m_stationIds.size(); i++) {
            if (m_stationIds[i] != stationsIds[i]) compatible = false;
        }
    }

    Array1DFloat referenceAxis = otherForecast->GetReferenceAxis();
    if (m_referenceAxis.size() != referenceAxis.size()) {
        compatible = false;
    }
    else {
        for (int i=0; i<m_referenceAxis.size(); i++) {
            if (m_referenceAxis[i] != referenceAxis[i]) compatible = false;
        }
    }

    if (!compatible)
    {
        asLogError(wxString::Format(_("The forecasts \"%s\" and \"%s\" are not compatible"), m_specificTagDisplay.c_str(), otherForecast->GetSpecificTagDisplay().c_str()));
        return false;
    }

    return true;
}

bool asResultsAnalogsForecast::IsSpecificForStationId(int stationId)
{
    for (int i=0; i<m_predictandStationIds.size(); i++)
    {
        if (m_predictandStationIds[i]==stationId)
        {
            return true;
        }
    }
    return false;
}

int asResultsAnalogsForecast::GetStationRowFromId(int stationId)
{
    for (int i=0; i<m_stationIds.size(); i++)
    {
        if (m_stationIds[i]==stationId)
        {
            return i;
        }
    }

    wxFAIL;
    asLogError(wxString::Format("The station ID %d was not found in the forecast results.", stationId));
    return -1;
}