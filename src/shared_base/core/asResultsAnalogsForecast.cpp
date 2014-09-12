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

asResultsAnalogsForecast::asResultsAnalogsForecast(const wxString &modelName)
:
asResults()
{
    m_ModelName = modelName;
    m_FilePath = wxEmptyString;
    m_HasReferenceValues = false;
    m_LeadTimeOrigin = 0.0;

    // Default values for former versions
    m_PredictandParameter = Precipitation;
    m_PredictandTemporalResolution = Daily;
    m_PredictandSpatialAggregation = Station;
    m_PredictandDatasetId = wxEmptyString;
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
    m_ReferenceAxis.resize(0);
    m_ReferenceValues.resize(0,0);

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
    wxASSERT(!m_ForecastsDirectory.IsEmpty());

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
    wxString modelname = m_ModelName;
    wxString nowstr = asTime::GetStringTime(m_LeadTimeOrigin, "YYYYMMDDhh");
    wxString ext = "fcst";
    wxString filename = wxString::Format("%s.%s.%s",nowstr.c_str(),modelname.c_str(),ext.c_str());
    m_FilePath.Append(filename);
}

bool asResultsAnalogsForecast::Save(const wxString &AlternateFilePath)
{
    wxASSERT(!m_ModelName.IsEmpty());
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
    wxASSERT(m_StationsLocCoordU.size()>0);
    wxASSERT(m_StationsLocCoordV.size()>0);

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
    ncFile.PutAtt("model_name", m_ModelName);
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
    ncFile.DefVar("loc_coord_u", NC_DOUBLE, 1, DimNamesStations);
    ncFile.DefVar("loc_coord_v", NC_DOUBLE, 1, DimNamesStations);
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
    ncFile.PutAtt("long_name", "Local coordinate U", "loc_coord_u");
    ncFile.PutAtt("var_desc", "Local coordinate for the U axis (west-east)", "loc_coord_u");
    ncFile.PutAtt("units", "m", "loc_coord_u");
    ncFile.PutAtt("long_name", "Local coordinate V", "loc_coord_v");
    ncFile.PutAtt("var_desc", "Local coordinate for the V axis (west-east)", "loc_coord_v");
    ncFile.PutAtt("units", "m", "loc_coord_v");
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
    ncFile.PutVarArray("loc_coord_u", startStations, countStations, &m_StationsLocCoordU(0));
    ncFile.PutVarArray("loc_coord_v", startStations, countStations, &m_StationsLocCoordV(0));
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
    if (version>1.2f)
    {
        asLogError(wxString::Format(_("The forecast file was made with more recent version of AtmoSwing (version %.1f). It cannot be opened here."), version));
        return false;
    }

    if (asTools::IsNaN(version) || version<1.1)
    {
        asLogWarning(_("The forecast file was made with an older version of AtmoSwing."));
        m_PredictandParameter = Precipitation;
        m_PredictandTemporalResolution = Daily;
        m_PredictandSpatialAggregation = Station;
        m_PredictandDatasetId = "MeteoSwiss-Rhone";
        m_ModelName = ncFile.GetAttString("modelName");
        m_DateProcessed = ncFile.GetAttDouble("dateProcessed");
        m_LeadTimeOrigin = ncFile.GetAttDouble("leadTimeOrigin");
        m_HasReferenceValues = true;
    }
    else
    {
        m_PredictandParameter = (DataParameter)ncFile.GetAttInt("predictand_parameter");
        m_PredictandTemporalResolution = (DataTemporalResolution)ncFile.GetAttInt("predictand_temporal_resolution");
        m_PredictandSpatialAggregation = (DataSpatialAggregation)ncFile.GetAttInt("predictand_spatial_aggregation");
        m_PredictandDatasetId = ncFile.GetAttString("predictand_dataset_id");
        m_ModelName = ncFile.GetAttString("model_name");
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
    m_StationsLocCoordU.resize( Nstations );
    m_StationsLocCoordV.resize( Nstations );

    if (asTools::IsNaN(version) || version<1.1)
    {
        ncFile.GetVar("targetdates", &m_TargetDates[0]);
        ncFile.GetVar("analogsnb", &m_AnalogsNb[0]);
        ncFile.GetVar("stationsnames", &m_StationsNames[0], Nstations);
        ncFile.GetVar("stationsids", &m_StationsIds[0]);
        ncFile.GetVar("stationsheights", &m_StationsHeights[0]);
        ncFile.GetVar("loccoordu", &m_StationsLocCoordU[0]);
        ncFile.GetVar("loccoordv", &m_StationsLocCoordV[0]);
    }
    else
    {
        ncFile.GetVar("target_dates", &m_TargetDates[0]);
        ncFile.GetVar("analogs_nb", &m_AnalogsNb[0]);
        ncFile.GetVar("stations_names", &m_StationsNames[0], Nstations);
        ncFile.GetVar("stations_ids", &m_StationsIds[0]);
        ncFile.GetVar("stations_heights", &m_StationsHeights[0]);
        ncFile.GetVar("loc_coord_u", &m_StationsLocCoordU[0]);
        ncFile.GetVar("loc_coord_v", &m_StationsLocCoordV[0]);
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

    wxASSERT(!m_ModelName.IsEmpty());
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
    wxASSERT(m_StationsLocCoordU.size()>0);
    wxASSERT(m_StationsLocCoordV.size()>0);
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
