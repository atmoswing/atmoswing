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
 
#include "asDataPredictandPrecipitation.h"

#include "wx/fileconf.h"

#include <asFileNetcdf.h>
#include <asTimeArray.h>
#include <asCatalog.h>
#include <asCatalogPredictands.h>


asDataPredictandPrecipitation::asDataPredictandPrecipitation(PredictandDB predictandDB)
:
asDataPredictand(predictandDB)
{
    //ctor
}

asDataPredictandPrecipitation::~asDataPredictandPrecipitation()
{
    //dtor
}

bool asDataPredictandPrecipitation::InitContainers()
{
    if (!InitBaseContainers()) return false;

    //m_ReturnPeriodValues.resize(m_StationsNb);
    //m_ReturnPeriodValues.fill(NaNFloat);

    asTimeArray timeArray(m_DateStart, m_DateEnd, m_TimeStepDays*24.0, asTimeArray::Simple);
    timeArray.Init();
    m_Time = timeArray.GetTimeArray();

    return true;
}

bool asDataPredictandPrecipitation::Load(const wxString &AlternateFilePath)
{
    // Get the file path
    wxString PredictandDBFilePath;
    if (AlternateFilePath.IsEmpty())
    {
        wxString FileName = asGlobEnums::PredictandDBEnumToString(m_PredictandDB);
        ThreadsManager().CritSectionConfig().Enter();
        PredictandDBFilePath = wxFileConfig::Get()->Read("/StandardPaths/DataPredictandDBDir", asConfig::GetDefaultUserWorkingDir() + FileName + ".nc");
        ThreadsManager().CritSectionConfig().Leave();
    }
    else
    {
        PredictandDBFilePath = AlternateFilePath;
    }

    // Open the NetCDF file
    asLogMessage(wxString::Format(_("Opening the file %s"), PredictandDBFilePath.c_str()));
    asFileNetcdf ncFile(PredictandDBFilePath, asFileNetcdf::ReadOnly);
    if(!ncFile.Open())
    {
        asLogError(wxString::Format(_("Couldn't open file %s"), PredictandDBFilePath.c_str()));
        return false;
    }
    else
    {
        asLogMessage(_("File successfully opened"));
    }

    // Get global attributes
    m_ReturnPeriodNormalization = ncFile.GetAttFloat("return_period_normalization");
    float version = ncFile.GetAttFloat("version");

    if (asTools::IsNaN(version) || version<m_FileVersion)
    {
        asLogError(_("The predictand DB file was made with an older version of Atmoswing and is not compatible. Please generate the file with the actual version."));
        return false;
    }

    // Get time
    m_TimeLength = ncFile.GetDimLength("time");
    m_Time.resize( m_TimeLength );
    ncFile.GetVar("time", &m_Time[0]);

    // Get stations properties
    m_StationsNb = ncFile.GetDimLength("stations");
    wxASSERT(m_StationsNb>0);
    m_StationsName.resize( m_StationsNb );
    ncFile.GetVar("stationsname", &m_StationsName[0], m_StationsNb);
    m_StationsIds.resize( m_StationsNb );
    ncFile.GetVar("stationsids", &m_StationsIds[0]);
    m_StationsHeight.resize( m_StationsNb );
    ncFile.GetVar("stationsheight", &m_StationsHeight[0]);
    m_StationsLon.resize( m_StationsNb );
    ncFile.GetVar("lon", &m_StationsLon[0]);
    m_StationsLat.resize( m_StationsNb );
    ncFile.GetVar("lat", &m_StationsLat[0]);
    m_StationsLocCoordU.resize( m_StationsNb );
    ncFile.GetVar("loccoordu", &m_StationsLocCoordU[0]);
    m_StationsLocCoordV.resize( m_StationsNb );
    ncFile.GetVar("loccoordv", &m_StationsLocCoordV[0]);
    m_StationsStart.resize( m_StationsNb );
    ncFile.GetVar("start", &m_StationsStart[0]);
    m_StationsEnd.resize( m_StationsNb );
    ncFile.GetVar("end", &m_StationsEnd[0]);

    // Get return periods properties
    int returnPeriodsNb = ncFile.GetDimLength("returnperiods");
    m_ReturnPeriods.resize( returnPeriodsNb );
    ncFile.GetVar("returnperiods", &m_ReturnPeriods[0]);
    size_t startReturnPeriodPrecip[2] = {0, 0};
    size_t countReturnPeriodPrecip[2] = {returnPeriodsNb, m_StationsNb};
    m_DailyPrecipitationsForReturnPeriods.resize( m_StationsNb, returnPeriodsNb );
    ncFile.GetVarArray("dailyprecipitationsforreturnperiods", startReturnPeriodPrecip, countReturnPeriodPrecip, &m_DailyPrecipitationsForReturnPeriods(0,0));

    // Get data
    size_t IndexStart[2] = {0,0};
    size_t IndexCount[2] = {m_TimeLength, m_StationsNb};
    m_DataGross.resize( m_TimeLength, m_StationsNb );
    ncFile.GetVarArray("datagross", IndexStart, IndexCount, &m_DataGross[0]);
    m_DataNormalized.resize( m_TimeLength, m_StationsNb );
    ncFile.GetVarArray("datanormalized", IndexStart, IndexCount, &m_DataNormalized[0]);

    return true;
}

bool asDataPredictandPrecipitation::Save(const wxString &AlternateDestinationDir)
{
    // Get the file path
    wxString PredictandDBFilePath;
    if (AlternateDestinationDir.IsEmpty())
    {
        wxString FileName = asGlobEnums::PredictandDBEnumToString(m_PredictandDB);
        ThreadsManager().CritSectionConfig().Enter();
        PredictandDBFilePath = wxFileConfig::Get()->Read("/StandardPaths/DataPredictandDBDir", asConfig::GetDefaultUserWorkingDir()) + DS + FileName + ".nc";
        ThreadsManager().CritSectionConfig().Leave();
    }
    else
    {
        wxString FileName = asGlobEnums::PredictandDBEnumToString(m_PredictandDB);
        PredictandDBFilePath = AlternateDestinationDir + DS + FileName + ".nc";
    }

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(PredictandDBFilePath, asFileNetcdf::Replace);
    if(!ncFile.Open()) return false;

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("stations", m_StationsNb);
    ncFile.DefDim("returnperiods", (int)m_ReturnPeriods.size());
    ncFile.DefDim("time");

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNameTime;
    DimNameTime.push_back("time");
    VectorStdString DimNameStations;
    DimNameStations.push_back("stations");
    VectorStdString DimNames2D;
    DimNames2D.push_back("time");
    DimNames2D.push_back("stations");
    VectorStdString DimNameReturnPeriods;
    DimNameReturnPeriods.push_back("returnperiods");
    VectorStdString DimNames2DReturnPeriods;
    DimNames2DReturnPeriods.push_back("returnperiods");
    DimNames2DReturnPeriods.push_back("stations");

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("time", NC_DOUBLE, 1, DimNameTime);
    ncFile.DefVar("datagross", NC_FLOAT, 2, DimNames2D);
    ncFile.DefVar("datanormalized", NC_FLOAT, 2, DimNames2D);
    ncFile.DefVar("returnperiods", NC_FLOAT, 1, DimNameReturnPeriods);
    ncFile.DefVar("dailyprecipitationsforreturnperiods", NC_FLOAT, 2, DimNames2DReturnPeriods);
    ncFile.DefVar("stationsname", NC_STRING, 1, DimNameStations);
    ncFile.DefVar("stationsids", NC_INT, 1, DimNameStations);
    ncFile.DefVar("stationsheight", NC_FLOAT, 1, DimNameStations);
    ncFile.DefVar("lon", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("lat", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("loccoordu", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("loccoordv", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("start", NC_DOUBLE, 1, DimNameStations);
    ncFile.DefVar("end", NC_DOUBLE, 1, DimNameStations);

    // Put general attributes
    ncFile.PutAtt("return_period_normalization", &m_ReturnPeriodNormalization);
    ncFile.PutAtt("version", &m_FileVersion);

    // Put attributes for the stations
    ncFile.PutAtt("long_name", "Stations names", "stationsname");
    ncFile.PutAtt("var_desc", "Name of the predictand stations", "stationsname");

    // Put attributes for the stations
    ncFile.PutAtt("long_name", "Stations IDs", "stationsids");
    ncFile.PutAtt("var_desc", "Internal IDs of the predictand stations", "stationsids");

    // Put attributes for the stations
    ncFile.PutAtt("long_name", "Stations height", "stationsheight");
    ncFile.PutAtt("var_desc", "Altitude of the predictand stations", "stationsheight");
    ncFile.PutAtt("units", "m", "stationsheight");

    // Put attributes for the lon variable
    ncFile.PutAtt("long_name", "Longitude", "lon");
    ncFile.PutAtt("var_desc", "Longitudes of the stations positions", "lon");
    ncFile.PutAtt("units", "degrees", "lon");

    // Put attributes for the lat variable
    ncFile.PutAtt("long_name", "Latitude", "lat");
    ncFile.PutAtt("var_desc", "Latitudes of the stations positions", "lat");
    ncFile.PutAtt("units", "degrees", "lat");

    // Put attributes for the loccoordu variable
    ncFile.PutAtt("long_name", "Local coordinate U", "loccoordu");
    ncFile.PutAtt("var_desc", "Local coordinate for the U axis (west-east)", "loccoordu");
    ncFile.PutAtt("units", "m", "loccoordu");

    // Put attributes for the loccoordv variable
    ncFile.PutAtt("long_name", "Local coordinate V", "loccoordv");
    ncFile.PutAtt("var_desc", "Local coordinate for the V axis (west-east)", "loccoordv");
    ncFile.PutAtt("units", "m", "loccoordv");

    // Put attributes for the start variable
    ncFile.PutAtt("long_name", "Start", "start");
    ncFile.PutAtt("var_desc", "Start of the stations data", "start");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "start");

    // Put attributes for the end variable
    ncFile.PutAtt("long_name", "End", "end");
    ncFile.PutAtt("var_desc", "End of the stations data", "end");
    ncFile.PutAtt("units", "Modified Julian Day Number (MJD)", "end");

    // Put attributes for the data variable
    ncFile.PutAtt("long_name", "Precipitation gross", "datagross");
    ncFile.PutAtt("var_desc", "Predictand gross data, whithout any treatment", "datagross");
    ncFile.PutAtt("units", "mm", "datagross");

    // Put attributes for the data variable
    ncFile.PutAtt("long_name", "Precipitation normalized", "datanormalized");
    ncFile.PutAtt("var_desc", "Predictand normalized data", "datanormalized");
    ncFile.PutAtt("units", "-", "datanormalized");

    // Put attributes for the return periods variable
    ncFile.PutAtt("long_name", "Return periods", "returnperiods");
    ncFile.PutAtt("var_desc", "Return periods", "returnperiods");
    ncFile.PutAtt("units", "year", "returnperiods");

    // Put attributes for the daily precipitations corresponding to the return periods
    ncFile.PutAtt("long_name", "Daily precipitation for return periods", "dailyprecipitationsforreturnperiods");
    ncFile.PutAtt("var_desc", "Daily precipitation corresponding to the return periods", "dailyprecipitationsforreturnperiods");
    ncFile.PutAtt("units", "mm", "dailyprecipitationsforreturnperiods");

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startTime[] = {0};
    size_t countTime[] = {m_TimeLength};
    size_t startStations[] = {0};
    size_t countStations[] = {m_StationsNb};
    size_t start2[] = {0, 0};
    size_t count2[] = {m_TimeLength, m_StationsNb};
    size_t startReturnPeriod[] = {0};
    size_t countReturnPeriod[] = {m_ReturnPeriods.size()};
    size_t startReturnPeriodPrecip[] = {0, 0};
    size_t countReturnPeriodPrecip[] = {m_ReturnPeriods.size(), m_StationsNb};

    // Write data
    ncFile.PutVarArray("time", startTime, countTime, &m_Time(0));
    ncFile.PutVarArray("stationsname", startStations, countStations, &m_StationsName[0], m_StationsName.size());
    ncFile.PutVarArray("stationsids", startStations, countStations, &m_StationsIds(0));
    ncFile.PutVarArray("stationsheight", startStations, countStations, &m_StationsHeight(0));
    ncFile.PutVarArray("lon", startStations, countStations, &m_StationsLon(0));
    ncFile.PutVarArray("lat", startStations, countStations, &m_StationsLat(0));
    ncFile.PutVarArray("loccoordu", startStations, countStations, &m_StationsLocCoordU(0));
    ncFile.PutVarArray("loccoordv", startStations, countStations, &m_StationsLocCoordV(0));
    ncFile.PutVarArray("start", startStations, countStations, &m_StationsStart(0));
    ncFile.PutVarArray("end", startStations, countStations, &m_StationsEnd(0));
    ncFile.PutVarArray("datagross", start2, count2, &m_DataGross(0,0));
    ncFile.PutVarArray("datanormalized", start2, count2, &m_DataNormalized(0,0));
    ncFile.PutVarArray("returnperiods", startReturnPeriod, countReturnPeriod, &m_ReturnPeriods(0));
    ncFile.PutVarArray("dailyprecipitationsforreturnperiods", startReturnPeriodPrecip, countReturnPeriodPrecip, &m_DailyPrecipitationsForReturnPeriods(0,0));

    // Close:save new netCDF dataset
    ncFile.Close();

    return true;
}
bool asDataPredictandPrecipitation::BuildPrecipitationDB(float returnPeriodNormalization, int makeSqrt, const wxString &AlternateCatalogFilePath, const wxString &AlternateDataDir, const wxString &AlternatePatternDir, const wxString &AlternateDestinationDir)
{
    if(!g_UnitTesting) asLogMessage(_("Building the predictand DB."));

    // Initialize the members
    if(!InitMembers(AlternateCatalogFilePath)) return false;

    // Resize matrices
    if(!InitContainers()) return false;;

    // Index for stations
    int stationIndex = 0;

    #ifndef UNIT_TESTING
        #if wxUSE_GUI
            // The progress bar
            asDialogProgressBar ProgressBar(_("Loading data from files.\n"), m_StationsNb);
        #endif
    #endif

    // Get the datasets IDs
    asCatalog::DatasetIdList datsetList = asCatalog::GetDatasetIdList(Predictand, AlternateCatalogFilePath);

    // Get the data
    for (size_t i_set=0; i_set<datsetList.Id.size(); i_set++)
    {
        // The dataset ID
        wxString datasetId = datsetList.Id[i_set];

        //Include in DB
        if (IncludeInDB(datasetId, AlternateCatalogFilePath))
        {
            // Get the stations list
            asCatalog::DataIdListInt stationsList = asCatalog::GetDataIdListInt(Predictand, datasetId, AlternateCatalogFilePath);

            for (size_t i_station=0; i_station<stationsList.Id.size(); i_station++)
            {
                // The station ID
                int stationId = stationsList.Id[i_station];

                // Load data properties
                asCatalogPredictands currentData(AlternateCatalogFilePath);
                if(!currentData.Load(datasetId, stationId)) return false;

                #ifndef UNIT_TESTING
                    #if wxUSE_GUI
                        // Update the progress bar.
                        wxString fileNameMessage = wxString::Format(_("Loading data from files.\nFile: %s"), currentData.GetStationFilename().c_str());
                        if(!ProgressBar.Update(stationIndex, fileNameMessage))
                        {
                            asLogError(_("The process has been canceled by the user."));
                            return false;
                        }
                    #endif
                #endif

                // Get station information
                if(!SetStationProperties(currentData, stationIndex)) return false;

                // Get file content
                if(!GetFileContent(currentData, stationIndex, AlternateDataDir, AlternatePatternDir)) return false;

                stationIndex++;
            }
        }
    }
    #ifndef UNIT_TESTING
        #if wxUSE_GUI
            ProgressBar.Destroy();
        #endif
    #endif

    // Make the Gumbel adjustment
    if(!MakeGumbelAdjustment()) return false;

    // Process the normalized Precipitation
    if(!BuildDataNormalized(returnPeriodNormalization, makeSqrt)) return false;

    // Process daily precipitations for all return periods
    if(!BuildDailyPrecipitationsForAllReturnPeriods()) return false;

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

bool asDataPredictandPrecipitation::MakeGumbelAdjustment()
{
    // Duration of the Precipitation
    Array1DDouble duration;
    if (m_TimeStepDays==1)
    {
        duration.resize(7);
        duration << 1,2,3,4,5,7,10;
    }
    else if (m_TimeStepDays<=1.0/24.0)
    {
        duration.resize(14);
        duration << 1.0/24.0,2.0/24.0,3.0/24.0,4.0/24.0,5.0/24.0,6.0/24.0,12.0/24.0,1,2,3,4,5,7,10;
    }
    else if (m_TimeStepDays<=2.0/24.0)
    {
        duration.resize(13);
        duration << 2.0/24.0,3.0/24.0,4.0/24.0,5.0/24.0,6.0/24.0,12.0/24.0,1,2,3,4,5,7,10;
    }
    else if (m_TimeStepDays<=6.0/24.0)
    {
        duration.resize(9);
        duration << 6.0/24.0,12.0/24.0,1,2,3,4,5,7,10;
    }
    else if (m_TimeStepDays<=12.0/24.0)
    {
        duration.resize(8);
        duration << 12.0/24.0,1,2,3,4,5,7,10;
    }
    else
    {
        asLogError(_("The data time steps is not correctly defined."));
        duration.resize(7);
        duration << 1,2,3,4,5,7,10;
    }

    // Preprocess cste
    float b_cst = sqrt(6.0)/g_Cst_Pi;

    // Resize containers
    m_GumbelDuration.resize(m_StationsNb, duration.size());
    m_GumbelParamA.resize(m_StationsNb, duration.size());
    m_GumbelParamB.resize(m_StationsNb, duration.size());

    #ifndef UNIT_TESTING
        #if wxUSE_GUI
            // The progress bar
            asDialogProgressBar ProgressBar(_("Making Gumbel adjustments."), duration.size()-1);
        #endif
    #endif

    for (float i_duration=0; i_duration<duration.size(); i_duration++)
    {
        // Get the annual max
        Array2DFloat annualMax = GetAnnualMax(duration[i_duration]);

        #ifndef UNIT_TESTING
            #if wxUSE_GUI
                if(!ProgressBar.Update(i_duration))
                {
                    asLogError(_("The process has been canceled by the user."));
                    return false;
                }
            #endif
        #endif

        for (int i_st=0; i_st<m_StationsNb; i_st++)
        {
            // Check the length of the data
            int dataLength = asTools::CountNotNaN(&annualMax(i_st,0), &annualMax(i_st, annualMax.cols()-1));
            if(dataLength<20)
            {
                asLogError(_("Caution, a time serie is shorter than 20 years. It is too short to process a Gumbel adjustment."));
                return false;
            }
            else if(dataLength<30)
            {
                asLogWarning(_("Caution, a time serie is shorter than 30 years. It is a bit short to process a Gumbel adjustment."));
            }

            if(!asTools::SortArray(&annualMax(i_st,0), &annualMax(i_st,annualMax.cols()-1), Asc)) return false;
            float mean = asTools::Mean(&annualMax(i_st,0), &annualMax(i_st,annualMax.cols()-1));
            float stdev = asTools::StDev(&annualMax(i_st,0), &annualMax(i_st,annualMax.cols()-1), asSAMPLE);

            float b = b_cst*stdev;
            float a = mean-b*g_Cst_Euler; // EUCON: Euler-Mascheroni constant in math.h

            m_GumbelDuration(i_st,i_duration) = duration[i_duration];
            m_GumbelParamA(i_st,i_duration) = a;
            m_GumbelParamB(i_st,i_duration) = b;
        }
    }
    #ifndef UNIT_TESTING
        #if wxUSE_GUI
            ProgressBar.Destroy();
        #endif
    #endif

    return true;
}

float asDataPredictandPrecipitation::GetPrecipitationOfReturnPeriod(int i_station, double duration, float returnPeriod)
{
    float F = 1-(1/returnPeriod); // Probability of not overtaking
    float u = -log(-log(F)); // Gumbel variable
    int i_duration = asTools::SortedArraySearch(&m_GumbelDuration(i_station,0), &m_GumbelDuration(i_station,m_GumbelDuration.cols()-1),duration,0.00001f);
    return m_GumbelParamB(i_station,i_duration)*u + m_GumbelParamA(i_station,i_duration);
}

bool asDataPredictandPrecipitation::BuildDailyPrecipitationsForAllReturnPeriods()
{
    float duration = 1; // day
    m_ReturnPeriods.resize(10);
    m_ReturnPeriods << 2,2.33f,5,10,20,50,100,200,300,500;
    m_DailyPrecipitationsForReturnPeriods.resize(m_StationsNb, m_ReturnPeriods.size());

    for (int i_station=0; i_station<m_StationsNb; i_station++)
    {
        for (int i_retperiod=0; i_retperiod<m_ReturnPeriods.size(); i_retperiod++)
        {
            float F = 1-(1/m_ReturnPeriods[i_retperiod]); // Probability of not overtaking
            float u = -log(-log(F)); // Gumbel variable
            int i_duration = asTools::SortedArraySearch(&m_GumbelDuration(i_station,0), &m_GumbelDuration(i_station,m_GumbelDuration.cols()-1),duration,0.00001f);
            float val = m_GumbelParamB(i_station,i_duration)*u + m_GumbelParamA(i_station,i_duration);
            wxASSERT(val>0);
            wxASSERT(val<500);
            m_DailyPrecipitationsForReturnPeriods(i_station, i_retperiod) = val;
        }
    }

    return true;
}

bool asDataPredictandPrecipitation::BuildDataNormalized(float returnPeriod, int makeSqrt)
{
    m_ReturnPeriodNormalization = returnPeriod;

    for (int i_st=0; i_st<m_StationsNb; i_st++)
    {
        float Prt = 1.0;
        if (returnPeriod!=0)
        {
            Prt = GetPrecipitationOfReturnPeriod(i_st, 1, returnPeriod);
        }

        for (int i_time=0; i_time<m_TimeLength; i_time++)
        {
            if (makeSqrt==asDONT_MAKE_SQRT)
            {
                m_DataNormalized(i_time,i_st) = m_DataGross(i_time, i_st)/Prt;
            }
            else
            {
                m_DataNormalized(i_time,i_st) = sqrt(m_DataGross(i_time, i_st)/Prt);
            }
        }
    }
    return true;
}
