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
 * Portions Copyright 2013 Pascal Horton, Terr@num.
 */

#include "asResultsAnalogsScoresMap.h"

#include <asParametersCalibration.h>
#include <asParametersScoring.h>
#include <asFileNetcdf.h>


asResultsAnalogsScoresMap::asResultsAnalogsScoresMap()
:
asResults()
{
    m_Scores.reserve(100);
    m_Lon.reserve(100);
    m_Lat.reserve(100);
    m_Level.reserve(100);
}

asResultsAnalogsScoresMap::~asResultsAnalogsScoresMap()
{
    //dtor
}

void asResultsAnalogsScoresMap::Init(asParametersScoring &params)
{
    BuildFileName(params);

    // Resize to 0 to avoid keeping old results
    m_MapLon.resize(0);
    m_MapLat.resize(0);
    m_MapLevel.resize(0);
    m_MapScores.resize(0);
    m_Scores.resize(0);
    m_Lon.resize(0);
    m_Lat.resize(0);
    m_Level.resize(0);
}

void asResultsAnalogsScoresMap::BuildFileName(asParametersScoring &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_FilePath = wxFileConfig::Get()->Read("/StandardPaths/IntermediateResultsDir", asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_FilePath.Append(DS);
    m_FilePath.Append("RelevanceMap");
    m_FilePath.Append(DS);
    m_FilePath.Append(wxString::Format("%s", GetPredictandStationIdsList().c_str()));
    m_FilePath.Append(".nc");
}

bool asResultsAnalogsScoresMap::Add(asParametersScoring &params, float score)
{
    if(!params.GetPredictorGridType(0,0).IsSameAs("Regular", false)) asThrowException(_("asResultsAnalogsScoresMap::Add is not ready to use on unregular grids"));

    m_Scores.push_back(score);
    m_Lon.push_back((params.GetPredictorUmin(0,0)+(params.GetPredictorUptsnb(0,0)-1)*params.GetPredictorUstep(0,0)/2.0));
    m_Lat.push_back((params.GetPredictorVmin(0,0)+(params.GetPredictorVptsnb(0,0)-1)*params.GetPredictorVstep(0,0)/2.0));
    m_Level.push_back(params.GetPredictorLevel(0,0));

    return true;
}

bool asResultsAnalogsScoresMap::MakeMap()
{

    Array1DFloat levels(asTools::ExtractUniqueValues(&m_Level[0], &m_Level[m_Level.size()-1], 0.0001f));
    Array1DFloat lons(asTools::ExtractUniqueValues(&m_Lon[0], &m_Lon[m_Lon.size()-1], 0.0001f));
    Array1DFloat lats(asTools::ExtractUniqueValues(&m_Lat[0], &m_Lat[m_Lat.size()-1], 0.0001f));

    m_MapLevel = levels;
    m_MapLon = lons;
    m_MapLat = lats;

    Array2DFloat tmpLatLon = Array2DFloat::Constant(m_MapLat.size(), m_MapLon.size(), NaNFloat);

    for (int i_level=0; i_level<=m_MapLevel.size(); i_level++)
    {
        m_MapScores.push_back(tmpLatLon);
    }

    int indexLon, indexLat, indexLevel;

    for (unsigned int i=0; i<m_Scores.size(); i++)
    {
        indexLon = asTools::SortedArraySearch(&m_MapLon[0], &m_MapLon[m_MapLon.size()-1], m_Lon[i], 0.0001f);
        indexLat = asTools::SortedArraySearch(&m_MapLat[0], &m_MapLat[m_MapLat.size()-1], m_Lat[i], 0.0001f);
        indexLevel = asTools::SortedArraySearch(&m_MapLevel[0], &m_MapLevel[m_MapLevel.size()-1], m_Level[i], 0.0001f);

        m_MapScores[indexLevel](indexLat, indexLon) = m_Scores[i];
    }

    return true;
}

bool asResultsAnalogsScoresMap::Save(asParametersCalibration &params, const wxString &AlternateFilePath)
{
    // Build the map (spatilize the data)
    MakeMap();

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
    size_t Nlon = m_MapLon.size();
    size_t Nlat = m_MapLat.size();
    size_t Nlevel = m_MapLevel.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(ResultsFile, asFileNetcdf::Replace);
    if(!ncFile.Open())
    {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("lon", Nlon);
    ncFile.DefDim("lat", Nlat);
    ncFile.DefDim("level", Nlevel);

    // The dimensions name array is used to pass the dimensions to the variable.
    VectorStdString DimNamesLon;
    DimNamesLon.push_back("lon");
    VectorStdString DimNamesLat;
    DimNamesLat.push_back("lat");
    VectorStdString DimNamesLevel;
    DimNamesLevel.push_back("level");
    VectorStdString DimNames3;
    DimNames3.push_back("level");
    DimNames3.push_back("lat");
    DimNames3.push_back("lon");

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("forecast_scores", NC_FLOAT, 3, DimNames3);
    ncFile.DefVar("lat", NC_FLOAT, 1, DimNamesLat);
    ncFile.DefVar("level", NC_FLOAT, 1, DimNamesLevel);
    ncFile.DefVar("lon", NC_FLOAT, 1, DimNamesLon);

    // Put global attributes
    ncFile.PutAtt("Conventions","COARDS");
    wxString title = params.GetForecastScoreName() + " of the " + params.GetMethodName(0) + " method";
    ncFile.PutAtt("title", title);

    // Put attributes
    DefLonLatAttributes(ncFile);
    DefLevelAttributes(ncFile);
    DefScoresMapAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startLon[] = {0};
    size_t countLon[] = {Nlon};
    size_t startLat[] = {0};
    size_t countLat[] = {Nlat};
    size_t startLevel[] = {0};
    size_t countLevel[] = {Nlevel};
    size_t start3[] = {0, 0, 0};
    size_t count3[] = {Nlevel, Nlat, Nlon};

    // Set the scores in a vector
    VectorFloat scores(Nlevel * Nlat * Nlon);
    int ind = 0;

    for (unsigned int i_level=0; i_level<Nlevel; i_level++)
    {
        for (unsigned int i_lat=0; i_lat<Nlat; i_lat++)
        {
            for (unsigned int i_lon=0; i_lon<Nlon; i_lon++)
            {
                ind = i_lon;
                ind += i_lat * Nlon;
                ind += i_level * Nlon * Nlat;
                scores[ind] = m_MapScores[i_level](i_lat,i_lon);
            }
        }
    }

    // Write data
//    int Leveldata = 850;
    ncFile.PutVarArray("lon", startLon, countLon, &m_MapLon(0));
    ncFile.PutVarArray("lat", startLat, countLat, &m_MapLat(0));
    ncFile.PutVarArray("level", startLevel, countLevel, &m_MapLevel(0));
    ncFile.PutVarArray("forecast_scores", start3, count3, &scores[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}
