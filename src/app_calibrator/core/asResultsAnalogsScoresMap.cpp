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
    m_scores.reserve(100);
    m_lon.reserve(100);
    m_lat.reserve(100);
    m_level.reserve(100);
}

asResultsAnalogsScoresMap::~asResultsAnalogsScoresMap()
{
    //dtor
}

void asResultsAnalogsScoresMap::Init(asParametersScoring &params)
{
    BuildFileName(params);

    // Resize to 0 to avoid keeping old results
    m_mapLon.resize(0);
    m_mapLat.resize(0);
    m_mapLevel.resize(0);
    m_mapScores.resize(0);
    m_scores.resize(0);
    m_lon.resize(0);
    m_lat.resize(0);
    m_level.resize(0);
}

void asResultsAnalogsScoresMap::BuildFileName(asParametersScoring &params)
{
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/IntermediateResultsDir", asConfig::GetDefaultUserWorkingDir() + "IntermediateResults" + DS);
    ThreadsManager().CritSectionConfig().Leave();
    m_filePath.Append(DS);
    m_filePath.Append("RelevanceMap");
    m_filePath.Append(DS);
    m_filePath.Append(wxString::Format("%s", GetPredictandStationIdsList()));
    m_filePath.Append(".nc");
}

bool asResultsAnalogsScoresMap::Add(asParametersScoring &params, float score)
{
    if(!params.GetPredictorGridType(0,0).IsSameAs("Regular", false)) asThrowException(_("asResultsAnalogsScoresMap::Add is not ready to use on unregular grids"));

    m_scores.push_back(score);
    m_lon.push_back((params.GetPredictorXmin(0,0)+(params.GetPredictorXptsnb(0,0)-1)*params.GetPredictorXstep(0,0)/2.0));
    m_lat.push_back((params.GetPredictorYmin(0,0)+(params.GetPredictorYptsnb(0,0)-1)*params.GetPredictorYstep(0,0)/2.0));
    m_level.push_back(params.GetPredictorLevel(0,0));

    return true;
}

bool asResultsAnalogsScoresMap::MakeMap()
{

    Array1DFloat levels(asTools::ExtractUniqueValues(&m_level[0], &m_level[m_level.size()-1], 0.0001f));
    Array1DFloat lons(asTools::ExtractUniqueValues(&m_lon[0], &m_lon[m_lon.size()-1], 0.0001f));
    Array1DFloat lats(asTools::ExtractUniqueValues(&m_lat[0], &m_lat[m_lat.size()-1], 0.0001f));

    m_mapLevel = levels;
    m_mapLon = lons;
    m_mapLat = lats;

    Array2DFloat tmpLatLon = Array2DFloat::Constant(m_mapLat.size(), m_mapLon.size(), NaNFloat);

    for (int i_level=0; i_level<=m_mapLevel.size(); i_level++)
    {
        m_mapScores.push_back(tmpLatLon);
    }

    for (unsigned int i=0; i<m_scores.size(); i++)
    {
        int indexLon = asTools::SortedArraySearch(&m_mapLon[0], &m_mapLon[m_mapLon.size()-1], m_lon[i], 0.0001f);
        int indexLat = asTools::SortedArraySearch(&m_mapLat[0], &m_mapLat[m_mapLat.size()-1], m_lat[i], 0.0001f);
        int indexLevel = asTools::SortedArraySearch(&m_mapLevel[0], &m_mapLevel[m_mapLevel.size()-1], m_level[i], 0.0001f);

        m_mapScores[indexLevel](indexLat, indexLon) = m_scores[i];
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
        ResultsFile = m_filePath;
    }
    else
    {
        ResultsFile = AlternateFilePath;
    }

    // Get the elements size
    size_t Nlon = m_mapLon.size();
    size_t Nlat = m_mapLat.size();
    size_t Nlevel = m_mapLevel.size();

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
    wxString title = params.GetForecastScoreName() + " of the analog method";
    ncFile.PutAtt("title", title);

    // Put attributes
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
                scores[ind] = m_mapScores[i_level](i_lat,i_lon);
            }
        }
    }

    // Write data
//    int Leveldata = 850;
    ncFile.PutVarArray("lon", startLon, countLon, &m_mapLon(0));
    ncFile.PutVarArray("lat", startLat, countLat, &m_mapLat(0));
    ncFile.PutVarArray("level", startLevel, countLevel, &m_mapLevel(0));
    ncFile.PutVarArray("forecast_scores", start3, count3, &scores[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}
