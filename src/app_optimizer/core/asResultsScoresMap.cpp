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

#include "asResultsScoresMap.h"

#include "asFileNetcdf.h"
#include "asParametersCalibration.h"

asResultsScoresMap::asResultsScoresMap()
    : asResults() {
    m_scores.reserve(100);
    m_lon.reserve(100);
    m_lat.reserve(100);
    m_level.reserve(100);
}

asResultsScoresMap::~asResultsScoresMap() {}

void asResultsScoresMap::Init() {
    BuildFileName();

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

void asResultsScoresMap::BuildFileName() {
    ThreadsManager().CritSectionConfig().Enter();
    m_filePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    if (!m_subFolder.IsEmpty()) {
        m_filePath.Append(DS);
        m_filePath.Append(m_subFolder);
    }
    m_filePath.Append(DS);
    m_filePath.Append("RelevanceMap");
    m_filePath.Append(DS);
    m_filePath.Append(asStrF("%s", GetPredictandStationIdsList()));
    m_filePath.Append(".nc");
}

bool asResultsScoresMap::Add(asParametersScoring& params, float score) {
    if (!params.GetPredictorGridType(0, 0).IsSameAs("Regular", false))
        throw runtime_error(_("asResultsScoresMap::Add is not ready to use on unregular grids"));

    m_scores.push_back(score);
    m_lon.push_back(
        (params.GetPredictorXmin(0, 0) + (params.GetPredictorXptsnb(0, 0) - 1) * params.GetPredictorXstep(0, 0) / 2.0));
    m_lat.push_back(
        (params.GetPredictorYmin(0, 0) + (params.GetPredictorYptsnb(0, 0) - 1) * params.GetPredictorYstep(0, 0) / 2.0));
    m_level.push_back(params.GetPredictorLevel(0, 0));

    return true;
}

bool asResultsScoresMap::MakeMap() {
    a1f levels(asExtractUniqueValues(&m_level[0], &m_level[m_level.size() - 1], 0.0001f));
    a1f lons(asExtractUniqueValues(&m_lon[0], &m_lon[m_lon.size() - 1], 0.0001f));
    a1f lats(asExtractUniqueValues(&m_lat[0], &m_lat[m_lat.size() - 1], 0.0001f));

    m_mapLevel = levels;
    m_mapLon = lons;
    m_mapLat = lats;

    a2f tmpLatLon = a2f::Constant(m_mapLat.size(), m_mapLon.size(), NAN);

    for (int iLevel = 0; iLevel <= m_mapLevel.size(); iLevel++) {
        m_mapScores.push_back(tmpLatLon);
    }

    for (int i = 0; i < m_scores.size(); i++) {
        int indexLon = asFind(&m_mapLon[0], &m_mapLon[m_mapLon.size() - 1], m_lon[i], 0.0001f);
        int indexLat = asFind(&m_mapLat[0], &m_mapLat[m_mapLat.size() - 1], m_lat[i], 0.0001f);
        int indexLevel = asFind(&m_mapLevel[0], &m_mapLevel[m_mapLevel.size() - 1], m_level[i], 0.0001f);

        if (indexLon > 0 && indexLat > 0 && indexLevel > 0) {
            m_mapScores[indexLevel](indexLat, indexLon) = m_scores[i];
        }
    }

    return true;
}

bool asResultsScoresMap::Save(asParametersCalibration& params) {
    // Build the map (spatialize the data)
    MakeMap();

    // Get the elements size
    size_t nLon = (size_t)m_mapLon.size();
    size_t nLat = (size_t)m_mapLat.size();
    size_t nLevel = (size_t)m_mapLevel.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(m_filePath, asFileNetcdf::Replace);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Define dimensions. Time is the unlimited dimension.
    ncFile.DefDim("lon", nLon);
    ncFile.DefDim("lat", nLat);
    ncFile.DefDim("level", nLevel);

    // The dimensions name array is used to pass the dimensions to the variable.
    vstds dimNamesLon;
    dimNamesLon.push_back("lon");
    vstds dimNamesLat;
    dimNamesLat.push_back("lat");
    vstds dimNamesLevel;
    dimNamesLevel.push_back("level");
    vstds dimNames3;
    dimNames3.push_back("level");
    dimNames3.push_back("lat");
    dimNames3.push_back("lon");

    // Define variables: the scores and the corresponding dates
    ncFile.DefVar("scores", NC_FLOAT, 3, dimNames3);
    ncFile.DefVar("lat", NC_FLOAT, 1, dimNamesLat);
    ncFile.DefVar("level", NC_FLOAT, 1, dimNamesLevel);
    ncFile.DefVar("lon", NC_FLOAT, 1, dimNamesLon);

    // Put global attributes
    ncFile.PutAtt("Conventions", "COARDS");
    wxString title = params.GetScoreName() + " of the analog method";
    ncFile.PutAtt("title", title);

    // Put attributes
    DefLevelAttributes(ncFile);
    DefScoresMapAttributes(ncFile);

    // End definitions: leave define mode
    ncFile.EndDef();

    // Provide sizes for variables
    size_t startLon[] = {0};
    size_t countLon[] = {nLon};
    size_t startLat[] = {0};
    size_t countLat[] = {nLat};
    size_t startLevel[] = {0};
    size_t countLevel[] = {nLevel};
    size_t start3[] = {0, 0, 0};
    size_t count3[] = {nLevel, nLat, nLon};

    // Set the scores in a vector
    vf scores(nLevel * nLat * nLon);
    int ind;

    for (int iLevel = 0; iLevel < nLevel; iLevel++) {
        for (int iLat = 0; iLat < nLat; iLat++) {
            for (int iLon = 0; iLon < nLon; iLon++) {
                ind = iLon;
                ind += iLat * nLon;
                ind += iLevel * nLon * nLat;
                scores[ind] = m_mapScores[iLevel](iLat, iLon);
            }
        }
    }

    // Write data
    //    int Leveldata = 850;
    ncFile.PutVarArray("lon", startLon, countLon, &m_mapLon(0));
    ncFile.PutVarArray("lat", startLat, countLat, &m_mapLat(0));
    ncFile.PutVarArray("level", startLevel, countLevel, &m_mapLevel(0));
    ncFile.PutVarArray("scores", start3, count3, &scores[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}
