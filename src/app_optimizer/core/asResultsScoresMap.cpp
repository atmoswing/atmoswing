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
#include "asIncludes.h"
#include "asParametersCalibration.h"

asResultsScoresMap::asResultsScoresMap()
    : asResults() {
    _scores.reserve(100);
    _lon.reserve(100);
    _lat.reserve(100);
    _level.reserve(100);
}

asResultsScoresMap::~asResultsScoresMap() {}

void asResultsScoresMap::Init() {
    BuildFileName();

    // Resize to 0 to avoid keeping old results
    _mapLon.resize(0);
    _mapLat.resize(0);
    _mapLevel.resize(0);
    _mapScores.resize(0);
    _scores.resize(0);
    _lon.resize(0);
    _lat.resize(0);
    _level.resize(0);
}

void asResultsScoresMap::BuildFileName() {
    ThreadsManager().CritSectionConfig().Enter();
    _filePath = wxFileConfig::Get()->Read("/Paths/ResultsDir", asConfig::GetDefaultUserWorkingDir());
    ThreadsManager().CritSectionConfig().Leave();
    if (!_subFolder.IsEmpty()) {
        _filePath.Append(DS);
        _filePath.Append(_subFolder);
    }
    _filePath.Append(DS);
    _filePath.Append("RelevanceMap");
    _filePath.Append(DS);
    _filePath.Append(asStrF("%s", GetPredictandStationIdsList()));
    _filePath.Append(".nc");
}

bool asResultsScoresMap::Add(asParametersScoring& params, float score) {
    if (!params.GetPredictorGridType(0, 0).IsSameAs("Regular", false))
        throw runtime_error(_("asResultsScoresMap::Add is not ready to use on unregular grids"));

    _scores.push_back(score);
    _lon.push_back(
        (params.GetPredictorXmin(0, 0) + (params.GetPredictorXptsnb(0, 0) - 1) * params.GetPredictorXstep(0, 0) / 2.0));
    _lat.push_back(
        (params.GetPredictorYmin(0, 0) + (params.GetPredictorYptsnb(0, 0) - 1) * params.GetPredictorYstep(0, 0) / 2.0));
    _level.push_back(params.GetPredictorLevel(0, 0));

    return true;
}

bool asResultsScoresMap::MakeMap() {
    _mapLevel = a1f(asExtractUniqueValues(&_level[0], &_level[_level.size() - 1], 0.0001f));
    _mapLon = a1f(asExtractUniqueValues(&_lon[0], &_lon[_lon.size() - 1], 0.0001f));
    _mapLat = a1f(asExtractUniqueValues(&_lat[0], &_lat[_lat.size() - 1], 0.0001f));

    a2f tmpLatLon = a2f::Constant(_mapLat.size(), _mapLon.size(), NAN);

    for (int iLevel = 0; iLevel <= _mapLevel.size(); iLevel++) {
        _mapScores.push_back(tmpLatLon);
    }

    for (int i = 0; i < _scores.size(); i++) {
        int indexLon = asFind(&_mapLon[0], &_mapLon[_mapLon.size() - 1], _lon[i], 0.0001f);
        int indexLat = asFind(&_mapLat[0], &_mapLat[_mapLat.size() - 1], _lat[i], 0.0001f);
        int indexLevel = asFind(&_mapLevel[0], &_mapLevel[_mapLevel.size() - 1], _level[i], 0.0001f);

        if (indexLon > 0 && indexLat > 0 && indexLevel > 0) {
            _mapScores[indexLevel](indexLat, indexLon) = _scores[i];
        }
    }

    return true;
}

bool asResultsScoresMap::Save(asParametersCalibration& params) {
    // Build the map (spatialize the data)
    MakeMap();

    // Get the elements size
    size_t nLon = (size_t)_mapLon.size();
    size_t nLat = (size_t)_mapLat.size();
    size_t nLevel = (size_t)_mapLevel.size();

    ThreadsManager().CritSectionNetCDF().Enter();

    // Create netCDF dataset: enter define mode
    asFileNetcdf ncFile(_filePath, asFileNetcdf::Replace);
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
                scores[ind] = _mapScores[iLevel](iLat, iLon);
            }
        }
    }

    // Write data
    //    int Leveldata = 850;
    ncFile.PutVarArray("lon", startLon, countLon, &_mapLon(0));
    ncFile.PutVarArray("lat", startLat, countLat, &_mapLat(0));
    ncFile.PutVarArray("level", startLevel, countLevel, &_mapLevel(0));
    ncFile.PutVarArray("scores", start3, count3, &scores[0]);

    // Close:save new netCDF dataset
    ncFile.Close();

    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}
