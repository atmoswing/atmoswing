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
 * Portions Copyright 2018-2019 Pascal Horton, University of Bern.
 */

#include "asFileGrib.h"

#include "eccodes.h"

asFileGrib::asFileGrib(const wxString& fileName, const FileMode& fileMode)
    : asFile(fileName, fileMode),
      m_filtPtr(nullptr),
      m_version(0),
      m_index(asNOT_FOUND) {
    switch (fileMode) {
        case (ReadOnly):
            // OK
            break;
        case (Write):
        case (Replace):
        case (New):
        case (Append):
        default:
            throw exception(_("Grib files edition is not implemented."));
    }
}

asFileGrib::~asFileGrib() {
    Close();
}

void asFileGrib::SetContext() {
    grib_context* context = grib_context_get_default();
    codes_context_set_definitions_path(context, asFileGrib::GetDefinitionsPath());
}

wxString asFileGrib::GetDefinitionsPath() {
    wxString definitionsPathEnv;
    wxGetEnv("ECCODES_DEFINITION_PATH", &definitionsPathEnv);
    wxConfigBase* pConfig = wxFileConfig::Get();
    wxString definitionsPath = pConfig->Read("/Libraries/EcCodesDefinitions", definitionsPathEnv);

    wxUniChar separator = wxFileName::GetPathSeparator();
    if (!definitionsPath.EndsWith("definitions") && !definitionsPath.EndsWith("definitions" + wxString(separator))) {
        definitionsPath += wxString(separator) + "definitions";
    }

    if (!wxDirExists(definitionsPath)) {
        wxLogWarning(_("The ecCodes definition path '%s' was not found."), definitionsPath);
    }

    return definitionsPath;
}

bool asFileGrib::Open() {
    if (!Find()) return false;
    wxLogVerbose(_("Grib file found."));

    if (!OpenDataset()) return false;

    m_opened = true;

    return true;
}

bool asFileGrib::Close() {
    if (m_filtPtr) {
        fclose(m_filtPtr);
        m_filtPtr = nullptr;
    }

    return true;
}

bool asFileGrib::OpenDataset() {
    // Filepath
    wxString filePath = m_fileName.GetFullPath();

    // Open file
    m_filtPtr = fopen(filePath.mb_str(), "rb");

    if (!m_filtPtr)  // Failed
    {
        wxLogError(_("The opening of the grib file failed."));
        wxFAIL;
        return false;
    }

    // Parse structure
    return ParseStructure();
}

bool asFileGrib::ParseStructure() {
    int err = 0;

    // Loop over the GRIB messages in the source
    wxLogVerbose(_("Creating handle from file %s"), m_fileName.GetFullPath());
    try {
        codes_handle* h;
        while ((h = codes_handle_new_from_file(NULL, m_filtPtr, PRODUCT_GRIB, &err)) != nullptr) {
            wxLogVerbose(_("Check if Grib error"));
            if (!CheckGribErrorCode(err)) {
                return false;
            }

            if (m_version == 0) {
                long version;
                CODES_CHECK(codes_get_long(h, "editionNumber", &version), 0);
                m_version = version;
            }
            wxASSERT(m_version == 1 || m_version == 2);

            ExtractAxes(h);
            ExtractLevel(h);
            ExtractTime(h);
            ExtractGribCode(h);

            codes_handle_delete(h);
        }
    } catch (std::exception& e) {
        wxString msg(e.what(), wxConvUTF8);
        wxLogError(_("Exception caught: %s"), msg);
        wxLogError(_("Failed to parse grib file (exception)."));
        return false;
    }

    return CheckGribErrorCode(err);
}

void asFileGrib::ExtractTime(codes_handle* h) {
    // Keys: https://apps.ecmwf.int/codes/grib/format/edition-independent/2/

    // Get reference date
    size_t dataDateLength = 20;
    char* buffer1 = NULL;
    buffer1 = static_cast<char*>(malloc(dataDateLength * sizeof(char)));
    CODES_CHECK(codes_get_string(h, "dataDate", &buffer1[0], &dataDateLength), 0);
    wxString dataDate(buffer1, wxConvUTF8);
    free(buffer1);
    double refDate = asTime::GetTimeFromString(dataDate, YYYYMMDD);
    m_refDates.push_back(refDate);

    size_t dataTimeLength = 20;
    char* buffer2 = NULL;
    buffer2 = static_cast<char*>(malloc(dataTimeLength * sizeof(char)));
    CODES_CHECK(codes_get_string(h, "dataTime", &buffer2[0], &dataTimeLength), 0);
    wxString dataTime(buffer2, wxConvUTF8);
    free(buffer2);
    double refTime;
    dataTime.ToDouble(&refTime);
    m_refTimes.push_back(refTime);

    // Get forecast time
    double forecastTime = 0;
    if (m_version == 2) {
        CODES_CHECK(codes_get_double(h, "forecastTime", &forecastTime), 0);
    } else if (m_version == 1) {
        CODES_CHECK(codes_get_double(h, "endStep", &forecastTime), 0);
    }
    m_forecastTimes.push_back(forecastTime);

    long timeUnit;
    CODES_CHECK(codes_get_long(h, "stepUnits", &timeUnit), 0);

    if (timeUnit == 0) {
        // Minutes
        forecastTime /= 1440;
    } else if (timeUnit == 1) {
        // Hours
        forecastTime /= 24;
    } else if (timeUnit == 2) {
        // Days -> nothing to do
    } else {
        throw exception(_("Error reading grib file: unlisted time unit."));
    }

    if (refTime > 100) {
        refTime /= 100;
    }
    double time = refDate + refTime / 24 + forecastTime;
    m_times.push_back(time);
}

void asFileGrib::ExtractLevel(codes_handle* h) {
    // Keys: https://apps.ecmwf.int/codes/grib/format/edition-independent/3/

    // Get level type
    size_t typeLength = 255;
    char* typeVal = NULL;
    typeVal = static_cast<char*>(malloc(typeLength * sizeof(char)));
    CODES_CHECK(codes_get_string(h, "typeOfLevel", &typeVal[0], &typeLength), 0);
    wxString type(typeVal, wxConvUTF8);
    free(typeVal);
    m_levelTypesStr.push_back(type);

    // Get level type code
    long typeCode;
    if (m_version == 2) {
        wxASSERT(codes_is_defined(h, "typeOfFirstFixedSurface"));
        CODES_CHECK(codes_get_long(h, "typeOfFirstFixedSurface", &typeCode), 0);
    } else if (m_version == 1) {
        wxASSERT(codes_is_defined(h, "indicatorOfTypeOfLevel"));
        CODES_CHECK(codes_get_long(h, "indicatorOfTypeOfLevel", &typeCode), 0);
    } else {
        throw exception(_("Error reading grib file: type of level not found."));
    }
    m_levelTypes.push_back((int)typeCode);

    // Get level value
    double level;
    CODES_CHECK(codes_get_double(h, "level", &level), 0);
    if (type.IsSameAs("isobaricInPa")) {
        level /= 100;
    }
    m_levels.push_back(level);
}

void asFileGrib::ExtractAxes(codes_handle* h) {
    // Keys: https://apps.ecmwf.int/codes/grib/format/edition-independent/1/
    long latsNb;
    CODES_CHECK(codes_get_long(h, "Nj", &latsNb), 0);
    long lonNb;
    CODES_CHECK(codes_get_long(h, "Ni", &lonNb), 0);
    double latStart;
    CODES_CHECK(codes_get_double(h, "latitudeOfFirstGridPointInDegrees", &latStart), 0);
    double lonStart;
    CODES_CHECK(codes_get_double(h, "longitudeOfFirstGridPointInDegrees", &lonStart), 0);
    double latEnd;
    CODES_CHECK(codes_get_double(h, "latitudeOfLastGridPointInDegrees", &latEnd), 0);
    double lonEnd;
    CODES_CHECK(codes_get_double(h, "longitudeOfLastGridPointInDegrees", &lonEnd), 0);
    if (lonEnd < lonStart) {
        lonStart -= 360;
    }

    a1d lonAxis = a1d::LinSpaced(lonNb, lonStart, lonEnd);
    a1d latAxis = a1d::LinSpaced(latsNb, latStart, latEnd);

    m_xAxes.push_back(lonAxis);
    m_yAxes.push_back(latAxis);
}

void asFileGrib::ExtractGribCode(codes_handle* h) {
    if (m_version == 2) {
        // Get discipline
        long discipline;
        CODES_CHECK(codes_get_long(h, "discipline", &discipline), 0);
        m_parameterCode1.push_back((int)discipline);

        // Get category
        long category;
        CODES_CHECK(codes_get_long(h, "parameterCategory", &category), 0);
        m_parameterCode2.push_back((int)category);

        // Get parameter number
        long number;
        CODES_CHECK(codes_get_long(h, "parameterNumber", &number), 0);
        m_parameterCode3.push_back((int)number);

    } else if (m_version == 1) {
        m_parameterCode1.push_back(0);

        // Get category
        long category;
        CODES_CHECK(codes_get_long(h, "table2Version", &category), 0);
        m_parameterCode2.push_back((int)category);

        // Get parameter number
        long number;
        CODES_CHECK(codes_get_long(h, "indicatorOfParameter", &number), 0);
        m_parameterCode3.push_back((int)number);
    }
}

bool asFileGrib::CheckGribErrorCode(int ierr) const {
    if (ierr == CODES_SUCCESS) {
        return true;
    }

    wxLogError(_("Grib error (file %s): %s"), m_fileName.GetFullName(), codes_get_error_message(ierr));

    return false;
}

bool asFileGrib::GetXaxis(a1d& uaxis) const {
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);
    wxASSERT(m_xAxes.size() > m_index);

    uaxis = m_xAxes[m_index];

    return true;
}

bool asFileGrib::GetYaxis(a1d& vaxis) const {
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);
    wxASSERT(m_yAxes.size() > m_index);

    vaxis = m_yAxes[m_index];

    return true;
}

bool asFileGrib::GetLevels(a1d& levels) const {
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);

    vd realLevels;
    double lastVal = -1;
    for (double level : m_levels) {
        if (level != lastVal) {
            realLevels.push_back(level);
            lastVal = level;
        }
    }

    levels.resize(realLevels.size());

    for (int i = 0; i < realLevels.size(); ++i) {
        levels[i] = realLevels[i];
    }

    return true;
}

vd asFileGrib::GetRealTimeArray() const {
    wxASSERT(m_opened);

    // Get independent time entries
    vd realTimeArray;
    double lastTimeVal = 0;

    for (double time : m_times) {
        if (time != lastTimeVal) {
            realTimeArray.push_back(time);
            lastTimeVal = time;
        }
    }

    return realTimeArray;
}

double asFileGrib::GetTimeStart() const {
    wxASSERT(m_opened);

    return GetRealTimeArray()[0];
}

double asFileGrib::GetTimeEnd() const {
    wxASSERT(m_opened);

    vd realTimeArray = GetRealTimeArray();

    return realTimeArray[realTimeArray.size() - 1];
}

int asFileGrib::GetTimeLength() const {
    wxASSERT(m_opened);

    return GetRealTimeArray().size();
}

double asFileGrib::GetTimeStepHours() const {
    wxASSERT(m_opened);

    vd realTimeArray = GetRealTimeArray();

    if (realTimeArray.size() == 1) {
        return 0;
    }

    return 24 * (realTimeArray[1] - realTimeArray[0]);
}

vd asFileGrib::GetRealReferenceDateArray() const {
    wxASSERT(m_opened);
    wxASSERT(m_forecastTimes.size() == m_refDates.size());

    // Get independent time entries
    vd refDateArray;
    double lastTimeVal = -1;

    for (int i = 0; i < m_refDates.size(); ++i) {
        if (m_times[i] != lastTimeVal) {
            refDateArray.push_back(m_refDates[i]);
            lastTimeVal = m_times[i];
        }
    }

    return refDateArray;
}

vd asFileGrib::GetRealReferenceTimeArray() const {
    wxASSERT(m_opened);
    wxASSERT(m_forecastTimes.size() == m_refTimes.size());

    // Get independent time entries
    vd refTimeArray;
    double lastTimeVal = -1;

    for (int i = 0; i < m_refTimes.size(); ++i) {
        if (m_times[i] != lastTimeVal) {
            refTimeArray.push_back(m_refTimes[i]);
            lastTimeVal = m_times[i];
        }
    }

    return refTimeArray;
}

vd asFileGrib::GetRealForecastTimeArray() const {
    wxASSERT(m_opened);

    // Get independent time entries
    vd forecastTimeArray;
    double lastTimeVal = -1;

    for (int i = 0; i < m_forecastTimes.size(); ++i) {
        if (m_times[i] != lastTimeVal) {
            forecastTimeArray.push_back(m_forecastTimes[i]);
            lastTimeVal = m_times[i];
        }
    }

    return forecastTimeArray;
}

bool asFileGrib::SetIndexPosition(const vi& gribCode, const float level, const bool useWarnings) {
    wxASSERT(gribCode.size() == 4);

    // Find corresponding data
    m_index = asNOT_FOUND;
    for (int i = 0; i < m_parameterCode3.size(); ++i) {
        if (m_parameterCode1[i] == gribCode[0] && m_parameterCode2[i] == gribCode[1] &&
            m_parameterCode3[i] == gribCode[2] && m_levelTypes[i] == gribCode[3] && m_levels[i] == level) {
            m_index = i;
            return true;
        }
    }

    if (useWarnings) {
        wxLogWarning(_("The desired parameter / level (%.0f) was not found in the file %s."), level,
                     m_fileName.GetFullName());
    } else {
        wxLogVerbose(_("The desired parameter / level (%.0f) was not found in the file %s."), level,
                     m_fileName.GetFullName());
    }

    return false;
}

bool asFileGrib::SetIndexPositionAnyLevel(const vi gribCode) {
    wxASSERT(gribCode.size() == 4);

    if (m_parameterCode1.empty()) {
        wxLogError(_("The file %s is empty."), m_fileName.GetFullName());
        return false;
    }

    // Find corresponding data
    m_index = asNOT_FOUND;
    for (int i = 0; i < m_parameterCode3.size(); ++i) {
        if (m_parameterCode1[i] == gribCode[0] && m_parameterCode2[i] == gribCode[1] &&
            m_parameterCode3[i] == gribCode[2] && m_levelTypes[i] == gribCode[3]) {
            m_index = i;
            return true;
        }
    }

    wxLogError(_("The desired parameter was not found in the file %s."), m_fileName.GetFullName());

    return false;
}

bool asFileGrib::GetVarArray(const int IndexStart[], const int IndexCount[], float* pValue) {
    wxASSERT(m_opened);
    wxASSERT(m_index != asNOT_FOUND);

    vd timeArray = GetRealTimeArray();
    vd forecastTimeArray = GetRealForecastTimeArray();
    vd referenceDateArray = GetRealReferenceDateArray();
    vd referenceTimeArray = GetRealReferenceTimeArray();
    wxASSERT(forecastTimeArray.size() == timeArray.size());
    wxASSERT(referenceDateArray.size() == timeArray.size());
    wxASSERT(referenceTimeArray.size() == timeArray.size());

    int iTimeStart = IndexStart[0];
    int iTimeEnd = IndexStart[0] + IndexCount[0] - 1;
    int iLonStart = IndexStart[1];
    int iLonEnd = IndexStart[1] + IndexCount[1] - 1;
    int iLatStart = IndexStart[2];
    int iLatEnd = IndexStart[2] + IndexCount[2] - 1;
    auto nLons = (int)m_xAxes[m_index].size();
    auto nLats = (int)m_yAxes[m_index].size();

    int finalIndex = 0;
    vd fullTimeArray(IndexCount[0]);
    std::copy(timeArray.begin() + iTimeStart, timeArray.begin() + iTimeEnd + 1, fullTimeArray.begin());

    int iTime = iTimeStart;

    for (auto& date : fullTimeArray) {
        wxASSERT(iTime < timeArray.size());

        wxString refDate = asTime::GetStringTime(referenceDateArray[iTime], YYYYMMDD);
        char refDateChar[10];
        strncpy(refDateChar, static_cast<const char*>(refDate.mb_str(wxConvUTF8)), 9);
        refDateChar[sizeof(refDateChar) - 1] = '\0';
        double refTime = referenceTimeArray[iTime];
        double forecastTime = forecastTimeArray[iTime];

        codes_index* index = nullptr;
        codes_handle* h = nullptr;
        int err = 0;
        int count = 0;

        if (m_version == 2) {
            index = codes_index_new(
                NULL, "discipline,parameterCategory,parameterNumber,level,dataDate,dataTime,endStep", &err);
        } else if (m_version == 1) {
            index = codes_index_new(NULL, "table2Version,indicatorOfParameter,level,dataDate,dataTime,endStep", &err);
        }

        if (!CheckGribErrorCode(err)) {
            return false;
        }

        err = codes_index_add_file(index, m_fileName.GetFullPath().mb_str());
        if (!CheckGribErrorCode(err)) {
            return false;
        }

        if (m_version == 2) {
            err = codes_index_select_long(index, "discipline", m_parameterCode1[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_long(index, "parameterCategory", m_parameterCode2[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_long(index, "parameterNumber", m_parameterCode3[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_double(index, "level", m_levels[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_string(index, "dataDate", refDateChar);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_double(index, "dataTime", refTime);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_double(index, "endStep", forecastTime);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
        } else if (m_version == 1) {
            err = codes_index_select_long(index, "table2Version", m_parameterCode2[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_long(index, "indicatorOfParameter", m_parameterCode3[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_double(index, "level", m_levels[m_index]);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_string(index, "dataDate", refDateChar);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_double(index, "dataTime", refTime);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            err = codes_index_select_double(index, "endStep", forecastTime);
            if (!CheckGribErrorCode(err)) {
                return false;
            }
        }

        while ((h = codes_handle_new_from_index(index, &err)) != NULL) {
            if (!CheckGribErrorCode(err)) {
                return false;
            }
            if (count > 0) {
                wxLogWarning(_("Multiple messages found in GRIB file for the given constraints."));
                return true;
            }
            count++;

            // Get data
            double* values = nullptr;
            size_t valuesLenth = 0;
            CODES_CHECK(codes_get_size(h, "values", &valuesLenth), nullptr);
            values = new double[valuesLenth + 1];
            CODES_CHECK(codes_get_double_array(h, "values", values, &valuesLenth), nullptr);

            if (nLats > 0 && m_yAxes[m_index][0] > m_yAxes[m_index][1]) {
                for (int iLat = nLats - 1; iLat >= 0; iLat--) {
                    if (iLat >= iLatStart && iLat <= iLatEnd) {
                        for (int iLon = 0; iLon < nLons; iLon++) {
                            if (iLon >= iLonStart && iLon <= iLonEnd) {
                                pValue[finalIndex] = (float)values[iLat * nLons + iLon];
                                finalIndex++;
                            }
                        }
                    }
                }
            } else {
                for (int iLat = 0; iLat < nLats; iLat++) {
                    if (iLat >= iLatStart && iLat <= iLatEnd) {
                        for (int iLon = 0; iLon < nLons; iLon++) {
                            if (iLon >= iLonStart && iLon <= iLonEnd) {
                                pValue[finalIndex] = (float)values[iLat * nLons + iLon];
                                finalIndex++;
                            }
                        }
                    }
                }
            }

            delete[] (values);
            codes_handle_delete(h);
        }

        if (count == 0) {
            wxLogError(_("GRIB message not found for the given constraints."));
            return false;
        }

        codes_index_delete(index);

        iTime++;
    }

    return true;
}