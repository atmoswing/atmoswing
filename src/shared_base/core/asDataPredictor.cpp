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

#include "asDataPredictor.h"

#include <asTimeArray.h>
#include <asGeoAreaCompositeGrid.h>
#include <wx/dir.h>


asDataPredictor::asDataPredictor(const wxString &dataId)
{
    m_dataId = dataId;
    m_level = 0;
    m_product = wxEmptyString;
    m_subFolder = wxEmptyString;
    m_isPreprocessed = false;
    m_isEnsemble = false;
    m_transformedBy = wxEmptyString;
    m_canBeClipped = true;
    m_latPtsnb = 0;
    m_lonPtsnb = 0;
    m_preprocessMethod = wxEmptyString;
    m_initialized = false;
    m_axesChecked = false;
    m_timeZoneHours = 0.0;
    m_timeStepHours = 0.0;
    m_firstTimeStepHours = 0.0;
    m_xAxisStep = 0.0f;
    m_yAxisStep = 0.0f;
    m_xAxisShift = 0.0f;
    m_yAxisShift = 0.0f;
    m_fileStructure.dimLatName = wxEmptyString;
    m_fileStructure.dimLonName = wxEmptyString;
    m_fileStructure.dimTimeName = wxEmptyString;
    m_fileStructure.dimLevelName = wxEmptyString;
    m_fileStructure.hasLevelDimension = true;
    m_fileStructure.singleLevel = false;
    m_fileIndexes.member = 0;
    m_fileExtension = wxEmptyString;
    int arr[] = {asNOT_FOUND, asNOT_FOUND, asNOT_FOUND, asNOT_FOUND};
    AssignGribCode(arr);

    if(dataId.Contains('/')) {
        wxString levelType = dataId.BeforeFirst('/');
        m_product = levelType;
        m_dataId = dataId.AfterFirst('/');
    } else {
        wxLogVerbose(_("The data ID (%s) does not contain the level type"), dataId);
    }

}

asDataPredictor::~asDataPredictor()
{

}

asDataPredictor::Parameter asDataPredictor::StringToParameterEnum(const wxString &ParameterStr)
{
    if (ParameterStr.CmpNoCase("AirTemperature") == 0) {
        return AirTemperature;
    } else if (ParameterStr.CmpNoCase("GeopotentialHeight") == 0) {
        return GeopotentialHeight;
    } else if (ParameterStr.CmpNoCase("PrecipitableWater") == 0) {
        return PrecipitableWater;
    } else if (ParameterStr.CmpNoCase("PrecipitationRate") == 0) {
        return PrecipitationRate;
    } else if (ParameterStr.CmpNoCase("RelativeHumidity") == 0) {
        return RelativeHumidity;
    } else if (ParameterStr.CmpNoCase("SpecificHumidity") == 0) {
        return SpecificHumidity;
    } else if (ParameterStr.CmpNoCase("VerticalVelocity") == 0) {
        return VerticalVelocity;
    } else if (ParameterStr.CmpNoCase("Wind") == 0) {
        return Wind;
    } else if (ParameterStr.CmpNoCase("Uwind") == 0) {
        return Uwind;
    } else if (ParameterStr.CmpNoCase("Vwind") == 0) {
        return Vwind;
    } else if (ParameterStr.CmpNoCase("SurfaceLiftedIndex") == 0) {
        return SurfaceLiftedIndex;
    } else if (ParameterStr.CmpNoCase("PotentialTemperature") == 0) {
        return PotentialTemperature;
    } else if (ParameterStr.CmpNoCase("Pressure") == 0) {
        return Pressure;
    } else if (ParameterStr.CmpNoCase("PotentialEvaporation") == 0) {
        return PotentialEvaporation;
    } else if (ParameterStr.CmpNoCase("SoilTemperature") == 0) {
        return SoilTemperature;
    } else if (ParameterStr.CmpNoCase("CloudCover") == 0) {
        return CloudCover;
    } else if (ParameterStr.CmpNoCase("SoilMoisture") == 0) {
        return SoilMoisture;
    } else if (ParameterStr.CmpNoCase("SnowWaterEquivalent") == 0) {
        return SnowWaterEquivalent;
    } else if (ParameterStr.CmpNoCase("Radiation") == 0) {
        return Radiation;
    } else if (ParameterStr.CmpNoCase("MomentumFlux") == 0) {
        return MomentumFlux;
    } else if (ParameterStr.CmpNoCase("GravityWaveStress") == 0) {
        return GravityWaveStress;
    } else if (ParameterStr.CmpNoCase("SeaSurfaceTemperature") == 0) {
        return SeaSurfaceTemperature;
    } else if (ParameterStr.CmpNoCase("SST") == 0) {
        return SeaSurfaceTemperature;
    } else if (ParameterStr.CmpNoCase("SeaSurfaceTemperatureAnomaly") == 0) {
        return SeaSurfaceTemperatureAnomaly;
    } else if (ParameterStr.CmpNoCase("SSTAnomaly") == 0) {
        return SeaSurfaceTemperatureAnomaly;
    } else {
        asThrowException(wxString::Format(_("The Parameter enumeration (%s) entry doesn't exists"), ParameterStr));
    }
    return GeopotentialHeight;
}

wxString asDataPredictor::ParameterEnumToString(asDataPredictor::Parameter dataParameter)
{
    switch (dataParameter) {
        case (AirTemperature):
            return "AirTemperature";
        case (GeopotentialHeight):
            return "GeopotentialHeight";
        case (PrecipitableWater):
            return "PrecipitableWater";
        case (PrecipitationRate):
            return "PrecipitationRate";
        case (RelativeHumidity):
            return "RelativeHumidity";
        case (SpecificHumidity):
            return "SpecificHumidity";
        case (VerticalVelocity):
            return "VerticalVelocity";
        case (Wind):
            return "Wind";
        case (Uwind):
            return "Uwind";
        case (Vwind):
            return "Vwind";
        case (SurfaceLiftedIndex):
            return "SurfaceLiftedIndex";
        case (PotentialTemperature):
            return "PotentialTemperature";
        case (Pressure):
            return "Pressure";
        case (PotentialEvaporation):
            return "PotentialEvaporation";
        case (SoilTemperature):
            return "SoilTemperature";
        case (CloudCover):
            return "CloudCover";
        case (SoilMoisture):
            return "SoilMoisture";
        case (SnowWaterEquivalent):
            return "SnowWaterEquivalent";
        case (Radiation):
            return "Radiation";
        case (MomentumFlux):
            return "MomentumFlux";
        case (GravityWaveStress):
            return "GravityWaveStress";
        case (SeaSurfaceTemperature):
            return "SeaSurfaceTemperature";
        case (SeaSurfaceTemperatureAnomaly):
            return "SeaSurfaceTemperatureAnomaly";
        default:
            wxLogError(_("The given data parameter type in unknown."));
    }
    return wxEmptyString;
}

asDataPredictor::Unit asDataPredictor::StringToUnitEnum(const wxString &UnitStr)
{

    if (UnitStr.CmpNoCase("nb") == 0) {
        return nb;
    } else if (UnitStr.CmpNoCase("number") == 0) {
        return nb;
    } else if (UnitStr.CmpNoCase("mm") == 0) {
        return mm;
    } else if (UnitStr.CmpNoCase("m") == 0) {
        return m;
    } else if (UnitStr.CmpNoCase("km") == 0) {
        return km;
    } else if (UnitStr.CmpNoCase("percent") == 0) {
        return percent;
    } else if (UnitStr.CmpNoCase("fraction") == 0) {
        return fraction;
    } else if (UnitStr.CmpNoCase("%") == 0) {
        return percent;
    } else if (UnitStr.CmpNoCase("degC") == 0) {
        return degC;
    } else if (UnitStr.CmpNoCase("degK") == 0) {
        return degK;
    } else if (UnitStr.CmpNoCase("Pa") == 0) {
        return Pa;
    } else if (UnitStr.CmpNoCase("Pa_s") == 0) {
        return Pa_s;
    } else if (UnitStr.CmpNoCase("Pa/s") == 0) {
        return Pa_s;
    } else if (UnitStr.CmpNoCase("kg_kg") == 0) {
        return kg_kg;
    } else if (UnitStr.CmpNoCase("kg/kg") == 0) {
        return kg_kg;
    } else if (UnitStr.CmpNoCase("m_s") == 0) {
        return m_s;
    } else if (UnitStr.CmpNoCase("m/s") == 0) {
        return m_s;
    } else if (UnitStr.CmpNoCase("W_m2") == 0) {
        return W_m2;
    } else if (UnitStr.CmpNoCase("W/m2") == 0) {
        return W_m2;
    } else if (UnitStr.CmpNoCase("kg_m2") == 0) {
        return kg_m2;
    } else if (UnitStr.CmpNoCase("kg/m2") == 0) {
        return kg_m2;
    } else if (UnitStr.CmpNoCase("kg_m2_s") == 0) {
        return kg_m2_s;
    } else if (UnitStr.CmpNoCase("kg/m2/s") == 0) {
        return kg_m2_s;
    } else if (UnitStr.CmpNoCase("N_m2") == 0) {
        return N_m2;
    } else if (UnitStr.CmpNoCase("N/m2") == 0) {
        return N_m2;
    } else {
        asThrowException(wxString::Format(_("The Unit enumeration (%s) entry doesn't exists"), UnitStr));
    }
    return m;
}

bool asDataPredictor::SetData(VArray2DFloat &val)
{
    wxASSERT(m_time.size() > 0);
    wxASSERT((int) m_time.size() == (int) val.size());

    m_latPtsnb = (int) val[0].rows();
    m_lonPtsnb = (int) val[0].cols();
    m_data.clear();
    m_data = val;

    return true;
}

bool asDataPredictor::CheckFilesPresence(VectorString &filesList)
{
    if (filesList.size() == 0) {
        wxLogError(_("Empty files list."));
        return false;
    }

    int nbDirsToRemove = 0;

    for (int i = 0; i < filesList.size(); i++) {
        if (i > 0 && nbDirsToRemove > 0) {
            wxFileName fileName(filesList[i]);
            for (int j = 0; j < nbDirsToRemove; ++j) {
                fileName.RemoveLastDir();
            }
            filesList[i] = fileName.GetFullPath();
        }

        if (!wxFile::Exists(filesList[i])) {
            // Search recursively in the parent directory
            wxFileName fileName(filesList[i]);
            while (true) {
                // Check for wildcards
                if (wxIsWild(fileName.GetPath())) {
                    wxLogError(_("No wildcard is yet authorized in the path (%s)"), fileName.GetPath());
                    return false;
                } else if (wxIsWild(fileName.GetFullName())) {
                    wxArrayString files;
                    size_t nb = wxDir::GetAllFiles(fileName.GetPath(), &files, fileName.GetFullName());
                    if (nb == 1) {
                        filesList[i] = files[0];
                        break;
                    } else if (nb > 1) {
                        wxLogError(_("Multiple files were found matching the name %s:"), fileName.GetFullName());
                        for (int j = 0; j < nb; ++j) {
                            wxLogError(files[j]);
                        }
                        return false;
                    }
                }

                if (i == 0) {
                    if (fileName.GetDirCount()<2) {
                        wxLogError(_("File not found: %s"), filesList[i]);
                        return false;
                    }

                    fileName.RemoveLastDir();
                    nbDirsToRemove++;
                    if (fileName.Exists()) {
                        filesList[i] = fileName.GetFullPath();
                        break;
                    }
                }
            }
        }
    }

    return true;
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid *desiredArea, asTimeArray &timeArray)
{
    if (!m_initialized) {
        if (!Init()) {
            wxLogError(_("Error at initialization of the predictor dataset %s."), m_datasetName);
            return false;
        }
    }

    try {
        // Check the time array
        if (!CheckTimeArray(timeArray)) {
            wxLogError(_("The time array is not valid to load data."));
            return false;
        }

        // Create a new area matching the dataset
        asGeoAreaCompositeGrid *dataArea = CreateMatchingArea(desiredArea);

        // Store time array
        m_time = timeArray.GetTimeArray();
        m_fileIndexes.timeStep = wxMax(timeArray.GetTimeStepHours() / m_timeStepHours, 1);

        // The desired level
        if (desiredArea) {
            m_level = desiredArea->GetComposite(0).GetLevel();
        }

        // Number of composites
        int compositesNb = 1;
        if (dataArea) {
            compositesNb = dataArea->GetNbComposites();
            wxASSERT(compositesNb > 0);
        }

        // Extract composite data from files
        VVArray2DFloat compositeData(compositesNb);
        if (!ExtractFromFiles(dataArea, timeArray, compositeData)) {
            wxLogWarning(_("Extracting data from files failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Merge the composites into m_data
        if (!MergeComposites(compositeData, dataArea)) {
            wxLogError(_("Merging the composites failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Interpolate the loaded data on the desired grid
        if (desiredArea && !InterpolateOnGrid(dataArea, desiredArea)) {
            wxLogError(_("Interpolation failed."));
            wxDELETE(dataArea);
            return false;
        }

        // Check the data container length
        if ((unsigned) m_time.size() != m_data.size()) {
            wxLogError(_("The date and the data array lengths do not match."));
            wxDELETE(dataArea);
            return false;
        }

        wxDELETE(dataArea);
    } catch (std::bad_alloc &ba) {
        wxString msg(ba.what(), wxConvUTF8);
        wxLogError(_("Bad allocation caught when loading data: %s"), msg);
        return false;
    } catch (asException &e) {
        wxString fullMessage = e.GetFullMessage();
        if (!fullMessage.IsEmpty()) {
            wxLogError(fullMessage);
        }
        wxLogError(_("Failed to load data."));
        return false;
    }

    return true;
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid &desiredArea, asTimeArray &timeArray)
{
    return Load(&desiredArea, timeArray);
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid &desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(&desiredArea, timeArray);
}

bool asDataPredictor::Load(asGeoAreaCompositeGrid *desiredArea, double date)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();

    return Load(desiredArea, timeArray);
}

bool asDataPredictor::LoadFullArea(double date, float level)
{
    asTimeArray timeArray(date, asTimeArray::SingleDay);
    timeArray.Init();
    m_level = level;

    return Load(NULL, timeArray);
}


bool asDataPredictor::ExtractFromNetcdfFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                   asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Open the NetCDF file
    ThreadsManager().CritSectionNetCDF().Enter();
    asFileNetcdf ncFile(fileName, asFileNetcdf::ReadOnly);
    if (!ncFile.Open()) {
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(ncFile, dataArea, timeArray, compositeData)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Adjust axes if necessary
    dataArea = AdjustAxes(dataArea, compositeData);
    if (dataArea) {
        wxASSERT(dataArea->GetNbComposites() > 0);
    }

    // Get indexes
    if (!GetAxesIndexes(dataArea, timeArray, compositeData)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        return false;
    }

    // Load data
    if (!GetDataFromFile(ncFile, compositeData)) {
        ncFile.Close();
        ThreadsManager().CritSectionNetCDF().Leave();
        wxFAIL;
        return false;
    }

    // Close the nc file
    ncFile.Close();
    ThreadsManager().CritSectionNetCDF().Leave();

    return true;
}

bool asDataPredictor::ExtractFromGribFile(const wxString &fileName, asGeoAreaCompositeGrid *&dataArea,
                                                 asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Open the Grib file
    ThreadsManager().CritSectionGrib().Enter();
    asFileGrib2 gbFile(fileName, asFileGrib2::ReadOnly);
    if (!gbFile.Open()) {
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Set index position
    if (!gbFile.SetIndexPosition(m_gribCode, m_level)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Parse file structure
    if (!ParseFileStructure(gbFile, dataArea, timeArray, compositeData)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Adjust axes if necessary
    dataArea = AdjustAxes(dataArea, compositeData);
    if (dataArea) {
        wxASSERT(dataArea->GetNbComposites() > 0);
    }

    // Get indexes
    if (!GetAxesIndexes(dataArea, timeArray, compositeData)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Load data
    if (!GetDataFromFile(gbFile, compositeData)) {
        gbFile.Close();
        ThreadsManager().CritSectionGrib().Leave();
        wxFAIL;
        return false;
    }

    // Close the nc file
    gbFile.Close();
    ThreadsManager().CritSectionGrib().Leave();

    return true;
}

bool asDataPredictor::ParseFileStructure(asFileNetcdf &ncFile, asGeoAreaCompositeGrid *&dataArea,
                                         asTimeArray &timeArray, VVArray2DFloat &compositeData)
{
    // Get full axes from the netcdf file
    m_fileStructure.axisLon = Array1DFloat(ncFile.GetVarLength(m_fileStructure.dimLonName));
    m_fileStructure.axisLat = Array1DFloat(ncFile.GetVarLength(m_fileStructure.dimLatName));

    wxASSERT(ncFile.GetVarType(m_fileStructure.dimLonName) == ncFile.GetVarType(m_fileStructure.dimLatName));
    nc_type ncTypeAxes = ncFile.GetVarType(m_fileStructure.dimLonName);
    switch (ncTypeAxes) {
        case NC_FLOAT:
            ncFile.GetVar(m_fileStructure.dimLonName, &m_fileStructure.axisLon[0]);
            ncFile.GetVar(m_fileStructure.dimLatName, &m_fileStructure.axisLat[0]);
            break;
        case NC_DOUBLE: {
            Array1DDouble axisLonDouble(ncFile.GetVarLength(m_fileStructure.dimLonName));
            Array1DDouble axisLatDouble(ncFile.GetVarLength(m_fileStructure.dimLatName));
            ncFile.GetVar(m_fileStructure.dimLonName, &axisLonDouble[0]);
            ncFile.GetVar(m_fileStructure.dimLatName, &axisLatDouble[0]);
            for (int i = 0; i < axisLonDouble.size(); ++i) {
                m_fileStructure.axisLon[i] = (float) axisLonDouble[i];
            }
            for (int i = 0; i < axisLatDouble.size(); ++i) {
                m_fileStructure.axisLat[i] = (float) axisLatDouble[i];
            }
        }
            break;
        default:
            wxLogError(_("Variable type not supported yet for the level dimension."));
            return false;
    }

    if (m_fileStructure.hasLevelDimension) {
        m_fileStructure.axisLevel = Array1DFloat(ncFile.GetVarLength(m_fileStructure.dimLevelName));

        nc_type ncTypeLevel = ncFile.GetVarType(m_fileStructure.dimLevelName);
        switch (ncTypeLevel) {
            case NC_FLOAT:
                ncFile.GetVar(m_fileStructure.dimLevelName, &m_fileStructure.axisLevel[0]);
                break;
            case NC_INT: {
                Array1DInt axisLevelInt(ncFile.GetVarLength(m_fileStructure.dimLevelName));
                ncFile.GetVar(m_fileStructure.dimLevelName, &axisLevelInt[0]);
                for (int i = 0; i < axisLevelInt.size(); ++i) {
                    m_fileStructure.axisLevel[i] = (float) axisLevelInt[i];
                }
            }
                break;
            case NC_DOUBLE: {
                Array1DDouble axisLevelDouble(ncFile.GetVarLength(m_fileStructure.dimLevelName));
                ncFile.GetVar(m_fileStructure.dimLevelName, &axisLevelDouble[0]);
                for (int i = 0; i < axisLevelDouble.size(); ++i) {
                    m_fileStructure.axisLevel[i] = (float) axisLevelDouble[i];
                }
            }
                break;
            default:
                wxLogError(_("Variable type not supported yet for the level dimension."));
                return false;
        }
    }

    // Time dimension takes ages to load !! Avoid and get the first value.
    m_fileStructure.axisTimeLength = ncFile.GetVarLength(m_fileStructure.dimTimeName);

    double timeFirstVal, timeLastVal;
    nc_type ncTypeTime = ncFile.GetVarType(m_fileStructure.dimTimeName);
    switch (ncTypeTime) {
        case NC_DOUBLE:
            timeFirstVal = ncFile.GetVarOneDouble(m_fileStructure.dimTimeName, 0);
            timeLastVal = ncFile.GetVarOneDouble(m_fileStructure.dimTimeName, m_fileStructure.axisTimeLength - 1);
            break;
        case NC_FLOAT:
            timeFirstVal = (double)ncFile.GetVarOneFloat(m_fileStructure.dimTimeName, 0);
            timeLastVal = (double)ncFile.GetVarOneFloat(m_fileStructure.dimTimeName, m_fileStructure.axisTimeLength - 1);
            break;
        case NC_INT:
            timeFirstVal = (double)ncFile.GetVarOneInt(m_fileStructure.dimTimeName, 0);
            timeLastVal = (double)ncFile.GetVarOneInt(m_fileStructure.dimTimeName, m_fileStructure.axisTimeLength - 1);
            break;
        default:
            wxLogError(_("Variable type not supported yet for the time dimension."));
            return false;
    }

    double refValue = NaNDouble;
    if (m_datasetId.IsSameAs("NASA_MERRA_2", false) || m_datasetId.IsSameAs("NASA_MERRA_2_subset", false)) {
        wxString refValueStr = ncFile.GetAttString("units", "time");
        refValueStr = refValueStr.Remove(0, 14);
        refValue = asTime::GetTimeFromString(refValueStr);
    } else if(m_datasetId.IsSameAs("NCEP_CFSR_subset", false)) {
        wxString refValueStr = ncFile.GetAttString("units", "time");
        refValueStr = refValueStr.Mid(12, 10);
        refValue = asTime::GetTimeFromString(refValueStr);
    }

    m_fileStructure.axisTimeFirstValue = ConvertToMjd(timeFirstVal, refValue);
    m_fileStructure.axisTimeLastValue = ConvertToMjd(timeLastVal, refValue);

    return CheckFileStructure();
}

bool asDataPredictor::ParseFileStructure(asFileGrib2 &gbFile, asGeoAreaCompositeGrid *&dataArea, asTimeArray &timeArray,
                                         VVArray2DFloat &compositeData)
{
    // Get full axes from the file
    gbFile.GetXaxis(m_fileStructure.axisLon);
    gbFile.GetYaxis(m_fileStructure.axisLat);

    if (m_fileStructure.hasLevelDimension && !m_fileStructure.singleLevel) {
        wxLogError(_("The level dimension is not yet implemented for Grib files."));
        return false;
    }

    // Yet handle a unique time value per file.
    m_fileStructure.axisTimeLength = 1;
    m_fileStructure.axisTimeFirstValue = gbFile.GetTime();
    m_fileStructure.axisTimeLastValue = gbFile.GetTime();

    return CheckFileStructure();
}

bool asDataPredictor::CheckFileStructure()
{
    // Check for breaks in the longitude axis.
    if (m_fileStructure.axisLon.size() > 1) {
        if (m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1] < m_fileStructure.axisLon[0]) {
            int i_break = 0;
            for (int i = 1; i < m_fileStructure.axisLon.size(); ++i) {
                if (m_fileStructure.axisLon[i] < m_fileStructure.axisLon[i-1]) {
                    if (i_break != 0) {
                        wxLogError(_("Longitude axis seems not consistent (multiple breaks)."));
                        return false;
                    }
                    i_break = i;
                }
            }
            for (int i = i_break; i < m_fileStructure.axisLon.size(); ++i) {
                m_fileStructure.axisLon[i] += 360;
            }
        }
    }

    return true;
}

asGeoAreaCompositeGrid *asDataPredictor::CreateMatchingArea(asGeoAreaCompositeGrid *desiredArea)
{
    if (desiredArea) {
        double dataXmin, dataYmin, dataXstep, dataYstep;
        int dataXptsnb, dataYptsnb;
        wxString gridType = desiredArea->GetGridTypeString();
        if (gridType.IsSameAs("Regular", false)) {
            dataXmin = floor((desiredArea->GetAbsoluteXmin() - m_xAxisShift) / m_xAxisStep) * m_xAxisStep + m_xAxisShift;
            dataYmin = floor((desiredArea->GetAbsoluteYmin() - m_yAxisShift) / m_yAxisStep) * m_yAxisStep + m_yAxisShift;
            double dataXmax = ceil((desiredArea->GetAbsoluteXmax() - m_xAxisShift) / m_xAxisStep) * m_xAxisStep + m_xAxisShift;
            double dataYmax = ceil((desiredArea->GetAbsoluteYmax() - m_yAxisShift) / m_yAxisStep) * m_yAxisStep + m_yAxisShift;
            dataXstep = m_xAxisStep;
            dataYstep = m_yAxisStep;
            dataXptsnb = wxRound((dataXmax - dataXmin) / dataXstep + 1);
            dataYptsnb = wxRound((dataYmax - dataYmin) / dataYstep + 1);
        } else {
            dataXmin = desiredArea->GetAbsoluteXmin();
            dataYmin = desiredArea->GetAbsoluteYmin();
            dataXstep = desiredArea->GetXstep();
            dataYstep = desiredArea->GetYstep();
            dataXptsnb = desiredArea->GetXaxisPtsnb();
            dataYptsnb = desiredArea->GetYaxisPtsnb();
            if (!asTools::IsNaN(m_xAxisStep) && !asTools::IsNaN(m_yAxisStep) &&
                (dataXstep != m_xAxisStep || dataYstep != m_yAxisStep)) {
                wxLogError(_("Interpolation is not allowed on irregular grids."));
                return NULL;
            }
        }

        asGeoAreaCompositeGrid *dataArea = asGeoAreaCompositeGrid::GetInstance(gridType, dataXmin, dataXptsnb,
                                                                               dataXstep, dataYmin, dataYptsnb,
                                                                               dataYstep, desiredArea->GetLevel(),
                                                                               asNONE, asFLAT_ALLOWED);

        // Get indexes steps
        if (gridType.IsSameAs("Regular", false)) {
            m_fileIndexes.lonStep = wxRound(dataArea->GetXstep() / m_xAxisStep);
            m_fileIndexes.latStep = wxRound(dataArea->GetYstep() / m_yAxisStep);
        } else {
            m_fileIndexes.lonStep = 1;
            m_fileIndexes.latStep = 1;
        }

        // Get axes length for preallocation
        m_lonPtsnb = dataArea->GetXaxisPtsnb();
        m_latPtsnb = dataArea->GetYaxisPtsnb();

        return dataArea;
    }

    return NULL;
}

asGeoAreaCompositeGrid *asDataPredictor::AdjustAxes(asGeoAreaCompositeGrid *dataArea, VVArray2DFloat &compositeData)
{
    wxASSERT(m_fileStructure.axisLon.size()>0);
    wxASSERT(m_fileStructure.axisLat.size()>0);

    if (!m_axesChecked) {
        if (dataArea == NULL) {
            // Get axes length for preallocation
            m_lonPtsnb = int(m_fileStructure.axisLon.size());
            m_latPtsnb = int(m_fileStructure.axisLat.size());
            m_axisLon = m_fileStructure.axisLon;
            m_axisLat = m_fileStructure.axisLat;
        } else {
            // Check that requested data do not overtake the file
            for (int i_comp = 0; i_comp < dataArea->GetNbComposites(); i_comp++) {
                Array1DDouble axisLonComp = dataArea->GetXaxisComposite(i_comp);

                wxASSERT(m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1] > m_fileStructure.axisLon[0]);
                wxASSERT(axisLonComp[axisLonComp.size() - 1] >= axisLonComp[0]);

                // Condition for change: The composite must not be fully outside (considered as handled).
                if (axisLonComp[axisLonComp.size() - 1] > m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1] &&
                    axisLonComp[0] <= m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1] ) {
                    // If the last value corresponds to the maximum value of the reference system, create a new composite
                    if (axisLonComp[axisLonComp.size() - 1] == dataArea->GetAxisXmax() && dataArea->GetNbComposites() == 1) {
                        dataArea->SetLastRowAsNewComposite();
                        compositeData = VVArray2DFloat(dataArea->GetNbComposites());
					} else if (axisLonComp[axisLonComp.size() - 1] == dataArea->GetAxisXmax() && dataArea->GetNbComposites() > 1) {
                        dataArea->RemoveLastRowOnComposite(i_comp);
                    } else if (axisLonComp[axisLonComp.size() - 1] != dataArea->GetAxisXmax()) {
                        wxLogVerbose(_("Correcting the longitude extent according to the file limits."));
                        double Xwidth = m_fileStructure.axisLon[m_fileStructure.axisLon.size() - 1] - dataArea->GetAbsoluteXmin();
                        wxASSERT(Xwidth >= 0);
                        int Xptsnb = 1 + Xwidth / dataArea->GetXstep();
                        wxLogDebug(_("xPtsNb = %d."), Xptsnb);
                        asGeoAreaCompositeGrid *newdataArea = asGeoAreaCompositeGrid::GetInstance(
                                dataArea->GetGridTypeString(), dataArea->GetAbsoluteXmin(), Xptsnb, dataArea->GetXstep(),
                                dataArea->GetAbsoluteYmin(), dataArea->GetYaxisPtsnb(), dataArea->GetYstep(),
                                dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLon = dataArea->GetXaxis();
            m_axisLon.resize(axisLon.size());
            for (int i = 0; i < axisLon.size(); i++) {
                m_axisLon[i] = (float) axisLon[i];
            }
            m_lonPtsnb = dataArea->GetXaxisPtsnb();
            wxASSERT_MSG(m_axisLon.size() == m_lonPtsnb,
                         wxString::Format("m_axisLon.size()=%d, m_lonPtsnb=%d", (int) m_axisLon.size(), m_lonPtsnb));

            // Check that requested data do not overtake the file
            for (int i_comp = 0; i_comp < dataArea->GetNbComposites(); i_comp++) {
                Array1DDouble axisLatComp = dataArea->GetYaxisComposite(i_comp);

                if (m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1] > m_fileStructure.axisLat[0]) {
                    wxASSERT(axisLatComp[axisLatComp.size() - 1] >= axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size() - 1] > m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1] &&
                        axisLatComp[0] < m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1]) {
                        wxLogVerbose(_("Correcting the latitude extent according to the file limits."));
                        double Ywidth = m_fileStructure.axisLat[m_fileStructure.axisLat.size() - 1] - dataArea->GetAbsoluteYmin();
                        wxASSERT(Ywidth >= 0);
                        int Yptsnb = 1 + Ywidth / dataArea->GetYstep();
                        wxLogDebug(_("yPtsNb = %d."), Yptsnb);
                        asGeoAreaCompositeGrid *newdataArea = asGeoAreaCompositeGrid::GetInstance(
                                dataArea->GetGridTypeString(), dataArea->GetAbsoluteXmin(), dataArea->GetXaxisPtsnb(),
                                dataArea->GetXstep(), dataArea->GetAbsoluteYmin(), Yptsnb, dataArea->GetYstep(),
                                dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }

                } else {
                    wxASSERT(axisLatComp[axisLatComp.size() - 1] >= axisLatComp[0]);

                    // Condition for change: The composite must not be fully outside (considered as handled).
                    if (axisLatComp[axisLatComp.size() - 1] > m_fileStructure.axisLat[0] && axisLatComp[0] < m_fileStructure.axisLat[0]) {
                        wxLogVerbose(_("Correcting the latitude extent according to the file limits."));
                        double Ywidth = m_fileStructure.axisLat[0] - dataArea->GetAbsoluteYmin();
                        wxASSERT(Ywidth >= 0);
                        int Yptsnb = 1 + Ywidth / dataArea->GetYstep();
                        wxLogDebug(_("yPtsNb = %d."), Yptsnb);
                        asGeoAreaCompositeGrid *newdataArea = asGeoAreaCompositeGrid::GetInstance(
                                dataArea->GetGridTypeString(), dataArea->GetAbsoluteXmin(), dataArea->GetXaxisPtsnb(),
                                dataArea->GetXstep(), dataArea->GetAbsoluteYmin(), Yptsnb, dataArea->GetYstep(),
                                dataArea->GetLevel(), asNONE, asFLAT_ALLOWED);

                        wxDELETE(dataArea);
                        dataArea = newdataArea;
                    }
                }
            }

            Array1DDouble axisLat = dataArea->GetYaxis();
            m_axisLat.resize(axisLat.size());
            for (int i = 0; i < axisLat.size(); i++) {
                // Latitude axis in reverse order
                m_axisLat[i] = (float) axisLat[axisLat.size() - 1 - i];
            }
            m_latPtsnb = dataArea->GetYaxisPtsnb();
            wxASSERT_MSG(m_axisLat.size() == m_latPtsnb,
                         wxString::Format("m_axisLat.size()=%d, m_latPtsnb=%d", (int) m_axisLat.size(), m_latPtsnb));
        }

        compositeData = VVArray2DFloat(dataArea->GetNbComposites());
        m_axesChecked = true;
    }

    return dataArea;
}

void asDataPredictor::AssignGribCode(const int arr[])
{
    m_gribCode.clear();
    for (int i = 0; i < 4; ++i) {
        m_gribCode.push_back(arr[i]);
    }
}

size_t *asDataPredictor::GetIndexesStartNcdf(int i_area) const
{
    if (!m_isEnsemble) {
        if (m_fileStructure.hasLevelDimension) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeStart;
            array[1] = (size_t) m_fileIndexes.level;
            array[2] = (size_t) m_fileIndexes.areas[i_area].latStart;
            array[3] = (size_t) m_fileIndexes.areas[i_area].lonStart;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeStart;
            array[1] = (size_t) m_fileIndexes.areas[i_area].latStart;
            array[2] = (size_t) m_fileIndexes.areas[i_area].lonStart;

            return array;
        }
    } else {
        if (m_fileStructure.hasLevelDimension) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeStart;
            array[1] = (size_t) m_fileIndexes.member;
            array[2] = (size_t) m_fileIndexes.level;
            array[3] = (size_t) m_fileIndexes.areas[i_area].latStart;
            array[4] = (size_t) m_fileIndexes.areas[i_area].lonStart;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeStart;
            array[1] = (size_t) m_fileIndexes.member;
            array[2] = (size_t) m_fileIndexes.areas[i_area].latStart;
            array[3] = (size_t) m_fileIndexes.areas[i_area].lonStart;

            return array;
        }
    }

    return NULL;
}

size_t *asDataPredictor::GetIndexesCountNcdf(int i_area) const
{
    if (!m_isEnsemble) {
        if (m_fileStructure.hasLevelDimension) {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeCount;
            array[1] = 1;
            array[2] = (size_t) m_fileIndexes.areas[i_area].latCount;
            array[3] = (size_t) m_fileIndexes.areas[i_area].lonCount;

            return array;
        } else {
            static size_t array[3] = {0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeCount;
            array[1] = (size_t) m_fileIndexes.areas[i_area].latCount;
            array[2] = (size_t) m_fileIndexes.areas[i_area].lonCount;

            return array;
        }
    } else {
        if (m_fileStructure.hasLevelDimension) {
            static size_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeCount;
            array[1] = 1;
            array[2] = 1;
            array[3] = (size_t) m_fileIndexes.areas[i_area].latCount;
            array[4] = (size_t) m_fileIndexes.areas[i_area].lonCount;

            return array;
        } else {
            static size_t array[4] = {0, 0, 0, 0};
            array[0] = (size_t) m_fileIndexes.timeCount;
            array[1] = 1;
            array[2] = (size_t) m_fileIndexes.areas[i_area].latCount;
            array[3] = (size_t) m_fileIndexes.areas[i_area].lonCount;

            return array;
        }
    }

    return NULL;
}

ptrdiff_t *asDataPredictor::GetIndexesStrideNcdf(int i_area) const
{
    if (!m_isEnsemble) {
        if (m_fileStructure.hasLevelDimension) {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t) m_fileIndexes.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t) m_fileIndexes.latStep;
            array[3] = (ptrdiff_t) m_fileIndexes.lonStep;

            return array;
        } else {
            static ptrdiff_t array[3] = {0, 0, 0};
            array[0] = (ptrdiff_t) m_fileIndexes.timeStep;
            array[1] = (ptrdiff_t) m_fileIndexes.latStep;
            array[2] = (ptrdiff_t) m_fileIndexes.lonStep;

            return array;
        }
    } else {
        if (m_fileStructure.hasLevelDimension) {
            static ptrdiff_t array[5] = {0, 0, 0, 0, 0};
            array[0] = (ptrdiff_t) m_fileIndexes.timeStep;
            array[1] = 1;
            array[2] = 1;
            array[3] = (ptrdiff_t) m_fileIndexes.latStep;
            array[4] = (ptrdiff_t) m_fileIndexes.lonStep;

            return array;
        } else {
            static ptrdiff_t array[4] = {0, 0, 0, 0};
            array[0] = (ptrdiff_t) m_fileIndexes.timeStep;
            array[1] = 1;
            array[2] = (ptrdiff_t) m_fileIndexes.latStep;
            array[3] = (ptrdiff_t) m_fileIndexes.lonStep;

            return array;
        }
    }


    return NULL;
}

int *asDataPredictor::GetIndexesStartGrib(int i_area) const
{
    static int array[2] = {0, 0};
    array[0] = m_fileIndexes.areas[i_area].lonStart;
    array[1] = m_fileIndexes.areas[i_area].latStart;

    return array;
}

int *asDataPredictor::GetIndexesCountGrib(int i_area) const
{
    static int array[2] = {0, 0};
    array[0] = m_fileIndexes.areas[i_area].lonCount;
    array[1] = m_fileIndexes.areas[i_area].latCount;

    return array;
}

bool asDataPredictor::GetDataFromFile(asFileNetcdf &ncFile, VVArray2DFloat &compositeData)
{
    bool isShort = (ncFile.GetVarType(m_fileVariableName) == NC_SHORT);
    bool isFloat = (ncFile.GetVarType(m_fileVariableName) == NC_FLOAT);

    if(!isShort && !isFloat) {
        wxLogError(_("Loading data other than short or float is not implemented yet."));
    }

    // Check if scaling is needed
    bool scalingNeeded = true;
    float dataAddOffset = 0, dataScaleFactor = 1;
    if (ncFile.HasAttribute("add_offset", m_fileVariableName)) {
        dataAddOffset = ncFile.GetAttFloat("add_offset", m_fileVariableName);
    }
    if (ncFile.HasAttribute("scale_factor", m_fileVariableName)) {
        dataScaleFactor = ncFile.GetAttFloat("scale_factor", m_fileVariableName);
    }
    if (dataAddOffset == 0 && dataScaleFactor == 1)
        scalingNeeded = false;

    VVectorFloat vectData;

    for (int i_area = 0; i_area < compositeData.size(); i_area++) {

        // Create the arrays to receive the data
        VectorFloat dataF;
        VectorShort dataS;

        // Resize the arrays to store the new data
        int totLength = m_fileIndexes.timeArrayCount * m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
        wxASSERT(totLength > 0);
        dataF.resize(totLength);
        if (isShort) {
            dataS.resize(totLength);
        }

        // Fill empty beginning with NaNs
        int indexBegining = 0;
        if (m_fileIndexes.cutStart > 0) {
            int latlonlength = m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
            for (int i_empty = 0; i_empty < m_fileIndexes.cutStart; i_empty++) {
                for (int i_emptylatlon = 0; i_emptylatlon < latlonlength; i_emptylatlon++) {
                    dataF[indexBegining] = NaNFloat;
                    indexBegining++;
                }
            }
        }

        // Fill empty end with NaNs
        int indexEnd = m_fileIndexes.timeCount * m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount - 1;
        if (m_fileIndexes.cutEnd > 0) {
            int latlonlength = m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
            for (int i_empty = 0; i_empty < m_fileIndexes.cutEnd; i_empty++) {
                for (int i_emptylatlon = 0; i_emptylatlon < latlonlength; i_emptylatlon++) {
                    indexEnd++;
                    dataF[indexEnd] = NaNFloat;
                }
            }
        }

        // In the netCDF Common Data Language, variables are printed with the outermost dimension first and the innermost dimension last.
        if (isFloat) {
            ncFile.GetVarSample(m_fileVariableName, GetIndexesStartNcdf(i_area), GetIndexesCountNcdf(i_area),
                                GetIndexesStrideNcdf(i_area), &dataF[indexBegining]);
        } else if (isShort) {
            ncFile.GetVarSample(m_fileVariableName, GetIndexesStartNcdf(i_area), GetIndexesCountNcdf(i_area),
                                GetIndexesStrideNcdf(i_area), &dataS[indexBegining]);
            for (int i = 0; i < dataS.size(); i++) {
                dataF[i] = (float) dataS[i];
            }
        }

        // Keep data for later treatment
        vectData.push_back(dataF);
    }

    // Allocate space into compositeData if not already done
    if (compositeData[0].capacity() == 0) {
        int totSize = 0;
        for (int i_area = 0; i_area < compositeData.size(); i_area++) {
            totSize += m_time.size() * m_fileIndexes.areas[i_area].latCount
                       * (m_fileIndexes.areas[i_area].lonCount + 1); // +1 in case of a border
        }
        compositeData.reserve(totSize);
    }

    // Transfer data
    for (int i_area = 0; i_area < compositeData.size(); i_area++) {
        // Extract data
        VectorFloat data = vectData[i_area];

        // Loop to extract the data from the array
        int ind = 0;
        for (int i_time = 0; i_time < m_fileIndexes.timeArrayCount; i_time++) {
            Array2DFloat latlonData = Array2DFloat(m_fileIndexes.areas[i_area].latCount,
                                                   m_fileIndexes.areas[i_area].lonCount);

            for (int i_lat = 0; i_lat < m_fileIndexes.areas[i_area].latCount; i_lat++) {
                for (int i_lon = 0; i_lon < m_fileIndexes.areas[i_area].lonCount; i_lon++) {
                    ind = i_lon
                          + i_lat * m_fileIndexes.areas[i_area].lonCount
                          + i_time * m_fileIndexes.areas[i_area].lonCount * m_fileIndexes.areas[i_area].latCount;
                    if (m_fileStructure.axisLat.size()>0 && m_fileStructure.axisLat[1] > m_fileStructure.axisLat[0]) {
                        int latRevIndex = m_fileIndexes.areas[i_area].latCount - 1 - i_lat;
                        ind = i_lon
                              + latRevIndex * m_fileIndexes.areas[i_area].lonCount
                              + i_time * m_fileIndexes.areas[i_area].lonCount * m_fileIndexes.areas[i_area].latCount;
                    }

                    if (scalingNeeded) {
                        latlonData(i_lat, i_lon) = data[ind] * dataScaleFactor + dataAddOffset;
                    } else {
                        latlonData(i_lat, i_lon) = data[ind];
                    }

                    // Check if not NaN
                    bool notNan = true;
                    for (size_t i_nan = 0; i_nan < m_nanValues.size(); i_nan++) {
                        if (data[ind] == m_nanValues[i_nan] || latlonData(i_lat, i_lon) == m_nanValues[i_nan]) {
                            notNan = false;
                        }
                    }
                    if (!notNan) {
                        latlonData(i_lat, i_lon) = NaNFloat;
                    }
                }
            }
            compositeData[i_area].push_back(latlonData);
        }
        data.clear();
    }

    return true;
}

bool asDataPredictor::GetDataFromFile(asFileGrib2 &gbFile, VVArray2DFloat &compositeData)
{
    VVectorFloat vectData;

    for (int i_area = 0; i_area < compositeData.size(); i_area++) {

        // Create the arrays to receive the data
        VectorFloat dataF;

        // Resize the arrays to store the new data
        int totLength = m_fileIndexes.timeArrayCount * m_fileIndexes.areas[i_area].latCount * m_fileIndexes.areas[i_area].lonCount;
        wxASSERT(totLength > 0);
        dataF.resize(totLength);

        // Extract data
        gbFile.GetVarArray(GetIndexesStartGrib(i_area), GetIndexesCountGrib(i_area), &dataF[0]);

        // Keep data for later treatment
        vectData.push_back(dataF);
    }

    // Allocate space into compositeData if not already done
    if (compositeData[0].capacity() == 0) {
        int totSize = 0;
        for (int i_area = 0; i_area < compositeData.size(); i_area++) {
            totSize += m_time.size() * m_fileIndexes.areas[i_area].latCount * (m_fileIndexes.areas[i_area].lonCount + 1); // +1 in case of a border
        }
        compositeData.reserve(totSize);
    }

    // Transfer data
    for (int i_area = 0; i_area < compositeData.size(); i_area++) {
        // Extract data
        VectorFloat data = vectData[i_area];

        // Loop to extract the data from the array
        int ind = 0;
        Array2DFloat latlonData = Array2DFloat(m_fileIndexes.areas[i_area].latCount,
                                               m_fileIndexes.areas[i_area].lonCount);

        for (int i_lat = 0; i_lat < m_fileIndexes.areas[i_area].latCount; i_lat++) {
            for (int i_lon = 0; i_lon < m_fileIndexes.areas[i_area].lonCount; i_lon++) {
                int latRevIndex = m_fileIndexes.areas[i_area].latCount - 1 - i_lat; // Index reversed in Grib files
                ind = i_lon + latRevIndex * m_fileIndexes.areas[i_area].lonCount;
                latlonData(i_lat, i_lon) = data[ind];

                // Check if not NaN
                bool notNan = true;
                for (size_t i_nan = 0; i_nan < m_nanValues.size(); i_nan++) {
                    if (data[ind] == m_nanValues[i_nan] || latlonData(i_lat, i_lon) == m_nanValues[i_nan]) {
                        notNan = false;
                    }
                }
                if (!notNan) {
                    latlonData(i_lat, i_lon) = NaNFloat;
                }
            }
        }
        compositeData[i_area].push_back(latlonData);

        data.clear();
    }

    return true;
}

bool asDataPredictor::Inline()
{
    //Already inlined
    if (m_lonPtsnb == 1 || m_latPtsnb == 1) {
        return true;
    }

    wxASSERT(m_data.size() > 0);

    int timeSize = m_data.size();
    int cols = m_data[0].cols();
    int rows = m_data[0].rows();

    Array2DFloat inlineData = Array2DFloat::Zero(1, cols * rows);

    VArray2DFloat newData;
    newData.reserve(m_time.size() * m_lonPtsnb * m_latPtsnb);
    newData.resize(timeSize);

    for (int i_time = 0; i_time < timeSize; i_time++) {
        for (int i_row = 0; i_row < rows; i_row++) {
            inlineData.block(0, i_row * cols, 1, cols) = m_data[i_time].row(i_row);
        }
        newData[i_time] = inlineData;
    }

    m_data = newData;

    m_latPtsnb = m_data[0].rows();
    m_lonPtsnb = m_data[0].cols();
    Array1DFloat emptyAxis(1);
    emptyAxis[0] = NaNFloat;
    m_axisLat = emptyAxis;
    m_axisLon = emptyAxis;

    return true;
}

bool asDataPredictor::MergeComposites(VVArray2DFloat &compositeData, asGeoAreaCompositeGrid *area)
{
    if (area) {
        // Get a container with the final size
        int sizeTime = compositeData[0].size();
        m_data = VArray2DFloat(sizeTime, Array2DFloat(m_latPtsnb, m_lonPtsnb));

        Array2DFloat blockUL, blockLL, blockUR, blockLR;
        int isblockUL = asNONE, isblockLL = asNONE, isblockUR = asNONE, isblockLR = asNONE;

        // Resize containers for composite areas
        for (int i_area = 0; i_area < area->GetNbComposites(); i_area++) {
            if ((area->GetComposite(i_area).GetXmax() == area->GetXmax()) &
                (area->GetComposite(i_area).GetYmin() == area->GetYmin())) {
                blockUL.resize(compositeData[i_area][0].rows(), compositeData[i_area][0].cols());
                isblockUL = i_area;
            } else if ((area->GetComposite(i_area).GetXmin() == area->GetXmin()) &
                       (area->GetComposite(i_area).GetYmin() == area->GetYmin())) {
                blockUR.resize(compositeData[i_area][0].rows(), compositeData[i_area][0].cols());
                isblockUR = i_area;
            } else if ((area->GetComposite(i_area).GetXmax() == area->GetXmax()) &
                       (area->GetComposite(i_area).GetYmax() == area->GetYmax())) {
                blockLL.resize(compositeData[i_area][0].rows(), compositeData[i_area][0].cols());
                isblockLL = i_area;
            } else if ((area->GetComposite(i_area).GetXmin() == area->GetXmin()) &
                       (area->GetComposite(i_area).GetYmax() == area->GetYmax())) {
                blockLR.resize(compositeData[i_area][0].rows(), compositeData[i_area][0].cols());
                isblockLR = i_area;
            } else {
                wxLogError(_("The data composite was not identified."));
                return false;
            }
        }

        // Merge the composite data together
        for (int i_time = 0; i_time < sizeTime; i_time++) {
            // Append the composite areas
            for (int i_area = 0; i_area < area->GetNbComposites(); i_area++) {
                if (i_area == isblockUL) {
                    blockUL = compositeData[i_area][i_time];
                    m_data[i_time].topLeftCorner(blockUL.rows(), blockUL.cols()) = blockUL;
                } else if (i_area == isblockUR) {
                    blockUR = compositeData[i_area][i_time];
                    m_data[i_time].block(0, m_lonPtsnb - blockUR.cols(), blockUR.rows(), blockUR.cols()) = blockUR;
                } else if (i_area == isblockLL) {
                    blockLL = compositeData[i_area][i_time];
                    // TODO (phorton#1#): Implement me!
                    wxLogError(_("Not yet implemented."));
                    return false;
                } else if (i_area == isblockLR) {
                    blockLR = compositeData[i_area][i_time];
                    // TODO (phorton#1#): Implement me!
                    wxLogError(_("Not yet implemented."));
                    return false;
                } else {
                    wxLogError(_("The data composite cannot be build."));
                    return false;
                }
            }
        }
    } else {
        m_data = compositeData[0];
    }

    return true;
}

bool asDataPredictor::InterpolateOnGrid(asGeoAreaCompositeGrid *dataArea, asGeoAreaCompositeGrid *desiredArea)
{
    wxASSERT(dataArea);
    wxASSERT(dataArea->GetNbComposites() > 0);
    wxASSERT(desiredArea);
    wxASSERT(desiredArea->GetNbComposites() > 0);
    bool changeXstart = false, changeXsteps = false, changeYstart = false, changeYsteps = false;

    // Check beginning on longitudes
    if (dataArea->GetAbsoluteXmin() != desiredArea->GetAbsoluteXmin()) {
        changeXstart = true;
    }

    // Check beginning on latitudes
    if (dataArea->GetAbsoluteYmin() != desiredArea->GetAbsoluteYmin()) {
        changeYstart = true;
    }

    // Check the cells size on longitudes
    if (!dataArea->GridsOverlay(desiredArea)) {
        changeXsteps = true;
        changeYsteps = true;
    }

    // Proceed to the interpolation
    if (changeXstart || changeYstart || changeXsteps || changeYsteps) {
        // Containers for results
        int finalLengthLon = desiredArea->GetXaxisPtsnb();
        int finalLengthLat = desiredArea->GetYaxisPtsnb();
        VArray2DFloat latlonTimeData(m_data.size(), Array2DFloat(finalLengthLat, finalLengthLon));

        // Creation of the axes
        Array1DFloat axisDataLon;
        if (dataArea->GetXaxisPtsnb() > 1) {
            axisDataLon = Array1DFloat::LinSpaced(Eigen::Sequential, dataArea->GetXaxisPtsnb(),
                                                  dataArea->GetAbsoluteXmin(), dataArea->GetAbsoluteXmax());
        } else {
            axisDataLon.resize(1);
            axisDataLon << dataArea->GetAbsoluteXmin();
        }

        Array1DFloat axisDataLat;
        if (dataArea->GetYaxisPtsnb() > 1) {
            axisDataLat = Array1DFloat::LinSpaced(Eigen::Sequential, dataArea->GetYaxisPtsnb(),
                                                  dataArea->GetAbsoluteYmax(),
                                                  dataArea->GetAbsoluteYmin()); // From top to bottom
        } else {
            axisDataLat.resize(1);
            axisDataLat << dataArea->GetAbsoluteYmax();
        }

        Array1DFloat axisFinalLon;
        if (desiredArea->GetXaxisPtsnb() > 1) {
            axisFinalLon = Array1DFloat::LinSpaced(Eigen::Sequential, desiredArea->GetXaxisPtsnb(),
                                                   desiredArea->GetAbsoluteXmin(), desiredArea->GetAbsoluteXmax());
        } else {
            axisFinalLon.resize(1);
            axisFinalLon << desiredArea->GetAbsoluteXmin();
        }

        Array1DFloat axisFinalLat;
        if (desiredArea->GetYaxisPtsnb() > 1) {
            axisFinalLat = Array1DFloat::LinSpaced(Eigen::Sequential, desiredArea->GetYaxisPtsnb(),
                                                   desiredArea->GetAbsoluteYmax(),
                                                   desiredArea->GetAbsoluteYmin()); // From top to bottom
        } else {
            axisFinalLat.resize(1);
            axisFinalLat << desiredArea->GetAbsoluteYmax();
        }

        // Indices
        int indexXfloor, indexXceil;
        int indexYfloor, indexYceil;
        int axisDataLonEnd = axisDataLon.size() - 1;
        int axisDataLatEnd = axisDataLat.size() - 1;

        // Pointer to last used element
        int indexLastLon = 0, indexLastLat = 0;

        // Variables
        double dX, dY;
        float valLLcorner, valULcorner, valLRcorner, valURcorner;

        // The interpolation loop
        for (unsigned int i_time = 0; i_time < m_data.size(); i_time++) {
            // Loop to extract the data from the array
            for (int i_lat = 0; i_lat < finalLengthLat; i_lat++) {
                // Try the 2 next latitudes (from the top)
                if (axisDataLat.size() > indexLastLat + 1 && axisDataLat[indexLastLat + 1] == axisFinalLat[i_lat]) {
                    indexYfloor = indexLastLat + 1;
                    indexYceil = indexLastLat + 1;
                } else if (axisDataLat.size() > indexLastLat + 2 &&
                           axisDataLat[indexLastLat + 2] == axisFinalLat[i_lat]) {
                    indexYfloor = indexLastLat + 2;
                    indexYceil = indexLastLat + 2;
                } else {
                    // Search for floor and ceil
                    indexYfloor = indexLastLat + asTools::SortedArraySearchFloor(&axisDataLat[indexLastLat],
                                                                                 &axisDataLat[axisDataLatEnd],
                                                                                 axisFinalLat[i_lat]);
                    indexYceil = indexLastLat + asTools::SortedArraySearchCeil(&axisDataLat[indexLastLat],
                                                                               &axisDataLat[axisDataLatEnd],
                                                                               axisFinalLat[i_lat]);
                }

                if (indexYfloor == asOUT_OF_RANGE || indexYfloor == asNOT_FOUND || indexYceil == asOUT_OF_RANGE ||
                    indexYceil == asNOT_FOUND) {
                    wxLogError(_("The desired point is not available in the data for interpolation. Latitude %f was not found inbetween %f (index %d) to %f (index %d) (size = %d)."),
                               axisFinalLat[i_lat], axisDataLat[indexLastLat], indexLastLat,
                               axisDataLat[axisDataLatEnd], axisDataLatEnd, (int) axisDataLat.size());
                    return false;
                }
                wxASSERT_MSG(indexYfloor >= 0,
                             wxString::Format("%f in %f to %f", axisFinalLat[i_lat], axisDataLat[indexLastLat],
                                              axisDataLat[axisDataLatEnd]));
                wxASSERT(indexYceil >= 0);

                // Save last index
                indexLastLat = indexYfloor;

                for (int i_lon = 0; i_lon < finalLengthLon; i_lon++) {
                    // Try the 2 next longitudes
                    if (axisDataLon.size() > indexLastLon + 1 && axisDataLon[indexLastLon + 1] == axisFinalLon[i_lon]) {
                        indexXfloor = indexLastLon + 1;
                        indexXceil = indexLastLon + 1;
                    } else if (axisDataLon.size() > indexLastLon + 2 &&
                               axisDataLon[indexLastLon + 2] == axisFinalLon[i_lon]) {
                        indexXfloor = indexLastLon + 2;
                        indexXceil = indexLastLon + 2;
                    } else {
                        // Search for floor and ceil
                        indexXfloor = indexLastLon + asTools::SortedArraySearchFloor(&axisDataLon[indexLastLon],
                                                                                     &axisDataLon[axisDataLonEnd],
                                                                                     axisFinalLon[i_lon]);
                        indexXceil = indexLastLon + asTools::SortedArraySearchCeil(&axisDataLon[indexLastLon],
                                                                                   &axisDataLon[axisDataLonEnd],
                                                                                   axisFinalLon[i_lon]);
                    }

                    if (indexXfloor == asOUT_OF_RANGE || indexXfloor == asNOT_FOUND || indexXceil == asOUT_OF_RANGE ||
                        indexXceil == asNOT_FOUND) {
                        wxLogError(_("The desired point is not available in the data for interpolation. Longitude %f was not found inbetween %f to %f."),
                                   axisFinalLon[i_lon], axisDataLon[indexLastLon], axisDataLon[axisDataLonEnd]);
                        return false;
                    }

                    wxASSERT(indexXfloor >= 0);
                    wxASSERT(indexXceil >= 0);

                    // Save last index
                    indexLastLon = indexXfloor;

                    // Proceed to the interpolation
                    if (indexXceil == indexXfloor) {
                        dX = 0;
                    } else {
                        dX = (axisFinalLon[i_lon] - axisDataLon[indexXfloor]) /
                             (axisDataLon[indexXceil] - axisDataLon[indexXfloor]);
                    }
                    if (indexYceil == indexYfloor) {
                        dY = 0;
                    } else {
                        dY = (axisFinalLat[i_lat] - axisDataLat[indexYfloor]) /
                             (axisDataLat[indexYceil] - axisDataLat[indexYfloor]);
                    }


                    if (dX == 0 && dY == 0) {
                        latlonTimeData[i_time](i_lat, i_lon) = m_data[i_time](indexYfloor, indexXfloor);
                    } else if (dX == 0) {
                        valLLcorner = m_data[i_time](indexYfloor, indexXfloor);
                        valULcorner = m_data[i_time](indexYceil, indexXfloor);

                        latlonTimeData[i_time](i_lat, i_lon) =
                                (1 - dX) * (1 - dY) * valLLcorner + (1 - dX) * (dY) * valULcorner;
                    } else if (dY == 0) {
                        valLLcorner = m_data[i_time](indexYfloor, indexXfloor);
                        valLRcorner = m_data[i_time](indexYfloor, indexXceil);

                        latlonTimeData[i_time](i_lat, i_lon) =
                                (1 - dX) * (1 - dY) * valLLcorner + (dX) * (1 - dY) * valLRcorner;
                    } else {
                        valLLcorner = m_data[i_time](indexYfloor, indexXfloor);
                        valULcorner = m_data[i_time](indexYceil, indexXfloor);
                        valLRcorner = m_data[i_time](indexYfloor, indexXceil);
                        valURcorner = m_data[i_time](indexYceil, indexXceil);

                        latlonTimeData[i_time](i_lat, i_lon) =
                                (1 - dX) * (1 - dY) * valLLcorner + (1 - dX) * (dY) * valULcorner +
                                (dX) * (1 - dY) * valLRcorner + (dX) * (dY) * valURcorner;
                    }
                }

                indexLastLon = 0;
            }

            indexLastLat = 0;
        }

        m_data = latlonTimeData;
        m_latPtsnb = finalLengthLat;
        m_lonPtsnb = finalLengthLon;
    }

    return true;
}

float asDataPredictor::GetMinValue() const
{
    float minValue = m_data[0](0, 0);
    float tmpValue;

    for (int i = 0; i < m_data.size(); ++i) {
        tmpValue = m_data[i].minCoeff();
        if (tmpValue < minValue) {
            minValue = tmpValue;
        }
    }

    return minValue;
}

float asDataPredictor::GetMaxValue() const
{
    float maxValue = m_data[0](0, 0);
    float tmpValue;

    for (int i = 0; i < m_data.size(); ++i) {
        tmpValue = m_data[i].maxCoeff();
        if (tmpValue > maxValue) {
            maxValue = tmpValue;
        }
    }

    return maxValue;
}

void asDataPredictor::CheckLevelTypeIsDefined()
{
    if(m_product.IsEmpty()) {
        asThrowException(_("The type of product must be defined for this dataset (prefix to the variable name. Ex: press/hgt)."));
    }
}


