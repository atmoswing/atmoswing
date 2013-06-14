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
 
#include "asGlobEnums.h"


FileFormat asGlobEnums::StringToFileFormatEnum(const wxString &FormatStr)
{
    if (FormatStr.CmpNoCase("dat")==0)
    {
        return dat;
    }
    else if (FormatStr.CmpNoCase("asc")==0)
    {
        return asc;
    }
    else if (FormatStr.CmpNoCase("txt")==0)
    {
        return txt;
    }
    else if (FormatStr.CmpNoCase("netcdf")==0)
    {
        return netcdf;
    }
    else if (FormatStr.CmpNoCase("grib")==0)
    {
        return grib;
    }
    else if (FormatStr.CmpNoCase("grib2")==0)
    {
        return grib2;
    }
    else if (FormatStr.CmpNoCase("xml")==0)
    {
        return xml;
    }
    else if (FormatStr.CmpNoCase("kml")==0)
    {
        return kml;
    }
    else
    {
        asLogError(wxString::Format(_("The Format enumeration (%s) entry doesn't exists"), FormatStr));
    }
    return NoFileFormat;
}

wxString asGlobEnums::FileFormatEnumToExtension(FileFormat format)
{
    switch (format)
    {
        case dat:
            return "dat";
        case asc:
            return "asc";
        case txt:
            return "txt";
        case netcdf:
            return "nc";
        case grib:
            return "grib";
        case grib2:
            return "grib2";
        case xml:
            return "xml";
        case kml:
            return "kml";
        default:
            asLogError(_("File format unknown."));
            return wxEmptyString;
    }
}

FileTimeLength asGlobEnums::StringToFileLengthEnum(const wxString &LengthStr)
{
    if ( (LengthStr.CmpNoCase("T")==0) | (LengthStr.CmpNoCase("Total")==0) )
    {
        return Total;
    }
    else if ( (LengthStr.CmpNoCase("Y")==0) | (LengthStr.CmpNoCase("Year")==0) )
    {
        return Year;
    }
    else if ( (LengthStr.CmpNoCase("H")==0) | (LengthStr.CmpNoCase("Hour")==0) )
    {
        return Hour;
    }
    else if ( (LengthStr.CmpNoCase("I")==0) | (LengthStr.CmpNoCase("Instantaneous")==0) )
    {
        return Instantaneous;
    }
    else if (LengthStr.IsEmpty())
    {
        return NoFileLength;
    }
    else
    {
        asLogError(wxString::Format(_("The Length enumeration (%s) entry doesn't exists"), LengthStr));
    }
    return NoFileLength;
}

DataParameter asGlobEnums::StringToDataParameterEnum(const wxString &ParameterStr)
{
    if (ParameterStr.CmpNoCase("Precipitation")==0)
    {
        return Precipitation;
    }
    else if (ParameterStr.CmpNoCase("AirTemperature")==0)
    {
        return AirTemperature;
    }
    else if (ParameterStr.CmpNoCase("GeopotentialHeight")==0)
    {
        return GeopotentialHeight;
    }
    else if (ParameterStr.CmpNoCase("PrecipitableWater")==0)
    {
        return PrecipitableWater;
    }
    else if (ParameterStr.CmpNoCase("RelativeHumidity")==0)
    {
        return RelativeHumidity;
    }
    else if (ParameterStr.CmpNoCase("SpecificHumidity")==0)
    {
        return SpecificHumidity;
    }
    else if (ParameterStr.CmpNoCase("Omega")==0)
    {
        return Omega;
    }
    else if (ParameterStr.CmpNoCase("Uwind")==0)
    {
        return Uwind;
    }
    else if (ParameterStr.CmpNoCase("Vwind")==0)
    {
        return Vwind;
    }
    else if (ParameterStr.CmpNoCase("SurfaceLiftedIndex")==0)
    {
        return SurfaceLiftedIndex;
    }
    else if (ParameterStr.CmpNoCase("PotentialTemperature")==0)
    {
        return PotentialTemperature;
    }
    else if (ParameterStr.CmpNoCase("Pressure")==0)
    {
        return Pressure;
    }
    else if (ParameterStr.CmpNoCase("PotentialEvaporation")==0)
    {
        return PotentialEvaporation;
    }
    else if (ParameterStr.CmpNoCase("SurfaceTemperature")==0)
    {
        return SurfaceTemperature;
    }
    else if (ParameterStr.CmpNoCase("ConvectivePrecipitation")==0)
    {
        return ConvectivePrecipitation;
    }
    else if (ParameterStr.CmpNoCase("LongwaveRadiation")==0)
    {
        return LongwaveRadiation;
    }
    else if (ParameterStr.CmpNoCase("ShortwaveRadiation")==0)
    {
        return ShortwaveRadiation;
    }
    else if (ParameterStr.CmpNoCase("SolarRadiation")==0)
    {
        return SolarRadiation;
    }
    else if (ParameterStr.CmpNoCase("GroundHeatFlux")==0)
    {
        return GroundHeatFlux;
    }
    else if (ParameterStr.CmpNoCase("LatentHeatFlux")==0)
    {
        return LatentHeatFlux;
    }
    else if (ParameterStr.CmpNoCase("NearIRFlux")==0)
    {
        return NearIRFlux;
    }
    else if (ParameterStr.CmpNoCase("SensibleHeatFlux")==0)
    {
        return SensibleHeatFlux;
    }
    else if (ParameterStr.CmpNoCase("Lightnings")==0)
    {
        return Lightnings;
    }
    else if (ParameterStr.CmpNoCase("SeaSurfaceTemperature")==0)
    {
        return SeaSurfaceTemperature;
    }
    else if (ParameterStr.CmpNoCase("SST")==0)
    {
        return SeaSurfaceTemperature;
    }
    else if (ParameterStr.CmpNoCase("SeaSurfaceTemperatureAnomaly")==0)
    {
        return SeaSurfaceTemperatureAnomaly;
    }
    else if (ParameterStr.CmpNoCase("SSTAnomaly")==0)
    {
        return SeaSurfaceTemperatureAnomaly;
    }
    else
    {
        asLogError(wxString::Format(_("The Parameter enumeration (%s) entry doesn't exists"), ParameterStr));
    }
    return NoDataParameter;
}

DataUnit asGlobEnums::StringToDataUnitEnum(const wxString &UnitStr)
{

    if (UnitStr.CmpNoCase("nb")==0)
    {
        return nb;
    }
    else if (UnitStr.CmpNoCase("number")==0)
    {
        return nb;
    }
    else if (UnitStr.CmpNoCase("mm")==0)
    {
        return mm;
    }
    else if (UnitStr.CmpNoCase("m")==0)
    {
        return m;
    }
    else if (UnitStr.CmpNoCase("km")==0)
    {
        return km;
    }
    else if (UnitStr.CmpNoCase("percent")==0)
    {
        return percent;
    }
    else if (UnitStr.CmpNoCase("%")==0)
    {
        return percent;
    }
    else if (UnitStr.CmpNoCase("degC")==0)
    {
        return degC;
    }
    else if (UnitStr.CmpNoCase("degK")==0)
    {
        return degK;
    }
    else if (UnitStr.CmpNoCase("Pascals")==0)
    {
        return Pascals;
    }
    else if (UnitStr.CmpNoCase("PascalsPerSec")==0)
    {
        return PascalsPerSec;
    }
    else if (UnitStr.CmpNoCase("Pascals/s")==0)
    {
        return PascalsPerSec;
    }
    else if (UnitStr.CmpNoCase("kgPerKg")==0)
    {
        return kgPerKg;
    }
    else if (UnitStr.CmpNoCase("kg/kg")==0)
    {
        return kgPerKg;
    }
    else if (UnitStr.CmpNoCase("mPerSec")==0)
    {
        return mPerSec;
    }
    else if (UnitStr.CmpNoCase("m/s")==0)
    {
        return mPerSec;
    }
    else if (UnitStr.CmpNoCase("WPerm2")==0)
    {
        return WPerm2;
    }
    else if (UnitStr.CmpNoCase("W/m2")==0)
    {
        return WPerm2;
    }
    else if (UnitStr.CmpNoCase("kgPerm2Pers")==0)
    {
        return kgPerm2Pers;
    }
    else if (UnitStr.CmpNoCase("kg/m2/s")==0)
    {
        return kgPerm2Pers;
    }
    else
    {
        asLogError(wxString::Format(_("The Unit enumeration (%s) entry doesn't exists"), UnitStr));
    }
    return NoDataUnit;
}

CoordSys asGlobEnums::StringToCoordSysEnum(const wxString &CoordSysStr)
{
    if (CoordSysStr.CmpNoCase("WGS84")==0)
    {
        return WGS84;
    }
    else if (CoordSysStr.CmpNoCase("CH1903")==0)
    {
        return CH1903;
    }
    else if (CoordSysStr.CmpNoCase("CH1903p")==0)
    {
        return CH1903p;
    }
    else
    {
        asLogError(_("The coordinate system in unknown"));
    }
    return NoCoordSys;
}

PredictandDB asGlobEnums::StringToPredictandDBEnum(const wxString &PredictandDBStr)
{
    if (PredictandDBStr.CmpNoCase("StationsDailyPrecipitation")==0)
    {
        return StationsDailyPrecipitation;
    }
    else if (PredictandDBStr.CmpNoCase("Stations6HourlyPrecipitation")==0)
    {
        return Stations6HourlyPrecipitation;
    }
    else if (PredictandDBStr.CmpNoCase("Stations6HourlyOfDailyPrecipitation")==0)
    {
        return Stations6HourlyOfDailyPrecipitation;
    }
    else if (PredictandDBStr.CmpNoCase("RegionalDailyPrecipitation")==0)
    {
        return RegionalDailyPrecipitation;
    }
    else if (PredictandDBStr.CmpNoCase("ResearchDailyPrecipitation")==0)
    {
        return ResearchDailyPrecipitation;
    }
    else if (PredictandDBStr.CmpNoCase("StationsDailyLightnings")==0)
    {
        return StationsDailyLightnings;
    }
    else if (PredictandDBStr.CmpNoCase("NoPredictandDB")==0)
    {
        return NoPredictandDB;
    }
    else
    {
        asLogError(wxString::Format(_("The given predictand DB type (%s) in unknown"), PredictandDBStr.c_str()));
    }
    return NoPredictandDB;
}

wxString asGlobEnums::PredictandDBEnumToString(PredictandDB predictandDB)
{
    switch (predictandDB)
    {
        case(StationsDailyPrecipitation):
            return "StationsDailyPrecipitation";
            break;
        case(Stations6HourlyPrecipitation):
            return "Stations6HourlyPrecipitation";
            break;
        case(Stations6HourlyOfDailyPrecipitation):
            return "Stations6HourlyOfDailyPrecipitation";
            break;
        case(RegionalDailyPrecipitation):
            return "RegionalDailyPrecipitation";
            break;
        case(ResearchDailyPrecipitation):
            return "ResearchDailyPrecipitation";
            break;
        case(StationsDailyLightnings):
            return "StationsDailyLightnings";
            break;
        case(NoPredictandDB):
            return "NoPredictandDB";
            break;

        default:
        asLogError(_("The given predictand DB type in unknown"));
    }
    return wxEmptyString;
}
