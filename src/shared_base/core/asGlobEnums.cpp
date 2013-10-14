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
        asLogError(wxString::Format(_("The Format enumeration (%s) entry doesn't exists"), FormatStr.c_str()));
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
        asLogError(wxString::Format(_("The Length enumeration (%s) entry doesn't exists"), LengthStr.c_str()));
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
    else if (ParameterStr.CmpNoCase("Wind")==0)
    {
        return Wind;
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
        asLogError(wxString::Format(_("The Parameter enumeration (%s) entry doesn't exists"), ParameterStr.c_str()));
    }
    return NoDataParameter;
}

wxString asGlobEnums::DataParameterEnumToString(DataParameter dataParameter)
{
	switch (dataParameter)
	{
		case(Precipitation):
			return "Precipitation";
		case(AirTemperature):
			return "AirTemperature";
		case(GeopotentialHeight):
			return "GeopotentialHeight";
		case(PrecipitableWater):
			return "PrecipitableWater";
		case(RelativeHumidity):
			return "RelativeHumidity";
		case(SpecificHumidity):
			return "SpecificHumidity";
		case(Omega):
			return "Omega";
		case(Wind):
			return "Wind";
		case(Uwind):
			return "Uwind";
		case(Vwind):
			return "Vwind";
		case(SurfaceLiftedIndex):
			return "SurfaceLiftedIndex";
		case(PotentialTemperature):
			return "PotentialTemperature";
		case(Pressure):
			return "Pressure";
		case(PotentialEvaporation):
			return "PotentialEvaporation";
		case(SurfaceTemperature):
			return "SurfaceTemperature";
		case(ConvectivePrecipitation):
			return "ConvectivePrecipitation";
		case(LongwaveRadiation):
			return "LongwaveRadiation";
		case(ShortwaveRadiation):
			return "ShortwaveRadiation";
		case(SolarRadiation):
			return "SolarRadiation";
		case(GroundHeatFlux):
			return "GroundHeatFlux";
		case(LatentHeatFlux):
			return "LatentHeatFlux";
		case(NearIRFlux):
			return "NearIRFlux";
		case(SensibleHeatFlux):
			return "SensibleHeatFlux";
		case(Lightnings):
			return "Lightnings";
		case(SeaSurfaceTemperature):
			return "SeaSurfaceTemperature";
		case(SeaSurfaceTemperatureAnomaly):
			return "SeaSurfaceTemperatureAnomaly";
		default:
        asLogError(_("The given data parameter type in unknown."));
	}
	return wxEmptyString;
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
        asLogError(wxString::Format(_("The Unit enumeration (%s) entry doesn't exists"), UnitStr.c_str()));
    }
    return NoDataUnit;
}

DataTemporalResolution asGlobEnums::StringToDataTemporalResolutionEnum(const wxString &TemporalResolution)
{

    if (TemporalResolution.CmpNoCase("Daily")==0)
    {
        return Daily;
    }
    else if (TemporalResolution.CmpNoCase("1 day")==0)
    {
        return Daily;
    }
    else if (TemporalResolution.CmpNoCase("SixHourly")==0)
    {
        return SixHourly;
    }
    else if (TemporalResolution.CmpNoCase("6 hours")==0)
    {
        return SixHourly;
    }
    else if (TemporalResolution.CmpNoCase("Hourly")==0)
    {
        return Hourly;
    }
    else if (TemporalResolution.CmpNoCase("1 hour")==0)
    {
        return Hourly;
    }
    else if (TemporalResolution.CmpNoCase("SixHourlyMovingDailyTemporalWindow")==0)
    {
        return SixHourlyMovingDailyTemporalWindow;
    }
    else if (TemporalResolution.CmpNoCase("MovingTemporalWindow")==0)
    {
        return SixHourlyMovingDailyTemporalWindow;
    }
    else if (TemporalResolution.CmpNoCase("TwoDays")==0)
    {
        return TwoDays;
    }
    else if (TemporalResolution.CmpNoCase("2 days")==0)
    {
        return TwoDays;
    }
    else if (TemporalResolution.CmpNoCase("ThreeDays")==0)
    {
        return ThreeDays;
    }
    else if (TemporalResolution.CmpNoCase("3 days")==0)
    {
        return ThreeDays;
    }
    else if (TemporalResolution.CmpNoCase("Weekly")==0)
    {
        return Weekly;
    }
    else if (TemporalResolution.CmpNoCase("1 week")==0)
    {
        return Weekly;
    }
    else
    {
        asLogError(wxString::Format(_("The TemporalResolution enumeration (%s) entry doesn't exists"), TemporalResolution.c_str()));
    }
    return NoDataTemporalResolution;
}

wxString asGlobEnums::DataTemporalResolutionEnumToString(DataTemporalResolution dataTemporalResolution)
{
	switch (dataTemporalResolution)
	{
		case(Daily):
			return "Daily";
		case(SixHourly):
			return "SixHourly";
		case(Hourly):
			return "Hourly";
		case(SixHourlyMovingDailyTemporalWindow):
			return "SixHourlyMovingDailyTemporalWindow";
		case(TwoDays):
			return "TwoDays";
		case(ThreeDays):
			return "ThreeDays";
		case(Weekly):
			return "Weekly";
		default:
        asLogError(_("The given data temporal resolution type in unknown."));
	}
	return wxEmptyString;
}

DataSpatialAggregation asGlobEnums::StringToDataSpatialAggregationEnum(const wxString &SpatialAggregation)
{

    if (SpatialAggregation.CmpNoCase("Station")==0)
    {
        return Station;
    }
    else if (SpatialAggregation.CmpNoCase("Groupment")==0)
    {
        return Groupment;
    }
    else if (SpatialAggregation.CmpNoCase("Catchment")==0)
    {
        return Catchment;
    }
    else
    {
        asLogError(wxString::Format(_("The SpatialAggregation enumeration (%s) entry doesn't exists"), SpatialAggregation.c_str()));
    }
    return NoDataSpatialAggregation;
}

wxString asGlobEnums::DataSpatialAggregationEnumToString(DataSpatialAggregation dataSpatialAggregation)
{
	switch (dataSpatialAggregation)
	{
		case(Station):
			return "Station";
		case(Groupment):
			return "Groupment";
		case(Catchment):
			return "Catchment";
		default:
        asLogError(_("The given data spatial aggregation type in unknown."));
	}
	return wxEmptyString;
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
