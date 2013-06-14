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
 
#ifndef ASGLOBENUMS_H_INCLUDED
#define ASGLOBENUMS_H_INCLUDED

#include "wx/wxprec.h"
#ifndef WX_PRECOMP
    #include "wx/wx.h"
#endif

//---------------------------------
// Symbolic constants
//---------------------------------

enum
{
    asSHOW_WARNINGS,
    asHIDE_WARNINGS
};

enum
{
    asOUT_OF_RANGE = -2,
    asNOT_FOUND = -1,
    asFAILED = 0,
    asEMPTY = -1,
    asNONE = -1,
    asCANCELLED = -1,
    asNOT_VALID = -9999,
    asSUCCESS = 1
};

enum
{
    asEDIT_FORBIDEN,
    asEDIT_ALLOWED
};

enum
{
    asFLAT_FORBIDDEN,
    asFLAT_ALLOWED
};

enum
{
    asUSE_NORMAL_METHOD,
    asUSE_ALTERNATE_METHOD
};

enum
{
    asUTM,
    asLOCAL
};

enum
{
    asSERIE_BEGINNING,
    asSERIE_END
};

enum
{
    asSAMPLE,
    asENTIRE_POPULATION
};

enum
{
    asMAKE_SQRT,
    asDONT_MAKE_SQRT
};

enum
{
    asCOEFF, // Direct access to the coefficients
    asCOEFF_NOVAR, // Direct access to the coefficients without any other variable declaration
    asLIN_ALGEBRA,	// Linear algebra using a library
    asLIN_ALGEBRA_NOVAR	// Linear algebra using a library without any other variable declaration
};

// Processor methods
enum
{
    asMULTITHREADS = 0,
    asINSERT = 1,
    asFULL_ARRAY = 2
};

// Windows ID
#if wxUSE_GUI
enum
{
    asWINDOW_MAIN = 101,
    asWINDOW_PREFERENCES = 102,
    asWINDOW_PREDICTANDDB = 104,
    asWINDOW_VIEWER_RINGS = 106,
    asWINDOW_VIEWER_DOTS = 107,
    asWINDOW_PLOTS_TIMESERIES = 108,
    asWINDOW_PLOTS_DISTRIBUTIONS = 109,
    asWINDOW_GRID_ANALOGS = 110,
    asWINDOW_PREDICTORS = 111
};
#endif

// Menus & Controls ID
enum
{
    asID_PREFERENCES = wxID_HIGHEST+1,
    asID_OPEN = wxID_HIGHEST+2,
    asID_RUN = wxID_HIGHEST+3,
    asID_RUN_PREVIOUS = wxID_HIGHEST+4,
    asID_CANCEL = wxID_HIGHEST+5,
    asID_DB_OPTIONS = wxID_HIGHEST+6,
    asID_DB_CREATE = wxID_HIGHEST+7,
    asID_PRINT = wxID_HIGHEST+9,
    asID_SELECT = wxID_HIGHEST+10,
    asID_ZOOM_IN = wxID_HIGHEST+11,
    asID_ZOOM_OUT = wxID_HIGHEST+12,
    asID_ZOOM_FIT = wxID_HIGHEST+13,
    asID_PAN = wxID_HIGHEST+14,
    asID_CROSS_MARKER = wxID_HIGHEST+15,
    asID_FRAME_VIEWER = wxID_HIGHEST+16,
    asID_FRAME_FORECASTER = wxID_HIGHEST+17,
    asID_FRAME_DOTS = wxID_HIGHEST+18,
    asID_FRAME_PLOTS = wxID_HIGHEST+19,
    asID_FRAME_GRID = wxID_HIGHEST+20,
    asID_FRAME_PREDICTORS = wxID_HIGHEST+21,
};


//---------------------------------
// Enumerations
//---------------------------------

enum Order
{
    Asc,	// Ascendant
    Desc,	// Descendant
    NoOrder
};

enum CoordSys //!< Enumaration of managed coordinate systems
{
    WGS84,	// World Geodetic System 1984
    CH1903,	// Former swiss projection
    CH1903p,// New swiss projection
    NoCoordSys
};

enum Season
{
    DJF,    //  Winter
    MAM,    // Spring
    JJA,    // Summer
    SON,    // Fall
    NoSeason
};

enum PredictandDB
{
    StationsDailyPrecipitation,
    Stations6HourlyPrecipitation,
    Stations6HourlyOfDailyPrecipitation,
    RegionalDailyPrecipitation,
    ResearchDailyPrecipitation,
    StationsDailyLightnings,
    NoPredictandDB
};

enum DataPurpose
{
    PredictorArchive,
    PredictorRealtime,
    Predictand,
    NoDataPurpose
};

enum DataParameter
{
    Precipitation,
    AirTemperature,
    GeopotentialHeight,
    PrecipitableWater,
    RelativeHumidity,
    SpecificHumidity,
    Omega,
    Uwind,
    Vwind,
    SurfaceLiftedIndex,
    PotentialTemperature,
    Pressure,
    PotentialEvaporation,
    SurfaceTemperature,
    ConvectivePrecipitation,
    LongwaveRadiation,
    ShortwaveRadiation,
    SolarRadiation,
    GroundHeatFlux,
    LatentHeatFlux,
    NearIRFlux,
    SensibleHeatFlux,
    Lightnings,
    SeaSurfaceTemperature,
    SeaSurfaceTemperatureAnomaly,
    NoDataParameter
};

enum DataUnit
{
    nb,
    mm,
    m,
    km,
    percent,
    degC,
    degK,
    Pascals,
    PascalsPerSec,
    kgPerKg,
    mPerSec,
    WPerm2,
    kgPerm2Pers,
    NoDataUnit
};

enum FileFormat
{
    dat,
    asc,
    txt,
    netcdf,
    grib,
    grib2,
    xml,
    kml,
    NoFileFormat
};

enum FileTimeLength
{
    Year,
    Hour,
    Total,
    Instantaneous,
    NoFileLength
};

enum TimeFormat
{
    classic,
    DDMMYYYY,
    YYYYMMDD,
    full,
    YYYYMMDDhh,
    DDMMYYYYhhmm,
    YYYYMMDDhhmm,
    DDMMYYYYhhmmss,
    YYYYMMDDhhmmss,
    timeonly,
    hhmm,
    nowplushours,
    nowminushours,
    concentrate,
    guess
};

enum ParametersList //!< Define available parameters sets (for the GUI)
{
    AnalogsNbMulti,
    AreaMoving
};


#include <asIncludes.h>

class asGlobEnums: public wxObject
{
public:


    static FileFormat StringToFileFormatEnum(const wxString &FormatStr);

    static wxString FileFormatEnumToExtension(FileFormat format);


    static FileTimeLength StringToFileLengthEnum(const wxString &LengthStr);


    static DataParameter StringToDataParameterEnum(const wxString &ParameterStr);


    static DataUnit StringToDataUnitEnum(const wxString &UnitStr);


    /** Transform a string to the corresponding CoordSys enum entry
     * \param CoordSysStr The entry to match
     * \return The corresponding CoordSys enum entry
     */
    static CoordSys StringToCoordSysEnum(const wxString &CoordSysStr);



    static PredictandDB StringToPredictandDBEnum(const wxString &PredictandDBStr);


    static wxString PredictandDBEnumToString(PredictandDB predictandDB);

protected:

private:

};

#endif // ASGLOBENUMS_H_INCLUDED

