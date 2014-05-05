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
    asCOEFF, // Direct access to the coefficients
    asCOEFF_NOVAR, // Direct access to the coefficients without any other variable declaration
    asLIN_ALGEBRA,    // Linear algebra using a library
    asLIN_ALGEBRA_NOVAR    // Linear algebra using a library without any other variable declaration
};

// Processor methods
enum
{
    asMULTITHREADS = 0,
    asINSERT = 1,
    asFULL_ARRAY = 2,
    asCUDA = 3,
};

// Optimization stages
enum
{
    asINITIALIZATION,
    asSORTING_AND_REFLECTION,
    asEVALUATE_REFLECTION,
    asCOMPARE_EXPANSION_REFLECTION,
    asEVALUATE_EXTERNAL_CONTRACTION,
    asEVALUATE_INTERNAL_CONTRACTION,
    asPROCESS_REDUCTION,
    asREASSESSMENT,
    asFINAL_REASSESSMENT,
    asCREATE_ROUGH_MAP,
    asEXPLORE_ROUGH_MAP,
    asCREATE_FINE_MAP,
    asEXPLORE_FINE_MAP,
    asRESIZE_DOMAIN,
    asOPTIMIZE_WEIGHTS,
    asOPTIMIZE_ANALOGSNB_STEP,
    asOPTIMIZE_ANALOGSNB_FINAL,
    asOPTIMIZE_DAYS_INTERVAL,
    asCHECK_CONVERGENCE
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
    };
#endif


//---------------------------------
// Enumerations
//---------------------------------

enum Order
{
    Asc,    // Ascendant
    Desc,    // Descendant
    NoOrder
};

enum CoordSys //!< Enumaration of managed coordinate systems
{
    WGS84,    // World Geodetic System 1984
    CH1903,    // Former swiss projection
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
    Wind,
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

enum DataTemporalResolution
{
    Daily,
    SixHourly,
    Hourly,
    SixHourlyMovingDailyTemporalWindow,
    TwoDays,
    ThreeDays,
    Weekly,
    NoDataTemporalResolution
};

enum DataSpatialAggregation
{
    Station,
    Groupment,
    Catchment,
    NoDataSpatialAggregation
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

enum Mode //!< Define frame mode
{
    Standard,
    Calibration,
    Optimization
};

#include <asIncludes.h>

class asGlobEnums: public wxObject
{
public:


    static FileFormat StringToFileFormatEnum(const wxString &FormatStr);

    static wxString FileFormatEnumToExtension(FileFormat format);


    static FileTimeLength StringToFileLengthEnum(const wxString &LengthStr);


    static DataParameter StringToDataParameterEnum(const wxString &ParameterStr);

    static wxString DataParameterEnumToString(DataParameter dataParameter);


    static DataUnit StringToDataUnitEnum(const wxString &UnitStr);

    static DataTemporalResolution StringToDataTemporalResolutionEnum(const wxString &TemporalResolution);
    
    static wxString DataTemporalResolutionEnumToString(DataTemporalResolution dataTemporalResolution);

    static DataSpatialAggregation StringToDataSpatialAggregationEnum(const wxString &SpatialAggregation);
    
    static wxString DataSpatialAggregationEnumToString(DataSpatialAggregation dataSpatialAggregation);

    /** Transform a string to the corresponding CoordSys enum entry
     * \param CoordSysStr The entry to match
     * \return The corresponding CoordSys enum entry
     */
    static CoordSys StringToCoordSysEnum(const wxString &CoordSysStr);


protected:

private:

};

#endif // ASGLOBENUMS_H_INCLUDED

